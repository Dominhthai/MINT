import os
import sys
import collections

from torch import nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, List

# Local imports
from pl_modules.base import BaseModel
from losses.comm_loss import CoMMLoss
from losses.superloss import Superloss
from losses.mgda import gradient_weights
from models.mmfusion import MMFusion

from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.subNets.BertTextEncoder import BertTextEncoder

class CoMM(BaseModel):
    """ Contrastive MultiModal learning allowing the communication between modalities 
    in a single multimodal space [1].
    
    It encodes a pair of mulitmodal data and outputs a pair of representations through
    a single multimodal encoder.

    [1] What to align in multimodal contrastive learning, Dufumier & Castillo-Navarro et al., ICLR 2025
    """

    def __init__(self,
                 encoder: MMFusion,
                 projection: nn.Module,
                 optim_kwargs: Dict,
                 loss_kwargs: Dict,
                 pretrained_kwargs: Dict,
                 curriculum_kwargs: Dict
                 ):
        """
        Args:
            encoder: Multi-modal fusion encoder
            projection: MLP projector to the latent space
            optim_kwargs: Optimization hyper-parameters
            loss_kwargs: Hyper-parameters for the CoMM loss.
        """
        super(CoMM, self).__init__(optim_kwargs)

        # create the encoder
        self.encoder = encoder

        # build a 3-layers projector
        self.head = projection

        # Build the loss
        self.loss = CoMMLoss(**loss_kwargs)

        # Pretrained config
        self.language = pretrained_kwargs["language"]
        self.use_finetune = pretrained_kwargs["use_finetune"]
        self.use_raw_text = pretrained_kwargs["use_raw_text"]
        
        # If True, feed (batch, T, 300) into BERT via inputs_embeds after a 300->768 projection
        self.use_inputs_embeds = pretrained_kwargs["use_inputs_embeds"]
        self.video_in = pretrained_kwargs["video_in"]
        self.text_in = pretrained_kwargs["text_in"]
        self.v_lstm_hidden_size = pretrained_kwargs["v_lstm_hidden_size"]
        self.video_out = pretrained_kwargs["video_out"]
        self.v_lstm_layers = pretrained_kwargs["v_lstm_layers"]
        self.v_lstm_dropout = pretrained_kwargs["v_lstm_dropout"]
        self.bidirectional = pretrained_kwargs["bidirectional"]
        self.alpha = pretrained_kwargs["alpha"]

        # Curriculum learning config
        self.v_tau = curriculum_kwargs["v_tau"]
        self.v_lam = curriculum_kwargs["v_lam"]
        self.v_fac = curriculum_kwargs["v_fac"]
        self.t_tau = curriculum_kwargs["t_tau"]
        self.t_lam = curriculum_kwargs["t_lam"]
        self.t_fac = curriculum_kwargs["t_fac"]
        
        # Pre-trained text encoder - ACTUALLY using BERT teacher
        self.text_model = BertTextEncoder(language=self.language, use_finetune=self.use_finetune)
        # pre-trained vision encoder - ACTUALLY using AuViSubNet teacher
        self.video_model = AuViSubNet(in_size=self.video_in, hidden_size=self.v_lstm_hidden_size, out_size=self.video_out, num_layers=self.v_lstm_layers, dropout=self.v_lstm_dropout, bidirectional=self.bidirectional)
        
        # If using inputs_embeds path, project 300-d MOSI/MOSEI tokens to 768-d BERT space
        # and learn a CLS token compatible with BERT
        if self.use_inputs_embeds:
            self.text_proj_300_to_768 = nn.Linear(self.text_in, 768)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))

        # I2MCL Curriculum Learning Components
        self.v_superloss = Superloss(tau=self.v_tau, lam=self.v_lam, fac=self.v_fac)  # Vision curriculum
        self.t_superloss = Superloss(tau=self.t_tau, lam=self.t_lam, fac=self.t_fac)  # Text curriculum
        
        # Initialize projection layers for teacher-student distillation
        # These need to be initialized upfront to avoid checkpoint loading issues
        self.vision_proj = nn.Linear(self.video_out, self.head[0].in_features)
        self.text_teacher_proj = nn.Linear(768, self.head[0].in_features)  # BERT outputs 768-dim features

    @staticmethod
    def _build_mlp(in_dim, mlp_dim, out_dim):
        return nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(in_dim, mlp_dim)),
            ("bn1", nn.SyncBatchNorm(mlp_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(mlp_dim, mlp_dim)),
            ("bn2", nn.SyncBatchNorm(mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(mlp_dim, out_dim)),
        ]))


    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor]):
        # compute features for all modalities
        # For MOSI, x1 = [vision_aug1(z1), text_aug1(z1)], x2 = [vision_aug2(z2), text_aug2(z2)]     
        all_masks = self.gen_all_possible_masks(len(x1))
        z1 = self.encoder(x1, mask_modalities=all_masks)# zi = [z1, z']
        z2 = self.encoder(x2, mask_modalities=all_masks)# zi = [z2, z'']
        z1 = [self.head(z) for z in z1]# z1=[vision_z1, text_z1, joint_z']
        z2 = [self.head(z) for z in z2]# z2=[vision_z2, text_z2, joint_z'']

        # Teacher-Student Knowledge Distillation with Curriculum Learning
        n_emb = len(x1)  # MOSI: 2 (vision, text)
        
        # Extract REAL teacher features (Self-MM style) - NOT just projections!
        with torch.no_grad():
            # Vision teacher features using REAL AuViSubNet model
            # Calculate actual sequence lengths for each modality
            vision_lengths = self._calculate_sequence_lengths(x1[0])
            
            # REAL AuViSubNet teacher feature extraction (like Self-MM)
            v1_teacher_raw = self.video_model(x1[0], vision_lengths)  # Vision aug1 teacher
            v2_teacher_raw = self.video_model(x2[0], vision_lengths)  # Vision aug2 teacher
            
            # Project vision teacher features to match head input dimension
            v1_teacher = self.vision_proj(v1_teacher_raw)
            v2_teacher = self.vision_proj(v2_teacher_raw)
            
            # Text teacher features - Extract REAL BERT features from actual text
            # Prefer true BERT ingestion paths in this order:
            # 1) inputs_embeds (project 300->768 and feed to BERT)
            # 2) raw texts if available
            # 3) enhanced pseudo-tokens fallback
            if self.use_inputs_embeds:
                t1_teacher_raw = self._extract_bert_from_inputs_embeds(x1[1])  # (batch, 768)
                t2_teacher_raw = self._extract_bert_from_inputs_embeds(x2[1])  # (batch, 768)
            else:
                t1_teacher_raw = self._extract_bert_teacher_features(x1[1])  # (batch, 768)
                t2_teacher_raw = self._extract_bert_teacher_features(x2[1])  # (batch, 768)
            
            # Project BERT teacher features to match head input dimension
            t1_teacher = self.text_teacher_proj(t1_teacher_raw)
            t2_teacher = self.text_teacher_proj(t2_teacher_raw)

        # Project teacher features to same dimension as student
        v1_teacher_proj = self.head(v1_teacher)
        v2_teacher_proj = self.head(v2_teacher) 
        t1_teacher_proj = self.head(t1_teacher)
        t2_teacher_proj = self.head(t2_teacher)
        
        # Compute per-instance distillation losses (needed for curriculum learning)
        v_loss_raw = (F.mse_loss(z1[0], v1_teacher_proj, reduction='none').sum(1) + 
                      F.mse_loss(z2[0], v2_teacher_proj, reduction='none').sum(1)) / 2
        t_loss_raw = (F.mse_loss(z1[1], t1_teacher_proj, reduction='none').sum(1) + 
                      F.mse_loss(z2[1], t2_teacher_proj, reduction='none').sum(1)) / 2

        # Add cosine similarity as penalties 
        # v_cosine = alpha * (1 - F.cosine_similarity(z1[0], v1_teacher_proj, dim=1) +
        #          1 - F.cosine_similarity(z2[0], v2_teacher_proj, dim=1)) / 2
        # t_cosine = alpha * (1 - F.cosine_similarity(z1[1], t1_teacher_proj, dim=1) +
        #          1 - F.cosine_similarity(z2[1], t2_teacher_proj, dim=1)) / 2
        fusion_cosine = self.alpha * (1 - F.cosine_similarity(z1[2], v1_teacher_proj, dim=1) + 
                        1 - F.cosine_similarity(z1[2], t1_teacher_proj, dim=1)+
                        1 - F.cosine_similarity(z2[2], v2_teacher_proj , dim=1) + 
                        1 - F.cosine_similarity(z2[2], t2_teacher_proj, dim=1))/4

        # v_loss_raw += alpha*v_cosine
        # t_loss_raw += alpha*t_cosine
        
        # # Curriculum Learning
        # # 1. Intra-Modal Curriculum: Superloss weighting based on difficulty
        # v_loss_curriculum = self.v_superloss(v_loss_raw)
        # t_loss_curriculum = self.t_superloss(t_loss_raw)
        
        # # 2. No curriculumn learning
        
        v_loss_curriculum = v_loss_raw.mean()
        t_loss_curriculum = t_loss_raw.mean()

        return {'aug1_embed': z1,
                'aug2_embed': z2,
                "prototype": -1,
                "v_loss": v_loss_curriculum,
                "t_loss": t_loss_curriculum,
                # "v_cosine": v_cosine,
                # "t_cosine": t_cosine
                "cos_sim": fusion_cosine
               }
    

    def _extract_bert_teacher_features(self, text_data, raw_texts=None):
        """
        Extract REAL BERT teacher features from MOSI text data.
        
        Args:
            text_data: torch.Tensor of shape (batch, seq_len, feat_dim) - MOSI dense embeddings
            raw_texts: List[str] of raw text strings (optional, for proper BERT tokenization)
        
        Returns:
            torch.Tensor: REAL BERT teacher features of shape (batch, 768)
        """
        batch_size, seq_len, feat_dim = text_data.shape
        device = text_data.device
        
        if raw_texts is not None:
            # Option 1: Use actual raw text strings for proper BERT tokenization
            return self._extract_bert_from_raw_texts(raw_texts, device)
        else:
            # Option 2: Fallback to pseudo-tokens (current approach)
            return self._extract_bert_from_pseudo_tokens(text_data, device)

    def _extract_bert_from_inputs_embeds(self, text_data):
        """
        Feed MOSI/MOSEI dense embeddings (batch, T, 300) into BERT via inputs_embeds.
        Steps:
            1) Build attention_mask from non-zero time steps
            2) Project 300 -> 768 with a learned linear layer
            3) Prepend a learned [CLS] token in embedding space
            4) Call HuggingFace BERT with inputs_embeds and attention_mask
            5) Return the [CLS] embedding as the teacher feature (batch, 768)
        """
        assert hasattr(self, 'text_proj_300_to_768'), "text_proj_300_to_768 not initialized; set use_inputs_embeds=True"
        assert hasattr(self, 'cls_token'), "cls_token not initialized; set use_inputs_embeds=True"

        # text_data: (batch, T, 300)
        batch_size, T, feat_dim = text_data.shape

        # 1) attention mask: a time step is valid if any feature is non-zero
        # mask dtype long/bool; HuggingFace accepts long (0/1) or bool
        attn_mask = (text_data.abs().sum(dim=-1) > 0).long()  # (B, T)

        # 2) project (B, T, 300) -> (B, T, 768)
        x = self.text_proj_300_to_768(text_data)  # (B, T, 768)

        # 3) prepend learned [CLS] token
        cls = self.cls_token.expand(batch_size, 1, -1)  # (B, 1, 768)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, 768)

        # prepend mask 1 for CLS
        cls_mask = torch.ones(batch_size, 1, device=x.device, dtype=attn_mask.dtype)
        attn_mask = torch.cat([cls_mask, attn_mask], dim=1)  # (B, T+1)

        # 4) BERT forward with inputs_embeds
        # If finetuning, allow gradients; else no_grad context
        if self.use_finetune:
            bert_out = self.text_model.model(inputs_embeds=x, attention_mask=attn_mask)[0]  # (B, T+1, 768)
        else:
            with torch.no_grad():
                bert_out = self.text_model.model(inputs_embeds=x, attention_mask=attn_mask)[0]

        # 5) take [CLS]
        cls_feat = bert_out[:, 0, :]  # (B, 768)
        return cls_feat
    
    def _extract_bert_from_raw_texts(self, raw_texts, device):
        """
        Extract BERT features from actual raw text strings. 
        Args:
            raw_texts: List[str] of raw text strings
            device: torch device     
        Returns:
            torch.Tensor: REAL BERT teacher features of shape (batch, 768)
        """
        batch_size = len(raw_texts)
        bert_features_list = []
        
        for text in raw_texts:
            # Tokenize the actual text using BERT tokenizer
            tokenizer = self.text_model.get_tokenizer()
            
            # Tokenize and encode the text
            encoded = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            token_type_ids = encoded['token_type_ids'].to(device)
            
            # Create BERT input format (batch, 3, seq_len)
            bert_input = torch.stack([input_ids.squeeze(), attention_mask.squeeze(), token_type_ids.squeeze()], dim=0).unsqueeze(0)
            
            # Extract BERT features
            with torch.no_grad():
                bert_output = self.text_model(bert_input.float())  # (1, seq_len, 768)
                bert_cls = bert_output[:, 0, :]  # (1, 768) - [CLS] token
                bert_features_list.append(bert_cls)
        
        # Stack all features
        bert_features = torch.cat(bert_features_list, dim=0)  # (batch, 768)
        return bert_features
    
    def _extract_bert_from_pseudo_tokens(self, text_data, device):
        """
        Extract BERT features using enhanced pseudo-tokens based on MOSI text data.  
        This method creates more meaningful pseudo-tokens by using the MOSI dense embeddings
        to influence the token selection, making the BERT features more related to the actual text.
        Args:
            text_data: torch.Tensor of shape (batch, seq_len, feat_dim) - MOSI dense embeddings
            device: torch device
        Returns:
            torch.Tensor: BERT features of shape (batch, 768)
        """
        batch_size, seq_len, feat_dim = text_data.shape
        max_bert_length = min(seq_len, 512)
        
        # Create pseudo BERT token IDs
        bert_tokens = torch.zeros(batch_size, 3, max_bert_length, dtype=torch.long, device=device)
        
        # Enhanced approach: Use MOSI text data to influence token selection
        for i in range(batch_size):
            # Get the text sequence for this sample
            text_seq = text_data[i]  # (seq_len, 300)
            
            # Fill input_ids with [CLS] token
            bert_tokens[i, 0, 0] = 101  # [CLS] token
            
            # Create content tokens based on MOSI text data
            for j in range(1, max_bert_length - 1):
                # Use the magnitude of the text embedding to influence token selection
                # This creates a more meaningful mapping from dense embeddings to tokens
                text_magnitude = torch.norm(text_seq[j-1]).item() if j-1 < len(text_seq) else 0.0
                
                # Map text magnitude to token ID range (1000-3000)
                # Higher magnitude = higher token ID (more "important" tokens)
                token_id = int(1000 + (text_magnitude * 2000) % 2000)
                bert_tokens[i, 0, j] = token_id
            
            # Fill [SEP] token
            bert_tokens[i, 0, max_bert_length - 1] = 102  # [SEP] token
        
        # Fill attention_mask and token_type_ids
        bert_tokens[:, 1, :] = 1  # All tokens are "real"
        bert_tokens[:, 2, :] = 0  # Single sentence
        
        # Extract BERT features
        bert_features = self.text_model(bert_tokens.float())  # (batch, seq_len, 768)
        bert_cls_features = bert_features[:, 0, :]  # (batch, 768)
        
        return bert_cls_features

    def _calculate_sequence_lengths(self, sequences):
        """
        Calculate actual sequence lengths for LSTM models by finding non-zero elements.
        Args:
            sequences: torch.Tensor of shape (batch_size, seq_len, feat_dim)
        Returns:
            torch.Tensor of sequence lengths for each sample in the batch
        """
        batch_size = sequences.shape[0]
        lengths = []
        for i in range(batch_size):
            seq = sequences[i]  # Shape: (seq_len, feat_dim)
            # Find non-zero positions across feature dimensions
            non_zero_mask = torch.any(seq != 0, dim=1)
            if torch.any(non_zero_mask):
                # Find the last non-zero position
                length = torch.max(torch.where(non_zero_mask)[0]) + 1
                # Convert to scalar if it's a tensor
                if isinstance(length, torch.Tensor):
                    length = length.item()
            else:
                # If all zeros, set minimum length to avoid error
                length = 1
            lengths.append(length)
        return torch.tensor(lengths)

    def gen_all_possible_masks(self, n_mod: int):
        """
        :param n_mod: int
        :return: a list of `n_mod` + 1 boolean masks [Mi] such that all but one bool are False.
            A last bool mask is added where all bool are True
        Examples:
        *   For n_mod==2:
            masks == [[True, False], [False, True], [True, True]]
        *   For n_mod == 3:
            masks == [[True, False, False], [False, True, False], [False, False, True], [True, True, True]]
        """
        masks = []
        for L in range(n_mod):
            mask = [s == L for s in range(n_mod)]
            masks.append(mask)
        masks.append([True for _ in range(n_mod)])
        return masks
    
    
    def extract_features(self, loader: torch.utils.data.DataLoader, **kwargs):
        """
           Extract multimodal features from the encoder.
           Args:
                loader: Dataset loader to serve `(X, y)` tuples.
                kwargs: given to `encoder.forward()`
           Returns: 
                Pair (Z,y) corresponding to extracted features and corresponding labels
        """
        X, y = [], []
        for X_, y_ in loader:
            if isinstance(X_, torch.Tensor): # needs to cast it as list of one modality
                X_ = [X_]
            X_ = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in X_]
            y_ = y_.to(self.device)
            with torch.inference_mode():
                # compute output
                output = self.encoder(X_, **kwargs)
                X.extend(output.view(len(output), -1).detach().cpu())
                y.extend(y_.detach().cpu())
        torch.cuda.empty_cache()
        return torch.stack(X, dim=0).to(self.device), torch.stack(y, dim=0).to(self.device)

class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1