%%writefile /kaggle/working/CoMM/losses/comm_loss.py
import os
import sys
import collections

import torch.nn.functional as func
import torch
import torch.nn as nn
from utils import all_gather_batch_with_grad

# from torch.autograd.function import Function
# from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# from models.subNets.BertTextEncoder import BertTextEncoder


class CoMMLoss(nn.Module):
    """
        Normalized Temperature Cross-Entropy Loss for Multi-Modal Contrastive Learning as defined in CoMM [1]

        [1] What to align in multimodal contrastive learning, Dufumier & Castillo-Navarro et al., ICLR 2025
    """

    def __init__(self, temperature=0.1, weights=None, curriculum_weight: bool=False):
        super().__init__()
        self.temperature = temperature
        self.weights = weights
        if not isinstance(curriculum_weight, bool):
            raise TypeError("CoMMLoss.curriculum_weight must be a boolean (True/False)")
        self.curriculum_weight = 1.0 if curriculum_weight else 0.0
        
        self.INF = 1e8
        
    # Always remember that this will compare/estimate mutual information across all batches
    def infonce(self, z1, z2):
        N = len(z1)
        sim_zii= (z1 @ z1.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z2 @ z2.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z1 @ z2.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z1.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z1.device)
        sim_Z = torch.cat([
            torch.cat([sim_zij, sim_zii], dim=1),
            torch.cat([sim_zjj, sim_zij.T], dim=1)], dim=0)
        log_sim_Z = func.log_softmax(sim_Z, dim=1)
        loss = - torch.diag(log_sim_Z).mean()
        
        # compute SSL accuracy
        with torch.no_grad():
            pred = torch.argmax(sim_zij, dim=1)
            correct = pred.eq(torch.arange(N, device=z1.device)).sum()
            acc = 100 * correct / N
        return loss, acc

    def forward(self, outputs):
        """
        :param outputs: Dict
            Dictionary with keys:
                - "aug1_embed", List of tensors with shape (bsize, feature_dim), 1st aug.
                - "aug2_embed", List of tensors with shape (bsize, feature_dim), 2nd aug.
                - "prototype", integer indicating where the multimodal representation Z 
                    is stored in "aug1_embed" and "aug2_embed".
        :return: {"loss": torch.Tensor(float), "ssl_acc": torch.Tensor(float)}
        """
        # Prepare embeddings (normalize + gather across all GPU)
        # For 3-modality MOSEI, z1 = [vision_z1, text_z1, audio_z1, joint_z'], z2 = [vision_z2, text_z2, audio_z2, joint_z'']
        z1, z2, prototype, vision_loss, text_loss, audio_loss, cos_sim = outputs["aug1_embed"], outputs["aug2_embed"], outputs["prototype"], outputs["v_loss"], outputs["t_loss"], outputs["a_loss"], outputs["cos_sim"]
        assert len(z1) == len(z2)
        n_emb = len(z1) # 3-modality: 4 (vision, text, audio, joint)
        z1 = [func.normalize(z, p=2, dim=-1) for z in z1]
        z2 = [func.normalize(z, p=2, dim=-1) for z in z2]
        Z = all_gather_batch_with_grad(z1 + z2)
        z1, z2 = Z[:n_emb], Z[n_emb:]

        # Apply InfoNCE between a "prototype embedding" and all the others
        loss = []
        acc = []
        for i in range(n_emb):
            # Estimate InfoNCE loss(I(zi, z') and I(zi, z'')) between each modality i and multimodal z1[prototype](z') and z2[prototype](z'')- See formula (7)
            # Mutual information is the loss.
            loss1, acc1 = self.infonce(z1[i], z2[prototype])
            loss2, acc2 = self.infonce(z2[i], z1[prototype])
            loss.append((loss1 + loss2) / 2.)
            acc.append((acc1 + acc2) / 2.)
        
        # Add curriculum learning losses 
        curriculum_losses = [vision_loss, text_loss, audio_loss]
        
        # Combine SSL contrastive losses with curriculum losses
        combined_losses = [
            (loss[0] + curriculum_losses[0])/2,  # Vision SSL + Vision curriculum
            (loss[1] + curriculum_losses[1])/2,  # Text SSL + Text curriculum  
            (loss[2] + curriculum_losses[2])/2,  # Audio SSL + Audio curriculum
            (loss[3] + self.curriculum_weight * cos_sim.mean())/2 # Joint SSL (no curriculum)
        ]
        
        ssl_acc = {"ssl_acc_%i"%i: acc_ for i, acc_ in enumerate(acc)}
        ssl_losses = {"ssl_loss_%i"%i: l for i, l in enumerate(combined_losses)}
        # curriculum_metrics = {"v_curriculum_loss": vision_loss, "t_curriculum_loss": text_loss}
        
        if self.weights is not None:
            # Apply weights to SSL losses only, curriculum losses keep their own weight
            weighted_ssl = torch.mean(torch.stack(loss) * torch.tensor(self.weights[:len(loss)], device=z1[0].device))
            total_loss = weighted_ssl + torch.mean(torch.stack(curriculum_losses))
        else:
            total_loss = torch.mean(torch.stack(combined_losses))
            
        avg_ssl_acc = torch.mean(torch.stack(acc))
        
        return {
            "loss": total_loss, 
            "ssl_acc": avg_ssl_acc, 
            **ssl_acc, 
            **ssl_losses,
            # **curriculum_metrics
        }

    def reconstruct_from_loss(self, loss, N: int, device=None):
        """
        Reconstruct one consistent set of similarity matrices from a scalar InfoNCE loss.

        Assumptions (non-unique reconstruction):
        - Uniform negatives with identical logit value b across each row
        - Diagonal positives share the same logit value a
        - Off-diagonals set to b=0 w.r.t. additive row-invariance of softmax

        Given L = -a + log(exp(a) + (2N-1)exp(b)) and setting b=0, we solve
            exp(a) = (2N-1) / (exp(L) - 1)

        Args:
            loss: scalar torch.Tensor or float (the InfoNCE loss value used in infonce)
            N: batch size used to form the block matrix (z1, z2)
            device: optional torch device for returned tensors

        Returns:
            dict with keys: 'log_sim_Z', 'sim_Z', 'sim_zii', 'sim_zjj', 'sim_zij', 'a'
        """
        if not torch.is_tensor(loss):
            loss = torch.tensor(float(loss))
        if device is None:
            device = loss.device

        # Solve for a assuming b=0 and 2N-1 negatives per row
        twoN_minus_1 = 2 * N - 1
        # numerical stability
        eps = 1e-12
        exp_a = twoN_minus_1 / (torch.exp(loss) - 1.0 + eps)
        a = torch.log(exp_a)

        # Build synthetic similarity blocks consistent with the InfoNCE construction
        sim_zij = torch.zeros((N, N), device=device)
        sim_zij[torch.arange(N), torch.arange(N)] = a
        sim_zii = torch.zeros((N, N), device=device)  # self-similarity block (diag masked during loss)
        sim_zjj = torch.zeros((N, N), device=device)

        sim_Z = torch.cat([
            torch.cat([sim_zij, sim_zii], dim=1),
            torch.cat([sim_zjj, sim_zij.T], dim=1)
        ], dim=0)

        log_sim_Z = func.log_softmax(sim_Z, dim=1)

        return {
            "log_sim_Z": log_sim_Z,
            "sim_Z": sim_Z,
            "sim_zii": sim_zii,
            "sim_zjj": sim_zjj,
            "sim_zij": sim_zij,
            "a": a
        }

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)