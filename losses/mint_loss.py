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


class MINTLoss(nn.Module):
    """
        Normalized Temperature Cross-Entropy Loss for Multi-Modal Contrastive Learning as defined in MINT [1]
    """

    def __init__(self, temperature=0.1, weights=None, curriculum_weight: bool=False):
        super().__init__()
        self.temperature = temperature
        self.weights = weights
        if not isinstance(curriculum_weight, bool):
            raise TypeError("MINTLoss.curriculum_weight must be a boolean (True/False)")
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
        # For MOSI, z1 = [vision_z1, text_z1, joint_z'], z2 = [vision_z2, text_z2, joint_z'']
        z1, z2, prototype, vision_loss, text_loss, cos_sim = outputs["aug1_embed"], outputs["aug2_embed"], outputs["prototype"], outputs["v_loss"], outputs["t_loss"], outputs["cos_sim"]
        assert len(z1) == len(z2)
        n_emb = len(z1) # MOSI:3
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
        curriculum_losses = [vision_loss, text_loss]
        
        # Combine SSL contrastive losses with curriculum losses
        combined_losses = [
            (loss[0] + curriculum_losses[0])/2,  # Vision SSL + Vision curriculum
            (loss[1] + curriculum_losses[1])/2,  # Text SSL + Text curriculum  
            (loss[2] + self.curriculum_weight * cos_sim.mean())/2 # Joint SSL (no curriculum)
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

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)