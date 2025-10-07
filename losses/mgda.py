import torch


def gradient_weights(grad1, grad2):
    """
    Compute Pareto optimal weights for gradient-based curriculum learning.
    From I2MCL paper - determines which objective each modality should focus on.
    
    Args:
        grad1: Gradients from first objective (e.g., distillation loss)
        grad2: Gradients from second objective (e.g., task loss)
        
    Returns:
        gamma: Weight in [0,1] determining objective preference
               gamma ≈ 0: Focus on objective 2 (task)
               gamma ≈ 1: Focus on objective 1 (distillation)
    """
    v1v1, v1v2, v2v2 = 0.0, 0.0, 0.0
    
    # Compute gradient relationships
    for g1, g2 in zip(grad1, grad2):
        v1v1 += torch.mul(g1, g1).sum().item()  # ||∇L₁||²
        v1v2 += torch.mul(g1, g2).sum().item()  # ∇L₁ · ∇L₂ 
        v2v2 += torch.mul(g2, g2).sum().item()  # ||∇L₂||²

    # Pareto optimal weight decision
    if v1v2 >= v2v2:
        # Gradients aligned → Focus on task (objective 2)
        gamma = 0.001
        cost = v2v2
    elif v1v2 >= v1v1:
        # Strong alignment → Focus on distillation (objective 1)
        gamma = 0.999
        cost = v1v1
    else:
        # Conflicting gradients → Pareto optimal solution
        gamma = (v2v2 - v1v2) / max(v1v1 + v2v2 - 2.0 * v1v2, 1e-8)
        cost = v2v2 + gamma * (v1v2 - v2v2)
    
    return gamma


def gradient_norm(grad, loss, type):
    """Compute gradient norm for analysis"""
    norm = 0.0
    for g in grad:
        norm += torch.mul(g, g).sum().item()
    return norm
