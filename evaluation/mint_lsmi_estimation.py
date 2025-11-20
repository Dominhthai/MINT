"""
LSMI Estimation for MINT Model
Adapted from "Efficient Quantification of Multimodal Interaction at Sample Level" (ICML 2025)

This module quantifies Redundancy (R), Uniqueness (U1, U2), and Synergy (S) 
in the multimodal representations learned by MINT.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List
import os


class MargKernel(nn.Module):
    """
    Marginal kernel density estimator for entropy estimation.
    Uses mixture of Gaussians with learned parameters.
    """
    def __init__(self, dim, init_samples=None):
        self.K = 5  # Number of Gaussian components
        self.d = dim
        self.use_tanh = True
        super(MargKernel, self).__init__()
        self.init_std = torch.tensor(1.0, dtype=torch.float32)
        self.logC = torch.tensor([-self.d / 2 * np.log(2 * np.pi)])

        init_samples = self.init_std * torch.randn(self.K, self.d)
        self.means = nn.Parameter(init_samples, requires_grad=True)  
        diag = self.init_std * torch.randn((1, self.K, self.d))
        self.logvar = nn.Parameter(diag, requires_grad=True)

        tri = self.init_std * torch.randn((1, self.K, self.d, self.d))
        tri = tri.to(init_samples.dtype)
        self.tri = nn.Parameter(tri, requires_grad=True)

        weigh = torch.ones((1, self.K))
        self.weigh = nn.Parameter(weigh, requires_grad=True)

    def logpdf(self, x):
        assert len(x.shape) == 2 and x.shape[1] == self.d, 'x has to have shape [N, d]'
        x = x[:, None, :]
        w = torch.log_softmax(self.weigh, dim=1)
        y = x - self.means
        logvar = self.logvar
        if self.use_tanh:
            logvar = logvar.tanh()
        var = logvar.exp()
        y = y * var
        y = y.to(self.tri.dtype)
        if self.tri is not None:
            y = y + torch.squeeze(torch.matmul(torch.tril(self.tri, diagonal=-1), y[:, :, :, None]), 3)
        y = torch.sum(y ** 2, dim=2)

        y = -y / 2 + torch.sum(torch.log(torch.abs(var) + 1e-8), dim=-1) + w
        y = torch.logsumexp(y, dim=-1)
        return self.logC.to(y.device) + y

    def forward(self, x):
        y = -self.logpdf(x)
        if self.training:
            return torch.mean(y)
        return y


class ClassifierNetwork(nn.Module):
    """Simple classifier for mutual information estimation."""
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super(ClassifierNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)


class FeatureDataset(Dataset):
    """Dataset for pre-extracted features."""
    def __init__(self, modal_1_features, modal_2_features, targets):
        self.modal_1_features = modal_1_features
        self.modal_2_features = modal_2_features
        self.targets = targets
        
    def __len__(self):
        return self.targets.size(0)
    
    def __getitem__(self, index):
        return self.modal_1_features[index], self.modal_2_features[index], self.targets[index]


def RUS_adjustment(rus):
    """
    Adjusts R, U1, U2, S to ensure non-negative values while preserving sums.
    
    Priority 1: Make R and S non-negative
    Priority 2: Make U1 and U2 non-negative
    
    Preserves:
    - (R + U1 + U2 + S) remains unchanged
    - (R + U1) remains unchanged  
    - (R + U2) remains unchanged
    """
    r_orig, u_1_orig, u_2_orig, s_orig = rus

    R_mean = r_orig.detach().mean()
    U1_mean = u_1_orig.detach().mean()
    U2_mean = u_2_orig.detach().mean()
    S_mean = s_orig.detach().mean()

    adj_factor = torch.tensor(0.0, dtype=R_mean.dtype, device=R_mean.device)

    # Priority 1: Address negative mean of r or s
    if R_mean < 0 or S_mean < 0:
        adj_factor = -torch.min(R_mean, S_mean)
          
    # Priority 2: If means of r and s are non-negative, address negative mean of u1 or u2
    elif U1_mean < 0 or U2_mean < 0:
        adj_factor = torch.min(U1_mean, U2_mean)

    r_adjusted = r_orig + adj_factor
    u_1_adjusted = u_1_orig - adj_factor
    u_2_adjusted = u_2_orig - adj_factor
    s_adjusted = s_orig + adj_factor
    
    return r_adjusted, u_1_adjusted, u_2_adjusted, s_adjusted


def train_discriminators(train_loader, modal_1_dim, modal_2_dim, n_classes, 
                         device='cuda', hidden_dim=128, num_epochs=50, lr=1e-3):
    """
    Train discriminator networks for mutual information estimation.
    
    Args:
        train_loader: DataLoader with (modal_1, modal_2, labels)
        modal_1_dim: Dimension of first modality features
        modal_2_dim: Dimension of second modality features
        n_classes: Number of classes
        device: Device to train on
        hidden_dim: Hidden dimension of networks
        num_epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        List of three trained classifier models [model_1, model_2, model_joint]
    """
    model_1 = ClassifierNetwork(modal_1_dim, hidden_dim, n_classes).to(device)
    model_2 = ClassifierNetwork(modal_2_dim, hidden_dim, n_classes).to(device)
    model_joint = ClassifierNetwork(modal_1_dim + modal_2_dim, hidden_dim, n_classes).to(device)
    
    models = [model_1, model_2, model_joint]
    optimizer = torch.optim.Adam([p for model in models for p in model.parameters()], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    print("Training discriminators for mutual information estimation...")
    for epoch in range(num_epochs):
        losses = 0.0
        num_samples = 0
        for batch in train_loader:
            modal_1, modal_2, labels = batch
            modal_1, modal_2, labels = modal_1.to(device), modal_2.to(device), labels.to(device)
            batch_size = modal_1.shape[0]
            
            out_1 = models[0](modal_1)
            out_2 = models[1](modal_2)
            out_joint = models[2](torch.cat([modal_1, modal_2], dim=1))
            
            optimizer.zero_grad()
            loss_1 = criterion(out_1, labels)
            loss_2 = criterion(out_2, labels)
            loss_joint = criterion(out_joint, labels)
            loss = loss_1 + loss_2 + loss_joint
            loss.backward()
            optimizer.step()
            
            losses += loss.item() * batch_size
            num_samples += batch_size
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'  Epoch [{epoch + 1}/{num_epochs}], Loss: {losses / num_samples:.4f}')
    
    return models


def train_entropy_estimators(train_loader, modal_1_dim, modal_2_dim, 
                             device='cuda', num_epochs=50, lr=1e-3):
    """
    Train entropy estimators for each modality.
    
    Args:
        train_loader: DataLoader with (modal_1, modal_2, labels)
        modal_1_dim: Dimension of first modality features
        modal_2_dim: Dimension of second modality features
        device: Device to train on
        num_epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        List of two trained entropy estimator models [estimator_1, estimator_2]
    """
    model_1 = MargKernel(dim=modal_1_dim).to(device)
    model_2 = MargKernel(dim=modal_2_dim).to(device)
    
    models = [model_1, model_2]
    optimizer = torch.optim.Adam([p for model in models for p in model.parameters()], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    print("Training entropy estimators...")
    for epoch in range(num_epochs):
        for model in models:
            model.train()
        losses = 0.0
        num_samples = 0
        for batch in train_loader:
            modal_1, modal_2, _ = batch
            modal_1, modal_2 = modal_1.to(device), modal_2.to(device)
            batch_size = modal_1.shape[0]
            
            loss_1 = model_1(modal_1)
            loss_2 = model_2(modal_2)
            loss = loss_1 + loss_2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses += loss.item() * batch_size
            num_samples += batch_size
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'  Epoch [{epoch + 1}/{num_epochs}], Loss: {losses / num_samples:.4f}')
    
    return models


def get_entropy(dataloader, entropy_model, modality_idx=0, device='cuda'):
    """Estimate entropy H(X) for a modality."""
    entropy_model.eval()
    entropies = []
    with torch.no_grad():
        for batch in dataloader:
            modal_1, modal_2, _ = batch
            modal_1, modal_2 = modal_1.to(device), modal_2.to(device)
            
            input_data = modal_1 if modality_idx == 0 else modal_2
            h = entropy_model(input_data)
            entropies.append(h)
    
    return torch.cat(entropies, dim=0).detach()


def get_mutual_info(dataloader, discriminator, modality='modal_1', n_classes=2, device='cuda'):
    """Estimate mutual information I(X;Y) using discriminator."""
    discriminator.eval()
    mi_values = []
    with torch.no_grad():
        for batch in dataloader:
            modal_1, modal_2, labels = batch
            modal_1, modal_2, labels = modal_1.to(device), modal_2.to(device), labels.to(device)
            
            if modality == 'modal_1':
                input_data = modal_1
            elif modality == 'modal_2':
                input_data = modal_2
            elif modality == 'joint':
                input_data = torch.cat([modal_1, modal_2], dim=1)
            
            batch_size = input_data.shape[0]
            rows = torch.arange(batch_size, device=device)
            out = discriminator(input_data)
            
            # I(X;Y) = log(n_classes) + E[log P(Y|X)]
            mi = np.log(n_classes) + torch.nn.Softmax(dim=1)(out)[rows, labels].log()
            mi_values.append(mi)
    
    return torch.cat(mi_values, dim=0).detach()


def estimate_LSMI(dataloader, discriminators, entropy_estimators, n_classes=2, device='cuda'):
    """
    Estimate LSMI decomposition: R, U1, U2, S
    
    Args:
        dataloader: DataLoader with extracted features
        discriminators: List of [disc_1, disc_2, disc_joint]
        entropy_estimators: List of [entropy_1, entropy_2]
        n_classes: Number of classes
        device: Device
        
    Returns:
        Tuple of (R, U1, U2, S) as scalars and per-sample tensors
    """
    print("\nEstimating LSMI components...")
    
    # Estimate mutual information
    I_X1Y = get_mutual_info(dataloader, discriminators[0], 'modal_1', n_classes, device)
    I_X2Y = get_mutual_info(dataloader, discriminators[1], 'modal_2', n_classes, device)
    I_X1X2Y = get_mutual_info(dataloader, discriminators[2], 'joint', n_classes, device)
    
    # Estimate entropy
    H_X1 = get_entropy(dataloader, entropy_estimators[0], modality_idx=0, device=device)
    H_X2 = get_entropy(dataloader, entropy_estimators[1], modality_idx=1, device=device)
    
    # Compute LSMI components
    r_plus = torch.minimum(H_X1, H_X2)
    r_minus = torch.minimum(H_X1 - I_X1Y, H_X2 - I_X2Y)
    r = r_plus - r_minus
    
    u_1 = I_X1Y - r
    u_2 = I_X2Y - r
    s = I_X1X2Y - r - u_1 - u_2
    
    # Adjust to ensure non-negative values
    r, u_1, u_2, s = RUS_adjustment([r, u_1, u_2, s])
    
    # Compute means
    R = torch.mean(r)
    U_1 = torch.mean(u_1)
    U_2 = torch.mean(u_2)
    S = torch.mean(s)
    
    print(f"\nLSMI Results:")
    print(f"  Redundancy (R):    {R.item():.4f}")
    print(f"  Uniqueness 1 (U1): {U_1.item():.4f}")
    print(f"  Uniqueness 2 (U2): {U_2.item():.4f}")
    print(f"  Synergy (S):       {S.item():.4f}")
    print(f"  Total: {(R + U_1 + U_2 + S).item():.4f}")
    
    return {
        'R_mean': R.item(),
        'U1_mean': U_1.item(),
        'U2_mean': U_2.item(),
        'S_mean': S.item(),
        'r_sample': r.cpu(),
        'u1_sample': u_1.cpu(),
        'u2_sample': u_2.cpu(),
        's_sample': s.cpu()
    }


def extract_mint_features(mint_model, dataloader, device='cuda'):
    """
    Extract features from MINT model for LSMI estimation.
    
    Extracts:
    - Unimodal features from individual encoders (before fusion)
    - Joint features from the full encoder (after fusion)
    
    Args:
        mint_model: Trained MINT model
        dataloader: DataLoader with multimodal data
        device: Device
        
    Returns:
        Dictionary with modal_1_features, modal_2_features, joint_features, targets
    """
    print("Extracting features from MINT model...")
    mint_model.eval()
    
    modal_1_features = []
    modal_2_features = []
    joint_features = []
    targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch
            if not isinstance(X, list):
                X = [X]
            X = [x.to(device) if isinstance(x, torch.Tensor) else x for x in X]
            y = y.to(device)
            
            # Extract UNIMODAL features from individual encoders (before fusion)
            # This is crucial for LSMI - we need features BEFORE multimodal fusion
            encoder = mint_model.encoder
            
            # Get individual encoder outputs for each modality
            z = []
            for enc_idx, (enc, xi) in enumerate(zip(encoder.encoders, X)):
                embedding = enc(xi)
                # Handle dict output (with attention mask)
                if isinstance(embedding, dict):
                    embedding = embedding["token_embeddings"]
                
                # Apply input adapter if present
                if encoder.input_adapters[enc_idx] is not None:
                    embedding = encoder.input_adapters[enc_idx](embedding)
                
                # Pool the embeddings (mean pooling over sequence dimension)
                if len(embedding.shape) == 3:  # (batch, seq, dim)
                    embedding = embedding.mean(dim=1)  # (batch, dim)
                
                z.append(embedding)
            
            # Extract unimodal features
            if len(z) >= 2:
                modal_1_features.append(z[0].detach().cpu())
                modal_2_features.append(z[1].detach().cpu())
            else:
                # Fallback if only one modality
                modal_1_features.append(z[0].detach().cpu())
                modal_2_features.append(z[0].detach().cpu())
            
            # Get joint (fused) features
            joint_out = mint_model.encoder(X)
            if isinstance(joint_out, list):
                joint_out = joint_out[-1]  # Get last output (joint)
            if len(joint_out.shape) == 3:
                joint_out = joint_out.mean(dim=1)
            joint_features.append(joint_out.detach().cpu())
            
            targets.append(y.detach().cpu())
    
    return {
        'modal_1_features': torch.cat(modal_1_features, dim=0),
        'modal_2_features': torch.cat(modal_2_features, dim=0),
        'joint_features': torch.cat(joint_features, dim=0),
        'targets': torch.cat(targets, dim=0)
    }


def run_mint_lsmi_estimation(mint_model, train_dataloader, val_dataloader, 
                             n_classes=2, device='cuda', 
                             discriminator_epochs=50, entropy_epochs=50,
                             batch_size=64, save_path=None):
    """
    Main function to run LSMI estimation on MINT model.
    
    Args:
        mint_model: Trained MINT model
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        n_classes: Number of classes
        device: Device to use
        discriminator_epochs: Epochs for training discriminators
        entropy_epochs: Epochs for training entropy estimators
        batch_size: Batch size for LSMI training
        save_path: Optional path to save results
        
    Returns:
        Dictionary with train and val LSMI results
    """
    print("="*60)
    print("Running LSMI Estimation for MINT Model")
    print("="*60)
    
    # Step 1: Extract features from MINT
    train_features = extract_mint_features(mint_model, train_dataloader, device)
    val_features = extract_mint_features(mint_model, val_dataloader, device)
    
    print(f"\nFeature shapes:")
    print(f"  Modality 1: {train_features['modal_1_features'].shape}")
    print(f"  Modality 2: {train_features['modal_2_features'].shape}")
    print(f"  Targets: {train_features['targets'].shape}")
    
    # Step 2: Create dataloaders for LSMI training
    train_dataset = FeatureDataset(
        train_features['modal_1_features'],
        train_features['modal_2_features'],
        train_features['targets']
    )
    val_dataset = FeatureDataset(
        val_features['modal_1_features'],
        val_features['modal_2_features'],
        val_features['targets']
    )
    
    lsmi_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    lsmi_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Step 3: Train discriminators
    modal_1_dim = train_features['modal_1_features'].shape[1]
    modal_2_dim = train_features['modal_2_features'].shape[1]
    
    discriminators = train_discriminators(
        lsmi_train_loader, modal_1_dim, modal_2_dim, n_classes,
        device=device, num_epochs=discriminator_epochs
    )
    
    # Step 4: Train entropy estimators
    entropy_estimators = train_entropy_estimators(
        lsmi_train_loader, modal_1_dim, modal_2_dim,
        device=device, num_epochs=entropy_epochs
    )
    
    # Step 5: Estimate LSMI on train and val sets
    print("\n" + "="*60)
    print("LSMI Estimation on Training Set")
    print("="*60)
    train_results = estimate_LSMI(lsmi_train_loader, discriminators, entropy_estimators, n_classes, device)
    
    print("\n" + "="*60)
    print("LSMI Estimation on Validation Set")
    print("="*60)
    val_results = estimate_LSMI(lsmi_val_loader, discriminators, entropy_estimators, n_classes, device)
    
    results = {
        'train': train_results,
        'val': val_results
    }
    
    # Save results if path provided
    if save_path:
        torch.save(results, save_path)
        print(f"\nResults saved to: {save_path}")
    
    return results

