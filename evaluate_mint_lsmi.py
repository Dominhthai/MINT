"""
Script to evaluate LSMI (Redundancy, Uniqueness, Synergy) on trained MINT model.

Usage:
    python evaluate_mint_lsmi.py \\
        --ckpt_path path/to/mint/checkpoint.ckpt \\
        --dataset mosi \\
        --batch_size 64 \\
        --device cuda

This will quantify:
- R (Redundancy): Information about Y shared between modality 1 and 2
- U1 (Uniqueness 1): Information about Y unique to modality 1
- U2 (Uniqueness 2): Information about Y unique to modality 2
- S (Synergy): Information about Y that emerges only when both modalities are combined
"""

import argparse
import torch
import os
import sys
from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.mint_lsmi_estimation import run_mint_lsmi_estimation


def load_mint_model(ckpt_path, config_path='./configs', device='cuda'):
    """Load trained MINT model from checkpoint."""
    print(f"Loading MINT model from: {ckpt_path}")
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Get hyperparameters from checkpoint if available
    if 'hyper_parameters' in checkpoint:
        cfg = checkpoint['hyper_parameters']
    else:
        # Load default config
        with hydra.initialize(config_path=config_path, version_base=None):
            cfg = hydra.compose(config_name="train")
    
    # Instantiate model architecture
    dataset = cfg.data.data_module.dataset
    
    # Setup encoders and adapters
    encoders = instantiate(cfg[dataset]["encoders"])
    adapters = instantiate(cfg[dataset]["adapters"])
    encoder_kwargs = {
        "encoders": encoders,
        "input_adapters": adapters
    }
    
    # Instantiate MINT model
    model = instantiate(cfg.model.model, optim_kwargs=cfg.optim, encoder=encoder_kwargs)
    
    # Load state dict
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model, cfg


def get_dataloaders(cfg, dataset_name):
    """Get train and validation dataloaders."""
    print(f"Loading {dataset_name} dataset...")
    
    # Create data module
    data_module = instantiate(
        cfg.data.data_module,
        model="Sup",  # Use supervised mode for labels
        modalities=cfg[dataset_name]["modalities"],
        task=cfg[dataset_name]["task"]
    )
    
    # Setup data module
    data_module.setup('fit')
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Evaluate LSMI on MINT model')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to MINT checkpoint')
    parser.add_argument('--dataset', type=str, default='mosi',
                        choices=['mosi', 'mosei', 'humor', 'sarcasm'],
                        help='Dataset to evaluate on')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for LSMI training')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--discriminator_epochs', type=int, default=50,
                        help='Epochs for training discriminators')
    parser.add_argument('--entropy_epochs', type=int, default=50,
                        help='Epochs for training entropy estimators')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save LSMI results')
    parser.add_argument('--config_path', type=str, default='./configs',
                        help='Path to config directory')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.ckpt_path):
        print(f"Error: Checkpoint not found at {args.ckpt_path}")
        return
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: Running on CPU. This will be slow.")
    
    # Load MINT model
    mint_model, cfg = load_mint_model(args.ckpt_path, args.config_path, device)
    
    # Override dataset in config if specified
    cfg.data.data_module.dataset = args.dataset
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(cfg, args.dataset)
    
    # Determine number of classes
    if args.dataset in ['mosi', 'mosei']:
        n_classes = 2  # Binary classification after thresholding
    elif args.dataset in ['humor', 'sarcasm']:
        n_classes = 2  # Binary classification
    else:
        n_classes = 2  # Default
    
    # Set save path
    if args.save_path is None:
        ckpt_dir = os.path.dirname(args.ckpt_path)
        args.save_path = os.path.join(ckpt_dir, f'lsmi_results_{args.dataset}.pt')
    
    # Run LSMI estimation
    results = run_mint_lsmi_estimation(
        mint_model=mint_model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        n_classes=n_classes,
        device=device,
        discriminator_epochs=args.discriminator_epochs,
        entropy_epochs=args.entropy_epochs,
        batch_size=args.batch_size,
        save_path=args.save_path
    )
    
    # Print summary
    print("\n" + "="*60)
    print("LSMI Estimation Summary")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.ckpt_path}")
    print("\nTraining Set:")
    print(f"  R (Redundancy):    {results['train']['R_mean']:.4f}")
    print(f"  U1 (Uniqueness 1): {results['train']['U1_mean']:.4f}")
    print(f"  U2 (Uniqueness 2): {results['train']['U2_mean']:.4f}")
    print(f"  S (Synergy):       {results['train']['S_mean']:.4f}")
    print("\nValidation Set:")
    print(f"  R (Redundancy):    {results['val']['R_mean']:.4f}")
    print(f"  U1 (Uniqueness 1): {results['val']['U1_mean']:.4f}")
    print(f"  U2 (Uniqueness 2): {results['val']['U2_mean']:.4f}")
    print(f"  S (Synergy):       {results['val']['S_mean']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()

