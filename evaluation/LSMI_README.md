# LSMI Estimation for MINT

This module implements Lightweight Sample-wise Multimodal Interaction (LSMI) estimation for the MINT model, based on the paper **"Efficient Quantification of Multimodal Interaction at Sample Level"** (ICML 2025).

## What is LSMI?

LSMI quantifies three types of information in multimodal learning using Partial Information Decomposition (PID) theory:

- **R (Redundancy)**: Information about the target Y that is shared between modality 1 (vision) and modality 2 (text)
- **U₁ (Uniqueness 1)**: Information about Y that is unique to modality 1 (vision only)
- **U₂ (Uniqueness 2)**: Information about Y that is unique to modality 2 (text only)
- **S (Synergy)**: Information about Y that emerges only when both modalities are considered together

These components satisfy:
```
I(X₁; Y) = R + U₁
I(X₂; Y) = R + U₂
I(X₁, X₂; Y) = R + U₁ + U₂ + S
```

## How It Works

1. **Feature Extraction**: Extract learned representations from your trained MINT model
2. **Discriminator Training**: Train classifiers to estimate mutual information I(X;Y)
3. **Entropy Estimation**: Train kernel density estimators to estimate entropy H(X)
4. **LSMI Computation**: Calculate R, U, S using the LSMI framework

## Usage

### Quick Start

```bash
# After training your MINT model, run LSMI estimation:
python evaluate_mint_lsmi.py \\
    --ckpt_path ./mint/mosi/best-epoch=50-acc1=0.85.ckpt \\
    --dataset mosi \\
    --batch_size 64 \\
    --device cuda
```

### Arguments

- `--ckpt_path`: Path to your trained MINT checkpoint (required)
- `--dataset`: Dataset name (mosi, mosei, humor, sarcasm)
- `--batch_size`: Batch size for LSMI training (default: 64)
- `--device`: Device to use (cuda/cpu, default: cuda)
- `--discriminator_epochs`: Epochs for training discriminators (default: 50)
- `--entropy_epochs`: Epochs for training entropy estimators (default: 50)
- `--save_path`: Path to save results (default: auto-generated)
- `--config_path`: Path to config directory (default: ./configs)

### Example Output

```
==============================================================
LSMI Estimation Summary
==============================================================
Dataset: mosi

Training Set:
  R (Redundancy):    0.3245
  U1 (Uniqueness 1): 0.1532
  U2 (Uniqueness 2): 0.2187
  S (Synergy):       0.1124

Validation Set:
  R (Redundancy):    0.3156
  U1 (Uniqueness 1): 0.1489
  U2 (Uniqueness 2): 0.2103
  S (Synergy):       0.1078
==============================================================
```

## Interpretation

### High Redundancy (R)
- Both modalities capture similar information
- Modalities are highly complementary
- Good multimodal fusion

### High Uniqueness (U₁, U₂)
- Each modality provides distinct information
- Potential for modality-specific processing
- Shows modality balance/imbalance

### High Synergy (S)
- Combining modalities creates new information
- Effective multimodal integration
- Strong cross-modal interactions

### For MINT Analysis

Compare LSMI values:
- **Before training**: Random/imbalanced representations
- **After MINT training**: Should show:
  - Increased R (recovered redundancy)
  - Balanced U₁ and U₂ (recovered uniqueness from weak modalities)
  - Maintained or increased S (preserved synergy)

This validates MINT's effectiveness in addressing modality imbalance!

## Integration with MINT

The LSMI estimator works by:

1. Loading your trained MINT model
2. Extracting features for both modalities (vision and text)
3. Training small auxiliary networks for LSMI estimation
4. Computing sample-level R, U, S values
5. Reporting statistics

**Note**: This is a post-hoc analysis method. It doesn't require retraining MINT.

## Implementation Details

### Files

- `mint_lsmi_estimation.py`: Core LSMI implementation
  - `MargKernel`: Entropy estimator using mixture of Gaussians
  - `ClassifierNetwork`: Discriminator for mutual information
  - `extract_mint_features()`: Extract features from MINT
  - `run_mint_lsmi_estimation()`: Main estimation pipeline

- `../evaluate_mint_lsmi.py`: Command-line interface

### Requirements

All dependencies are already in MINT's `requirements.txt`:
- pytorch
- numpy
- hydra-core
- omegaconf

### Computational Cost

- Discriminator training: ~50 epochs (~2-5 minutes on GPU)
- Entropy estimation: ~50 epochs (~2-5 minutes on GPU)  
- Total time: ~5-10 minutes per dataset

## Citation

If you use LSMI estimation in your research, please cite both papers:

```bibtex
@inproceedings{yang2025Efficient,
  title={Efficient Quantification of Multimodal Interaction at Sample Level},
  author={Yang, Zequn and Wang, Hongfa and Hu, Di},
  booktitle={Forty-Second International Conference on Machine Learning},
  year={2025}
}

@article{your_mint_paper,
  title={MINT: Multimodal Information-Theoretic Framework for Balanced Representation Learning},
  author={Your Name},
  journal={Your Venue},
  year={2025}
}
```

## Troubleshooting

### Out of Memory
- Reduce `--batch_size`
- Use `--device cpu` (slower but uses less memory)

### Feature Dimension Mismatch
- Ensure your MINT model outputs separate features for each modality
- Check that encoder outputs are correctly extracted

### Negative Values
- Small negative values after adjustment are normal
- The `RUS_adjustment` function ensures physical constraints are met

## Advanced Usage

### Using in Python Code

```python
from evaluation.mint_lsmi_estimation import run_mint_lsmi_estimation

# After loading your MINT model and dataloaders
results = run_mint_lsmi_estimation(
    mint_model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    n_classes=2,
    device='cuda',
    discriminator_epochs=50,
    entropy_epochs=50,
    batch_size=64,
    save_path='lsmi_results.pt'
)

# Access results
print(f"Redundancy: {results['val']['R_mean']}")
print(f"Synergy: {results['val']['S_mean']}")

# Access per-sample values
sample_redundancy = results['val']['r_sample']  # Shape: (num_samples,)
```

### Analyzing Specific Samples

```python
import torch

# Load saved results
results = torch.load('lsmi_results.pt')

# Get per-sample values
r = results['val']['r_sample']
u1 = results['val']['u1_sample']
u2 = results['val']['u2_sample']
s = results['val']['s_sample']

# Find samples with high synergy
high_synergy_idx = torch.where(s > s.mean() + s.std())[0]
print(f"High synergy samples: {high_synergy_idx.tolist()}")

# Analyze modality imbalance
imbalance = torch.abs(u1 - u2)
balanced_samples = torch.where(imbalance < imbalance.median())[0]
```

## Contact

For questions about LSMI implementation, refer to the original paper or contact the authors at **zqyang@ruc.edu.cn**.

