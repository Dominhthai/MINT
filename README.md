## MINT: Multimodal Information NCE Training

A clean PyTorch Lightning codebase for training and evaluating self-supervised and supervised multimodal models on MultiBench-style datasets. This repository implements multiple methods, shared data modules, and utilities for logging, checkpointing, and linear probing.

- **Implemented methods**: MINT
- **Objectives**: InfoNCE-style contrastive learning with careful handling of cross-view and same-view similarities
- **Trainer**: PyTorch Lightning with Hydra-configurable experiments and TensorBoard logging 

### Table of Contents
- Overview
- Methods: MINT
- Objective and Accuracy
- Checkpointing and Linear Probing
- Installation
- Datasets
- Training and Evaluation
- Logging and Checkpoints
- Reproducing Results and Tips
- License
- Citation and References

## Overview
MINT provides a unified training entrypoint for a set of multimodal learning methods on MultiBench-style datasets. It uses Hydra for configuration management and PyTorch Lightning for scalable training, logging, and checkpointing. Methods differ in how they encode modalities and how contrastive/joint objectives are applied, but share the same data module and evaluation utilities for apples-to-apples comparisons.

Core entry script: `main_multibench.py` (Hydra config at `./configs`). Utilities: `utils.py`. Environment: `environment.yml`.

## Methods

### MINT (ours)
- Learns modality-specific encoders (e.g., vision and text) and a joint representation via contrastive objectives.
- Constructs similarity matrices across and within views, explicitly removing self-similarities in same-view matrices to avoid trivial positives.
- Uses multiple InfoNCE calls to align: vision ↔ joint, text ↔ joint, and joint ↔ joint, encouraging consistent cross-modal and joint-space alignment.

## Objective and Accuracy
- The training loss is a contrastive InfoNCE objective built from cross-view (`z1 @ z2^T`) and optionally same-view (`z1 @ z1^T`, `z2 @ z2^T`) similarities.
- Self-similarities in same-view matrices are suppressed (treated as −∞ in the softmax) to prevent degenerate positives and to strengthen the negative set.
- Validation reports both loss and an SSL accuracy proxy (top-1 index match in the cross-view similarity matrix) to track representation quality.

## Checkpointing and Linear Probing
PyTorch Lightning controls checkpointing, and by default it saves the best model according to a monitored validation metric.

- By default (SSL methods), the code monitors `acc1` (SSL accuracy) and saves the top checkpoint automatically. For supervised models, it monitors `val_loss`.
- Linear probing runs as a callback after validation. It trains a lightweight downstream classifier and logs metrics to TensorBoard, but it does not directly decide checkpoint saving unless you explicitly monitor its logged scalar with a `ModelCheckpoint`.
- You can customize the monitored metric in configs or programmatically (e.g., to use linear probe accuracy) by setting a `ModelCheckpoint(monitor=...)` in the trainer callbacks.

## Installation
Set up the environment (conda is recommended):

```bash
conda env create -f environment.yml
conda activate multimodal
```

## Datasets
This repository targets MultiBench-style datasets with multiple modality combinations. Configure dataset, modalities, and task via Hydra configs in `./configs`.

Common configuration fields:
- `data.data_module.dataset`: dataset key (e.g., `mosi`, `visionandtouch`, etc.)
- `model.name`: one of `CoMM`, `CrossSelf`, `CLIP`, `SupervisedClassifier`
- `modalities`: per-dataset modalities are specified in the config
- Additional dataset-specific `encoders`, `adapters`, or projection heads (instantiated via Hydra)

## Training and Evaluation
Hydra entrypoint is `main_multibench.py` with `config_name="train_multibench"` and `config_path="./configs"`.

Run self-supervised training (example):
```bash
python main_multibench.py \
  mode=train \
  model.name=CoMM \
  trainer.max_epochs=100 \
  trainer.devices=1 \
  data.data_module.dataset=YOUR_DATASET_KEY
```

Run CLIP-style training:
```bash
python main_multibench.py \
  mode=train \
  model.name=CLIP \
  data.data_module.dataset=YOUR_DATASET_KEY
```

Supervised finetuning / evaluation:
```bash
python main_multibench.py \
  mode=train \
  model.name=SupervisedClassifier \
  data.data_module.dataset=YOUR_DATASET_KEY
```

Test with a checkpoint:
```bash
python main_multibench.py \
  mode=test \
  model.name=CoMM \
  data.data_module.dataset=YOUR_DATASET_KEY \
  ckpt_path=PATH/TO/checkpoints/best-epoch.ckpt
```

Adjust dataset-specific fields (encoders, adapters, projections) in the corresponding config group.

## Logging and Checkpoints
- Logs: TensorBoard logs are written under the trainer `default_root_dir`, organized by `model.name` and dataset.
- Checkpoints: Best and last checkpoints are saved automatically via `ModelCheckpoint`, monitored by `acc1` for SSL or `val_loss` for supervised unless configured otherwise.

You can set an experiment name to nest results further:
```bash
python main_multibench.py exp_name=exp001
```

## Reproducing Results and Tips
- Ensure seeds are set via config (`seed`) for reproducibility.
- Monitor both `val_loss` and `acc1` to diagnose training quality.
- When using linear probing, remember that probe metrics are logged after validation—monitor an appropriate scalar if you want them to control checkpointing.

## License
MIT License. See `LICENSE` for details.

## Citation and References
If you use this repository or ideas described here, please cite appropriately. For code access and issues:


For PyTorch Lightning and Hydra usage, consult their official documentation.