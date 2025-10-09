## MINT: Multimodal Information-Theoretic Framework for Balanced Representation Learning

### Abstract
Multimodal learning integrates complementary information from language, vision, and audio into unified representations. However, such representations often suffer from imbalance, where dominant modalities overshadow weaker ones, leading to biased embeddings and the lost unimodal information during fusion. In this work, we revisit this problem from an information-theoretic perspective, revealing that imbalance distorts the decomposition of multimodal information: uniqueness from weaker modalities is lost, redundancy from dominant ones is amplified, and synergy becomes less informative. To address this issue, we propose MINT, a framework for balanced multimodal representation. It consists of three complementary modules: Multimodal Contrastive Alignment (MCA) preserves redundancy and synergy through cross-modal alignment, Unimodal Self-Alignment (USA) recovers lost unimodal information by strengthening weaker modalities via knowledge distillation and curriculum learning, and Task Self-Alignment (TSA) enforces task-level consistency to maintain overall information balance. Experiments on multiple benchmarks show that MINT outperforms strong baselines and yields more balanced and informative multimodal representations.

- **Implemented methods**: MINT
- **Keywords** : Multimodal Imbalance Learning, Knowledge Distillation, Curriculumn Learning, Partial Information Decomposition(PID)
- **Objectives**: InfoNCE-style and cosine similarity function, with careful handling of cross-view and same-view similarities
- **Trainer**: PyTorch Lightning with Hydra-configurable experiments run on GPU.

### Table of Contents
- [Abstract](#abstract)
- [Overview](#overview)
- [Methods](#methods)
  - [MINT (ours)](#mint-ours)
- [Objective and Accuracy](#objective-and-accuracy)
- [Checkpointing and Linear Probing](#checkpointing-and-linear-probing)
- [Environment Installation](#environment-installation)
- [Datasets](#datasets)
- [How to run?](#how-to-run)
  - [Training](#training)
  - [Evaluation with linear classification (binary)](#evaluation-with-linear-classification-binary)
- [Logging and Checkpoints](#logging-and-checkpoints)
- [Reproducing Results and Tips](#reproducing-results-and-tips)
- [License](#license)

## Overview
We proposed MINT, an information-theoretic framework for balanced multimodal representation learning. MINT alleviates modality imbalance by combining multimodal contrastive alignment with unimodal self-alignment through knowledge distillation and curriculum learning, while maintaining task-level consistency. Extensive experiments across four benchmark datasets show that MINT consistently outperforms prior methods, producing more stable and balanced multimodal representations. Further analyses confirm its ability to recover lost unimodal information and enhance cross-modal synergy, establishing MINT as a unified and robust framework for multimodal learning.

Core entry script: `main.py` (Hydra config at `./configs`). Utilities: `utils.py`. Environment: `environment.yml`.

## Methods

### MINT (ours)
* We analyze multimodal representation from an information-theoretic perspective, identifying the loss of unimodal information as a key factor that degrades representation quality.
* We propose the MINT framework, integrating MCA, USA, and TSA to preserve cross-modal redundancy, recover unimodal uniqueness, and maintain task-level consistency. 
* We empirically demonstrate that MINT achieves SOTA performance on benchmark datasets while producing more balanced and informative multimodal representations. 

## Objective and Accuracy
  - The training loss is:
    - Contrastive InfoNCE objective built from cross-view (`z1 @ z2^T`) and same-view (`z1 @ z1^T`, `z2 @ z2^T`) similarities (**MCA**)
    - Negative Knowledge Distillation loss (KD loss) that recovers lost information within each modality (**USA**)
    - Cosine similarity that captures task/label self-referred information (**TSA**)
- Validation reports both loss and an SSL accuracy proxy (top-1 index match in the cross-view similarity matrix) to track representation quality.

## Checkpointing and Linear Probing
PyTorch Lightning controls checkpointing, and by default it saves the best model according to a monitored validation metric.

- By default (SSL methods), the code monitors `acc1` (downstream validation accuracy) and saves the top checkpoint automatically. 
- Linear probing runs as a callback after validation. It trains a lightweight downstream classifier and logs metrics to TensorBoard, but it does not directly decide checkpoint saving unless you explicitly monitor its logged scalar with a `ModelCheckpoint`.
- You can customize the monitored metric in configs or programmatically (e.g., to use linear probe accuracy) by setting a `ModelCheckpoint(monitor=...)` in the trainer callbacks.

## Environment Installation
Set up the environment (conda is recommended):

```bash
conda env create -f environment.yml
conda activate multimodal
```

## Datasets
This repository targets MultiBench-style datasets with multiple modality combinations. Configure dataset, modalities, and task via Hydra configs in `./configs`.

Common configuration fields:
- `data.data_module.dataset`: dataset key (e.g., `mosi`, `visionandtouch`, etc.)
- `model.name`: mint
- `modalities`: per-dataset modalities are specified in the config
- Additional dataset-specific `encoders`, `adapters`, or projection heads (instantiated via Hydra)

## How to run?
Hydra entrypoint is `main.py` with `config_name="train"`, `dataset="multibench"`, `model="mint"` and `config_path="./configs"`.

### Self-supervised Training Mode
```bash
python3 main.py \
    data.data_module.dataset=$dataset$ \
    model=mint \
    mode="train" \
    model.model.loss_kwargs.curriculum_weight=true \
    model.model.pretrained_kwargs.use_finetune=false \
    model.model.pretrained_kwargs.use_inputs_embeds=true \
    model.model.pretrained_kwargs.text_in=300 \
    model.model.pretrained_kwargs.video_in=20 \
    model.model.pretrained_kwargs.v_lstm_hidden_size=64 \
    model.model.pretrained_kwargs.video_out=32 \
    model.model.pretrained_kwargs.v_lstm_dropout=0.1 \
    model.model.pretrained_kwargs.alpha=1.0 \
    model.model.curriculum_kwargs.v_tau=0.0 \
    model.model.curriculum_kwargs.v_lam=1.0 \
    model.model.curriculum_kwargs.v_fac=0.9 \
    model.model.curriculum_kwargs.t_tau=0.0 \
    model.model.curriculum_kwargs.t_lam=1.5 \
    model.model.curriculum_kwargs.t_fac=0.9 \
    seed=42 \
    trainer.max_epochs=100 \
    trainer.strategy=auto \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    optim.lr=1e-3 \
    optim.weight_decay=1e-2 
```

### Evaluation with linear classification (binary)
```bash
!python3 main.py \
    data.data_module.dataset=$dataset$ \
    model=mint \
    mode="test" \
    model.model.loss_kwargs.curriculum_weight=true \
    model.model.pretrained_kwargs.use_finetune=false \
    model.model.pretrained_kwargs.use_inputs_embeds=true \
    model.model.pretrained_kwargs.text_in=300 \
    model.model.pretrained_kwargs.video_in=20 \
    model.model.pretrained_kwargs.v_lstm_hidden_size=64 \
    model.model.pretrained_kwargs.video_out=32 \
    model.model.pretrained_kwargs.v_lstm_dropout=0.0 \
    model.model.pretrained_kwargs.alpha=0.0 \
    model.model.curriculum_kwargs.v_tau=0.0 \
    model.model.curriculum_kwargs.v_lam=0.0 \
    model.model.curriculum_kwargs.v_fac=0.0 \
    model.model.curriculum_kwargs.t_tau=0.0 \
    model.model.curriculum_kwargs.t_lam=0.0 \
    model.model.curriculum_kwargs.t_fac=0.0 \
    seed=42 \
    trainer.strategy=auto \
    trainer.devices=1 \
    trainer.num_nodes=1 
```

## Logging and Checkpoints
- Logs: TensorBoard logs are written under the trainer `default_root_dir`, organized by `model.name` and dataset.
- Checkpoints: Best and last checkpoints are saved automatically via `ModelCheckpoint`, monitored by `acc1` for SSL or `val_loss` for supervised unless configured otherwise.

Change the checkpoint path `ckpt_path` in [configs/train.yaml](configs/train.yaml)

## Reproducing Results and Tips
- Ensure seeds are set via config (`seed`) for reproducibility.
- Monitor both `val_loss` and `acc1` to diagnose training quality.
- When using linear probing, remember that probe metrics are logged after validation—monitor an appropriate scalar if you want them to control checkpointing.

## License
MIT License. See `LICENSE` for details.


For PyTorch Lightning and Hydra usage, consult their official documentation.