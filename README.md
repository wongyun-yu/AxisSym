# AxisSym: Axis-level Symmetry Detection with Group-Equivariant Representation

Official implementation of **"Axis-level Symmetry Detection with Group-Equivariant Representation"** (ICCV 2025).

**Authors:** Wongyun Yu, Ahyun Seo, Minsu Cho
**Institution:** POSTECH

## Overview

AxisSym detects **reflection symmetry axes** and **rotation symmetry centers** as explicit geometric primitives using a D8-equivariant neural network. Key contributions:

- **Axis-level detection**: Predicts symmetry axes as line segments (midpoint, orientation, length) rather than dense heatmaps
- **Orientational anchor expansion**: Exploits the orientation dimension of D_N-equivariant features for orientation-specific detection
- **Equivariant matching modules**: Reflectional and rotational matching that leverage group structure for symmetry-consistent feature comparison

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.0+, CUDA 12.x

## Data Preparation

Download the datasets and place them under `data/`:

| Dataset | Description | Download |
|---------|-------------|----------|
| DENDI   | Primary training/evaluation dataset | [Link] |
| LDRS    | Cross-dataset evaluation | [Link] |
| SDRW    | Cross-dataset evaluation | [Link] |

Expected directory structure:
```
data/
├── dendi/
│   ├── symmetry/          # images
│   ├── train_rot_final.json
│   ├── val_rot_final.json
│   └── test_rot_final.json
├── dendi_synthetic_rot_654654_233455/  # augmented training data
│   └── train_test_val_rot_final.json
├── LDRS/                  # LDRS images
├── SDRW/                  # SDRW images
├── LDRS_annotations_test.json
├── SDRW_annotations_test.json
└── cat_LDRS_SDRW_annotations_train_revised.json
```

## Pretrained Model

Download the pretrained model and place it in `weights/`:

| Model | Ref sAP@15 | Rot Center sAP@15 | Rot Fold sAP@15 | Download |
|-------|------------|-------------------|-----------------|----------|
| AxisSym (D8, ResNet-34) | 24.7 | 40.0 | 28.9 | [best_model.pt](weights/best_model.pt) |

## Training

```bash
# Multi-GPU training (recommended)
torchrun --nproc_per_node=4 train.py --cfg configs/train_dendi.py

# Single-GPU training
python train.py --cfg configs/train_dendi.py
```

## Evaluation

```bash
# Evaluate on DENDI test set
python test.py --cfg configs/eval_dendi.py --weight weights/best_model.pt

# Cross-dataset evaluation (LDRS)
python test.py --cfg configs/eval_cross_dataset.py --weight weights/best_model.pt
```

## Results

### DENDI (sAP)

| Method | sAP@5 | sAP@10 | sAP@15 |
|--------|-------|--------|--------|
| **Reflection** | 18.7 | 22.7 | 24.7 |
| **Rot Center** | 36.8 | 39.1 | 40.0 |
| **Rot Fold** | 26.6 | 28.3 | 28.9 |

## Citation

```bibtex
@inproceedings{yu2025axis,
  title={Axis-level Symmetry Detection with Group-Equivariant Representation},
  author={Yu, Wongyun and Seo, Ahyun and Cho, Minsu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

## License

This project is released under the MIT License.
