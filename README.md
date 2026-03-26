# Axis-level Symmetry Detection with Group-Equivariant Representation

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
