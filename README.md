# MMFF: Multi-Modality Feature Fusion Network for Action Recognition

Implementation of "Skeleton Sequence and RGB Frame Based Multi-Modality Feature Fusion Network for Action Recognition"

## Project Structure
```
Thesis_MMFF/
├── models/
│   ├── __init__.py
│   ├── skeleton.py          # ST-GCN backbone
│   ├── rgb.py               # Xception backbone
│   ├── attention.py         # Attention modules
│   ├── fusion.py            # Late fusion module
│   └── mmff_net.py          # Main MMFF network
├── utils/
│   ├── __init__.py
│   ├── dataset.py           # Dataset loader
│   └── graph.py             # Graph definition
├── train.py                 # Training script
├── test.py                  # Testing script
├── train_three_stage.py     # Three-stage training
├── requirements.txt
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data
Organize your data as:
```
data/
├── train_data.npy
├── train_label.pkl
├── val_data.npy
├── val_label.pkl
└── images/
    ├── sample1.jpg
    └── ...
```

### 2. Training

**Three-stage training (Recommended, as per paper):**
```bash
python train_three_stage.py \
    --data-dir ./data \
    --dataset ut-mhad \
    --num-classes 11 \
    --num-joints 20 \
    --batch-size 16 \
    --epochs-stage1 40 \
    --epochs-stage2 20 \
    --epochs-stage3 40
```

**End-to-end training:**
```bash
python train.py \
    --data-dir ./data \
    --dataset ut-mhad \
    --num-classes 11 \
    --num-joints 20 \
    --batch-size 16 \
    --epochs 80
```

### 3. Testing
```bash
python test.py \
    --data-dir ./data \
    --checkpoint ./checkpoints/best_model.pth \
    --num-classes 11 \
    --num-joints 20 \
    --save-confusion-matrix
```

## Results

### UT-MHAD Dataset
- Setting-1: XX.X%
- Setting-2: XX.X%

### NTU RGB+D Dataset
- Cross-Subject: XX.X%
- Cross-View: XX.X%

## Citation
```
@article{zhu2021skeleton,
  title={Skeleton Sequence and RGB Frame Based Multi-Modality Feature Fusion Network for Action Recognition},
  author={Zhu, Xiaoguang and Zhu, Ye and Wang, Haoyu and Wen, Honglin and Yan, Yan and Liu, Peilin},
  journal={ACM Trans. Multimedia Comput. Commun. Appl.},
  year={2021}
}
```