from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # General parameters
    batch_size: int = 4
    num_workers: int = 4
    input_size: List[int] = field(default_factory=lambda: [511, 511])
    map_size: List[int] = field(default_factory=lambda: [256, 256])
    kernel_size: int = 5
    num_anchor: int = 4
    sigma: float = 0.6
    rot_center_sigma: float = 0.6

    # Model parameters
    equiv: bool = True
    reg_conv: int = 3
    loc_conv: int = 4
    include_ref: bool = True
    include_rot: bool = True
    rot_center_conv: int = 4
    rot_fold_conv: int = 3
    backbone: str = "resnet34"
    channel_multiplier: float = 1.0
    rot_group: int = 8
    dropout: Optional[float] = 0.5
    disable_permutation: bool = False
    branch: Optional[str] = "dihedral"
    branch_padding: Optional[bool] = True
    reg_kernel: Optional[int] = 1
    freeze_backbone: bool = True
    rot_feature_pooling: bool = True
    orientational_anchor = True

    pretrained_weights: Optional[str] = None
    use_focal_loss: bool = False
    use_focal_loss_ref = False
    use_focal_loss_rot = True
    alpha: float = 0.95
    gamma: float = 2.0

    # Matching parameters
    matching_resolution: int = 64
    ref_feature: str = "cat"
    rot_feature: str = "cat"
    ref_matching_channels: List[int] = field(default_factory=lambda: [2, 3, 3])
    rot_matching_channels: List[int] = field(default_factory=lambda: [2, 3, 3])
    ref_matching_patches_size: List[int] = field(
        default_factory=lambda: [1, 3, 5])
    rot_matching_patches_size: List[int] = field(
        default_factory=lambda: [1, 3, 5])

    # Dataset parameters
    dataset: Optional[str] = "dendi"
    dataset_val: Optional[str] = None
    dataset_test: Optional[str] = None
    synthetic: bool = False
    num_data: Optional[int] = None
    resize: bool = False

    # Training parameters (needed for dataclass completeness)
    epochs: int = 100
    max_lr: float = 1e-3
    weight_decay: float = 1e-5
    bce_weight: float = 5
    mid_weight: float = 2
    rho_weight: float = 2
    theta_weight: float = 300
    rot_center_weight: float = 2
    fix_seed: bool = False
    save_interval: int = 25
    save_epoch: int = 0

    # Testing parameters
    weight: str = "weights/best_model.pt"
    threshold: List[float] = field(default_factory=lambda: [5.0, 10.0, 15.0])

    # Logging parameters
    run_name: Optional[str] = None

    # DDP parameters
    distributed: bool = True
    world_size: int = 1
    gpu_ids: List[int] = field(default_factory=lambda: ["0"])
    port: int = 2022
    local_rank: int = 0
    dist_url: str = "env://"


config = Config()
