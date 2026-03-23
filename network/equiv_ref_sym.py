"""
AxisSym: Equivariant Axis-level Symmetry Detection Network Module

This module implements the main architecture for equivariant axis-level symmetry detection,
including backbone networks, decoder blocks, and various detection branches.
"""

# Standard library imports
import random
import math
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

# E2CNN imports
import e2cnn.nn as enn
from e2cnn import gspaces
from e2cnn.nn import FieldType

# Local imports
from .backbone.equiv_backbone.base_backbone import BaseBackbone
from .dcn import DeformableConv2d as DCN
from utils.reflectional_matching import SingleMatching, SingleMatchingGroup
from utils.rotational_matching import RotationalMatching, RotationalMatchingGroup
from utils.utils import PermMatrix, CustomCenterCrop, rotate_tensor_ccw, rotate_tensor_cw, flip_tensor_vertical

# Constants
TASK_CHANNELS = {
    "loc": 1,  # Localization branch output channels
    "reg": 2,  # Regression branch output channels
    "des": 1   # Description branch output channels
}


class Sigmoid:
    """Custom sigmoid activation with clamping"""
    def __init__(self) -> None:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)


class Interpolate(nn.Module):
    """Upsampling module with configurable scale factor and mode"""
    def __init__(self, size: int = 2, mode: str = "bilinear") -> None:
        super().__init__()
        self.size = size
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.size, mode=self.mode, align_corners=False)


class DecoderBlock(nn.Module):
    """Equivariant decoder block for feature upsampling and fusion
    
    This block performs upsampling of high-level features and combines them with
    skip connections from the encoder, maintaining equivariance throughout.
    
    Args:
        in_type1: Input field type for high-level features
        in_type2: Input field type for skip connection features
        out_type: Output field type after initial processing
        final_type: Final output field type after fusion
    """
    def __init__(
        self,
        in_type1: FieldType,
        in_type2: FieldType,
        out_type: FieldType,
        final_type: FieldType
    ) -> None:
        """Initialize decoder block
        
        Args:
            in_type1: Input field type for high-level features
            in_type2: Input field type for skip connection features
            out_type: Output field type after initial processing
            final_type: Final output field type after fusion
        """
        super(DecoderBlock, self).__init__()

        self.in_type1 = in_type1
        self.in_type2 = in_type2
        self.out_type = out_type

        self.conv1_1 = enn.R2Conv(
            in_type1, out_type, kernel_size=1, bias=False, padding=0
        )  
        self.conv1_2 = enn.R2Conv(
            in_type2, out_type, kernel_size=1, bias=False, padding=0
        )  
        self.bn1_1 = enn.InnerBatchNorm(out_type)
        self.bn1_2 = enn.InnerBatchNorm(out_type)

        self.conv_block = enn.SequentialModule(
            enn.R2Conv(final_type, final_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(final_type),
            enn.ReLU(final_type, inplace=True),
            enn.R2Conv(final_type, final_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(final_type),
        )

        self.relu1 = enn.ReLU(out_type, inplace=True)
        self.relu2 = enn.ReLU(final_type, inplace=True)

    def forward(
        self, 
        x1: torch.Tensor, 
        x2: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of decoder block
        
        Args:
            x1: High-level features to be upsampled
            x2: Skip connection features
            
        Returns:
            Processed and fused feature maps
        """
        self.interpolation = enn.R2Upsampling(
            self.in_type1, size=x2.shape[-2:], mode="bilinear", align_corners=False
        )

        x1 = self.interpolation(x1)
        x1 = self.conv1_1(x1) 
        x1 = self.bn1_1(x1)
        x1 = self.relu1(x1)

        x2 = self.conv1_2(x2) 
        x2 = self.bn1_2(x2)
        x2 = self.relu1(x2)

        x = enn.tensor_directsum([x1, x2]) 

        residual = x  
        out = self.conv_block(x) 

        out += residual
        out = self.relu2(out)

        return out


class BaseBranch(nn.Module):
    """Base class for detection branches (localization and regression)
    
    Implements common functionality including permutation handling,
    convolution operations, and feature processing.
    
    Args:
        device: Computation device
        in_ch: Number of input channels
        branch_conv: Number of convolution layers
        out_ch: Number of output channels
        activation: Whether to use activation in final layer
        dropout: Dropout probability
        fix_seed: Whether to fix random seed
        branch: Branch type ("cyclic" or "dihedral")
        branch_pad: Whether to use padding
        num_anchor: Number of anchor points
    """
    def __init__(
        self,
        device: torch.device,
        in_ch: int,
        branch_conv: int,
        out_ch: int,
        activation: bool,
        dropout: Optional[float],
        fix_seed: bool,
        branch: str,
        branch_pad: bool,
        num_anchor: int,
        branch_name: str,
        ref_feature_choice: str,
        map_size: List[int],
        orientational_anchor: bool,
    ) -> None:
        super().__init__()
        self.device = device
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.activation = activation
        self.branch = branch
        self.branch_pad = branch_pad
        self.num_anchor = num_anchor
        self.branch_name = branch_name
        self.ref_feature_choice = ref_feature_choice
        self.map_size = map_size
        self.orientational_anchor = orientational_anchor
        if fix_seed:
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)

        self._init_transforms()
        self._init_permutation_matrices()
        self._init_conv_layers(branch_conv, dropout)

        if self.activation:
            self.sigmoid = Sigmoid()

    def _init_transforms(self) -> None:
        # self.center_crop = T.CenterCrop((256, 256))
        self.center_crop = CustomCenterCrop((self.map_size[0], self.map_size[1]))

    def _init_permutation_matrices(self) -> None:
        perm_matrix = PermMatrix(
            self.device, self.in_ch, self.num_anchor, self.branch, self.ref_feature_choice
        )
        if self.branch == "cyclic":
            self.perms = perm_matrix.get_perms()
        else:  # dihedral
            self.perms, self.perms_d = perm_matrix.get_perms()
            # self.ref_block = torch.eye(8).flip(dims=[1]).roll(1, 0).to(self.device)

    def zero_out_corners(self, tensor: torch.Tensor, r: int) -> torch.Tensor:
        """Suppress corner regions in the output tensor
        
        Creates masks to zero out corner regions of the output tensor,
        helping to prevent false detections at image corners.
        
        Args:
            tensor: Input tensor to process
            r: Radius of corner region to suppress
            
        Returns:
            Tensor with corner regions suppressed
        """
        b, c, h, w = tensor.shape
        mask_product = torch.ones_like(tensor).to(tensor.device)
        mask_sum = torch.zeros_like(tensor).to(tensor.device)

        for i in range(r):
            for j in range(c):
                if j % 2 != 0:
                    mask_product[:, j, i, : r - i + 1] = 0
                    mask_product[:, j, i, w - r + i :] = 0
                    mask_product[:, j, h - 1 - i, : r - i + 1] = 0
                    mask_product[:, j, h - 1 - i, w - r + i :] = 0

                    mask_sum[:, j, i, : r - i + 1] = 1e-10
                    mask_sum[:, j, i, w - r + i :] = 1e-10
                    mask_sum[:, j, h - 1 - i, : r - i + 1] = 1e-10
                    mask_sum[:, j, h - 1 - i, w - r + i :] = 1e-10

        return tensor * mask_product + mask_sum

    def _init_single_conv_blocks(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1),
        )

    def _init_conv_layers(self, branch_conv: int, dropout: Optional[float]) -> None:
        layers: List[nn.Module] = [
            DCN(self.in_ch, self.in_ch, kernel_size=(3, 3), stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.in_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout is not None else nn.Identity(),
        ]

        for _ in range(branch_conv - 2):
            layers.extend([
                nn.Conv2d(self.in_ch, self.in_ch, 3, padding=2, dilation=2),
                nn.BatchNorm2d(self.in_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout) if dropout is not None else nn.Identity(),
            ])

        self.conv = nn.Sequential(*layers)
        if self.branch == "dihedral":
            self.midpoint_d8_c8_blocks = self._init_single_conv_blocks()
            self.length_d8_c8_blocks = self._init_single_conv_blocks()
            self.orientation_d8_c8_blocks = self._init_single_conv_blocks()

        # self.midpoint_8_4_blocks = self._init_single_conv_blocks()
        # self.length_8_4_blocks = self._init_single_conv_blocks()
        # self.orientation_8_4_blocks = self._init_single_conv_blocks()

    def _process_output(self, geometric_out: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _process_reflection(self, out_tensor, conv_block, swap=False):
        out = out_tensor.clone()
        out = out.reshape(
            out_tensor.shape[0], 2, -1, out_tensor.shape[-2], out_tensor.shape[-1]
        )
        if swap:
            out = out[:, [1, 0], :, :, :]
        out = out.permute(0, 2, 1, 3, 4).reshape(
            -1, 2, out_tensor.shape[-2], out_tensor.shape[-1]
        )
        out = conv_block(out)
        return out.reshape(
            -1, out_tensor.shape[-3] // 2, out_tensor.shape[-2], out_tensor.shape[-1]
        )

    def forward(self, x):
        _, _, h, _ = x.shape
        pad = math.ceil(h / (2 ** (3 / 2) + 2))
        out_list = []

        for i in range(int(self.num_anchor * 2)):
            out = torch.einsum("mn, bnhw -> bmhw", self.perms[i], x)
            if self.branch_pad and i % 2 != 0:
                out = F.pad(out, pad=(pad, pad, pad, pad))
            out = rotate_tensor_ccw(out, i)
            out = self.conv(out)
            out = rotate_tensor_cw(out, i)
            if self.branch_pad and i % 2 != 0:
                out = self.center_crop(out)
            out_list.append(out)

        if self.branch == "dihedral":
            for j in range(int(self.num_anchor * 2)):
                out = torch.einsum("mn, bnhw -> bmhw", self.perms_d[j], x)
                if self.branch_pad and j % 2 != 0:
                    out = F.pad(out, pad=(pad, pad, pad, pad))
                out = flip_tensor_vertical(rotate_tensor_ccw(out, j))
                out = self.conv(out)
                out = rotate_tensor_cw(flip_tensor_vertical(out), j)
                if self.branch_pad and j % 2 != 0:
                    out = self.center_crop(out)
                out_list.append(out)

        out_tensor = torch.cat(out_list, dim=1) # D8 or C8 group feature (B, C, H, W)

        if not self.orientational_anchor:
            if self.branch_name == "midpoint":
                return self.sigmoid(torch.max(out_tensor, dim=1, keepdim=True)[0])
            else:
                return torch.max(out_tensor, dim=1, keepdim=True)[0]

        # Make 16 to 8
        if self.branch == "dihedral":
            # 1
            if self.branch_name == "theta":
                # 1
                out_tensor_1 = self._process_reflection(
                    out_tensor, self.orientation_d8_c8_blocks, swap=False
                )
                out_tensor_2 = self._process_reflection(
                    out_tensor, self.orientation_d8_c8_blocks, swap=True
                )
                out_tensor = out_tensor_1 - out_tensor_2


            elif self.branch_name == "rho" or self.branch_name == "midpoint":
                # 1
                d8_c8_conv = self.length_d8_c8_blocks if self.branch_name == "rho" else self.midpoint_d8_c8_blocks
                out_tensor_1 = self._process_reflection(
                    out_tensor, d8_c8_conv, swap=False
                )
                out_tensor_2 = self._process_reflection(
                    out_tensor, d8_c8_conv, swap=True
                )
                out_tensor = out_tensor_1 + out_tensor_2
          

        if self.branch_name == "rot_center":
            out_tensor, _ = torch.max(out_tensor, dim=1, keepdim=True)
        else:
            out_tensor = out_tensor[:, : self.num_anchor] + out_tensor[:, self.num_anchor :]

        return self._process_output(out_tensor)


class Regression_branch(BaseBranch):
    """Branch for geometric parameter regression (rho, theta)
    
    Extends BaseBranch with upsampling and convolution layers specific to 
    geometric parameter regression. Outputs raw predictions without activation.
    """
    def __init__(self, *args, **kwargs) -> None:
        """Initialize regression branch
        
        Args:
            *args: Variable length argument list passed to BaseBranch
            **kwargs: Arbitrary keyword arguments passed to BaseBranch
        """
        super().__init__(*args, **kwargs)
        # Add regression-specific conv extension with upsampling
        self.conv.extend(
            nn.Sequential(
                Interpolate(size=self.map_size[0]/128), nn.Conv2d(self.in_ch, self.out_ch, 3, 1, 1)
            )
        )

    def _process_output(self, geometric_out: torch.Tensor) -> torch.Tensor:
        """Process regression branch output
        
        Args:
            geometric_out: Raw output tensor from the branch
            
        Returns:
            Unmodified output tensor (no activation needed for regression)
        """
        return geometric_out


class Localization_branch(BaseBranch):
    """Branch for midpoint localization
    
    Extends BaseBranch with convolution and upsampling layers specific to 
    midpoint detection. Includes corner suppression for better localization.
    """
    def __init__(self, *args, **kwargs) -> None:
        """Initialize localization branch
        
        Args:
            *args: Variable length argument list passed to BaseBranch
            **kwargs: Arbitrary keyword arguments passed to BaseBranch
        """
        super().__init__(*args, **kwargs)
        # Add localization-specific conv extension with upsampling
        self.conv.extend(
            nn.Sequential(
                nn.Conv2d(self.in_ch, self.out_ch, 3, 1, 1), Interpolate(size=self.map_size[0]/128)
            )
        )

    def _process_output(self, localize_out: torch.Tensor) -> torch.Tensor:
        """Process localization branch output

        Applies sigmoid activation and optional corner suppression.

        Args:
            localize_out: Raw output tensor from the branch

        Returns:
            Processed tensor with sigmoid activation and corner suppression
        """
        localize_out = self.sigmoid(localize_out)

        if not self.branch_pad:
            _, _, h, _ = localize_out.shape
            r = math.ceil(h * 2 / (2 + math.sqrt(2)))
            localize_out = self.zero_out_corners(localize_out, r)
        
        return localize_out


class Rot_center_fold_branch(nn.Module):
    """Branch for rotation center detection using regular convolutions

    Outputs a single-channel confidence map for rotation centers.
    """

    def __init__(
        self,
        device,
        in_channels: int,
        num_conv: int,
        binary_center: bool,
        dropout: Optional[float] = 0.3,
        map_size: List[int] = [256, 256],
    ) -> None:
        """Initialize rotation center branch

        Args:
            in_channels: Number of input channels
            num_conv: Number of convolution layers
            binary_center: Whether to use binary center
            dropout: Dropout probability (optional)
            map_size: Output map size [H, W]
        """
        super().__init__()
        self.binary_center = binary_center

        layers = [
            # Initial deformable conv for feature adaptation
            DCN(in_channels, in_channels, kernel_size=(
                3, 3), stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout is not None else nn.Identity(),
        ]

        # Add intermediate conv layers
        for _ in range(num_conv - 2):
            layers.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(
                    p=dropout) if dropout is not None else nn.Identity(),
            ])

        # Final conv to reduce to single channel
        out_ch = 1 if binary_center else 8
        layers.extend([
            # 0, 2, 3, 4, 5, 6, 8
            nn.Conv2d(in_channels, out_ch, 3, padding=1),
            Interpolate(size=map_size[0]/128)
        ])

        self.conv = nn.Sequential(*layers).to(device)
        self.sigmoid = Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Rotation center confidence map (B, 1, H, W)
        """
        x = self.conv(x) # B, 8, H, W
        out = F.softmax(x, dim=1) if not self.binary_center else self.sigmoid(x)
        return out


class EquivRefSym(BaseBackbone):
    """Main equivariant axis-level symmetry detection network
    
    Implements the complete network architecture including backbone,
    decoder blocks, and detection branches with equivariant feature processing.
    """

    # [in1, in2, out, final]
    DECODER_CONFIGS: List[Dict[str, Union[List[int], str]]] = [
        {"channels": [512, 256, 128, 256], "name": "up1"},
        {"channels": [256, 128, 64, 128], "name": "up2"},
        {"channels": [128, 64, 32, 64], "name": "up3"},  # for rot_fold_conv
    ]


    def __init__(
        self,
        device: torch.device = torch.device("cuda:0"),
        map_size: List[int] = [256, 256],
        channel_multiplier: int = 1,
        num_anchor: int = 4,
        rot_group: int = 8,
        include_rot: bool = False,
        include_ref: bool = True,
        rot_center_conv: Optional[int] = None,
        rot_fold_conv: Optional[int] = None,
        loc_conv: Optional[int] = None,
        reg_conv: Optional[int] = None,
        dropout: Optional[float] = None,
        ref_feature_choice: str = "cat",
        rot_feature_choice: str = "cat",
        backbone: str = "resnet34",
        ref_matching_channels: Union[int, List[int]] = 2,
        rot_matching_channels: Union[int, List[int]] = 2,
        matching_resolution: int = 128,
        ref_matching_patches_size: Union[int, List[int]] = 1,
        rot_matching_patches_size: Union[int, List[int]] = 1,
        fix_seed: bool = False,
        split: Optional[str] = None,
        disable_permutation: bool = False,
        branch: str = "cyclic",
        branch_pad: bool = False,
        freeze_backbone: bool = False,
        rot_feature_pooling: bool = True,
        orientational_anchor: bool = True,
        binary_center: bool = False,
    ) -> None:
        """Initialize EquivRefSym network
        
        Args:
            device: Computation device
            channel_multiplier: Channel multiplier
            num_anchor: Number of anchor points
            rot_group: Rotation group order
            loc_conv: Number of convolutions in localization branch
            rot_center_conv: Number of convolutions in rot_center branch
            reg_conv: Number of convolutions in regression branch
            dropout: Dropout probability
            feature_choice: Feature combination method
            backbone: Backbone network type
            matching_channels: Matching channels (int or list)
            matching_resolution: Matching resolution
            matching_patches_size: Matching kernel size (int or list)
            fix_seed: Whether to fix random seed
            split: Dataset split (train/test)
            disable_permutation: Whether to use non-permuted features
            branch: Branch type ("cyclic" or "dihedral")
            branch_pad: Whether to use padding in branch
            freeze_backbone: Whether to freeze the backbone
        """
        super().__init__()


        self._init_config(locals())

        # Initialize basic components
        self._init_basic_components()

        # Initialize decoder blocks
        self._init_decoders()

        # Initialize matching layers
        self._init_matching_layers()

        # Initialize branch channels
        self._setup_branches(loc_conv, reg_conv,
                             rot_center_conv, rot_fold_conv, dropout, binary_center)

        # Initialize weights
        self._init_weight()

        # Initialize backbone
        self._init_backbone()

    def _init_config(
        self, 
        params: Dict[str, Union[str, int, float, bool, List[int], torch.device]]
    ) -> None:
        """Initialize network configuration parameters
        
        Args:
            params: Dictionary containing network parameters including:
                - device: Computation device
                - channel_multiplier: Channel multiplier
                - num_anchor: Number of anchor points
                - rot_group: Rotation group order
                - feature_choice: Feature combination method
                - backbone: Backbone network type
                - map_size: Map size
                - fix_seed: Whether to fix random seed
                - matching_channels: Matching channels
                - matching_resolution: Matching resolution
                - matching_patches_size: Matching kernel size
                - split: Dataset split
                - disable_permutation: Whether to use non-permuted features
                - branch: Branch type
                - branch_pad: Whether to use padding
                - freeze_backbone: Whether to freeze the backbone
                - loc_conv: Number of localization convolutions
                - reg_conv: Number of regression convolutions
                - dropout: Dropout probability
        """
        self.device = params['device']
        self.channel_multiplier = params['channel_multiplier']
        self.num_anchor = params['num_anchor']
        self.map_size = params['map_size']
        self.rot_group = params['rot_group']
        self.ref_feature_choice = params['ref_feature_choice']
        self.rot_feature_choice = params['rot_feature_choice']
        self.backbone = params['backbone']
        self.fix_seed = params['fix_seed']
        self.ref_matching_channels = params['ref_matching_channels']
        self.rot_matching_channels = params['rot_matching_channels']
        self.matching_resolution = params['matching_resolution']
        self.ref_matching_patches_size = params['ref_matching_patches_size']
        self.rot_matching_patches_size = params['rot_matching_patches_size']
        self.split = params['split']
        self.disable_permutation = params['disable_permutation']
        self.branch = params['branch']
        self.branch_pad = params['branch_pad']
        self.freeze_backbone = params['freeze_backbone']
        self.include_rot = params['include_rot']
        self.include_ref = params['include_ref']
        self.rot_feature_pooling = params['rot_feature_pooling']
        self.orientational_anchor = params['orientational_anchor']
        self.binary_center = params['binary_center']
        # Store conv and dropout parameters separately as they're used in _setup_branches
        self.loc_conv = params['loc_conv']
        self.reg_conv = params['reg_conv']
        self.dropout = params['dropout']

    def _init_basic_components(self) -> None:
        """Initialize basic network components
        
        Initializes:
            - Random seeds if fix_seed is True
            - Sigmoid activation
            - Geometric spaces
            - 1x1 convolutions for num_anchor=1 case
        """
        if self.fix_seed:
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)

        self.sigmoid = Sigmoid()
        self.gspace = gspaces.FlipRot2dOnR2(self.rot_group)
        self.gspace_rot = gspaces.Rot2dOnR2(self.rot_group)

        if self.num_anchor == 1:
            self.mid1x1 = nn.Conv2d(4, 1, 1)
            self.rho1x1 = nn.Conv2d(4, 1, 1)
            self.theta1x1 = nn.Conv2d(4, 1, 1)

    def _init_decoders(self) -> None:
        """Initialize all decoder blocks
        
        Creates decoder blocks according to DECODER_CONFIGS and
        stores the final type of up3 block for later use.
        """

        if self.include_rot:
            self.DECODER_CONFIGS.append(
                {"channels": [128, 64, 64, 128], "name": "up3_1"}
            )

        for config in self.DECODER_CONFIGS:
            decoder, final_type = self.generate_decoderblock(
                config["channels"], 
                config["name"]
            )
            if config["name"] == "up3":
                self.final_type = final_type

    def _init_matching_layers(self) -> None:
        """Initialize reflective matching layers
        
        Sets up either SingleMatching or SingleMatchingGroup layers based on configuration.
        Only initializes if feature_choice is one of: sim, cat, sum, sep, test, apr
        """
        if self.ref_feature_choice not in ["sim", "cat", "sum", "sep", "test", "apr"]:
            return
        if self.rot_feature_choice not in ["sim", "cat", "sum", "sep", "test", "apr"]:
            return

        # Common matching parameters
        ref_matching_params = {
            "device": self.device,
            "in_type": self.final_type,
            "num_channels": self.ref_matching_channels,
            "feature_resolution": self.matching_resolution,
            "num_rotations": self.rot_group,
            "fix_seed": self.fix_seed,
        }

        rot_matching_params = {
            "device": self.device,
            "in_type": self.final_type,
            "num_channels": self.rot_matching_channels,
            "feature_resolution": self.matching_resolution, 
            "num_rotations": self.rot_group, 
            "fix_seed": self.fix_seed,
        }

        # Additional parameters for SingleMatchingGroup
        ref_group_params = {
            "patch_size": self.ref_matching_patches_size,  # patch_size
            "disable_permutation": self.disable_permutation,  # disable_permutation
        }

        rot_group_params = {
            "patch_size": self.rot_matching_patches_size,  # patch_size
            "disable_permutation": self.disable_permutation,  # disable_permutation 
        }

        # Choose matching layer type based on feature choice
        if self.ref_matching_patches_size == [1]:
            if self.ref_feature_choice in ["sim", "cat", "sum"]:
                self.ref_matching = SingleMatching(**ref_matching_params)
        else: 
            if self.ref_feature_choice in ["sim", "cat", "sum"]:
                self.ref_matching = SingleMatchingGroup(
                    **ref_matching_params, **ref_group_params
                )

        if self.rot_matching_patches_size == [1]:
            if self.rot_feature_choice in ["sim", "cat", "sum"]:
                self.rot_matching = RotationalMatching(**rot_matching_params)

        else:
            if self.rot_feature_choice in ["sim", "cat", "sum"]:
                self.rot_matching = RotationalMatchingGroup(
                    **rot_matching_params, **rot_group_params
                )  

    def _setup_branches(
        self, loc_conv: int, reg_conv: int, rot_center_conv: int, rot_fold_conv: int, dropout: Optional[float], binary_center: bool
    ) -> None:
        """Setup detection branches

        Args:
            loc_conv: Number of convolutions in localization branch
            reg_conv: Number of convolutions in regression branch
            rot_center_conv: Number of convolutions in rot_center branch
            dropout: Dropout probability (None for no dropout)
        """
        # Calculate input channels for branches
        if self.ref_feature_choice == "cat":
            sim_c = (
                sum(self.ref_matching_channels)
                if not isinstance(self.ref_matching_channels, int)
                else self.ref_matching_channels
            )
            in_ch_branch_ref = int(64 + sim_c * 8)

        elif self.ref_feature_choice == "sim":
            sim_c = (
                sum(self.ref_matching_channels)
                if not isinstance(self.ref_matching_channels, int)
                else self.ref_matching_channels
            )
            in_ch_branch_ref = int(sim_c * 8)
        else:
            in_ch_branch_ref = 64

        if self.rot_feature_choice == "cat":
            sim_c = (
                sum(self.rot_matching_channels)
                if not isinstance(self.rot_matching_channels, int)
                else self.rot_matching_channels
            )
            in_ch_branch_rot = int(8 + sim_c * 4)
        elif self.rot_feature_choice == "sim":
            sim_c = (
                sum(self.rot_matching_channels)
                if not isinstance(self.rot_matching_channels, int)
                else self.rot_matching_channels
            )
            in_ch_branch_rot = int(sim_c * 4)
        else:
            in_ch_branch_rot = 8


        if self.include_rot:
            self.rot_center_fold = Rot_center_fold_branch(
                device=self.device,
                in_channels=in_ch_branch_rot,
                num_conv=rot_center_conv,
                dropout=dropout,
                map_size=self.map_size,
                binary_center=binary_center,
            )

        if self.include_ref:
            # Initialize branches
            self.midpoint = Localization_branch(
                device=self.device,
                in_ch=in_ch_branch_ref,
                branch_conv=loc_conv,
                dropout=dropout,
                fix_seed=self.fix_seed,
                out_ch=TASK_CHANNELS["loc"],
                activation=True,
                branch=self.branch,
                branch_pad=self.branch_pad,
                num_anchor=self.num_anchor,
                branch_name="midpoint",
                ref_feature_choice=self.ref_feature_choice,
                map_size=self.map_size,
                orientational_anchor=self.orientational_anchor,
            )

            for name in ["rho", "theta"]:
                setattr(
                    self,
                    name,
                    Regression_branch(
                        device=self.device,
                        in_ch=in_ch_branch_ref,
                        branch_conv=reg_conv,
                        dropout=dropout,
                        fix_seed=self.fix_seed,
                        out_ch=TASK_CHANNELS["loc"],
                        activation=False,
                        branch=self.branch,
                        branch_pad=self.branch_pad,
                        num_anchor=self.num_anchor,
                        branch_name=name,
                        ref_feature_choice=self.ref_feature_choice,
                        map_size=self.map_size,
                        orientational_anchor=self.orientational_anchor,
                    ),
                )

    def _init_backbone(self) -> None:
        """Initialize backbone network based on configuration

        Supports:
            - resnet34: Regular ResNet34 backbone
            - redet: ReDet backbone
        Loads pretrained weights and handles state dict conversion.
        """
        from .backbone.equiv_backbone.re_resnet import ReResNet
        frozen_stages = 4 if self.freeze_backbone else -1
        self.resnet = ReResNet(
            depth=34,
            channel_multiplier=self.channel_multiplier,
            frozen_stages=frozen_stages,
            orientation=self.rot_group,
        )
        weight_dict = torch.load("weights/backbone_resnet34.pth")

        new_state_dict = {}
        for key, value in weight_dict["state_dict"].items():
            new_key = key.replace("backbone.", "")
            new_state_dict[new_key] = value
        keys_to_remove = ["head.fc.weight", "head.fc.bias"]
        for key in keys_to_remove:
            new_state_dict.pop(key, None)
        self.resnet.load_state_dict(new_state_dict, strict=False)

    def _init_weight(self) -> None:
        """Initialize network weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def generate_decoderblock(
        self, 
        channels: List[int], 
        name: str
    ) -> Tuple[DecoderBlock, FieldType]:
        """Generate decoder block with specific name
        
        Args:
            channels: List of channel dimensions [in1, in2, out, final]
            name: Name of the decoder block (up1, up2, or up3)
            
        Returns:
            Tuple containing:
                - Initialized decoder block
                - Final field type of the block
        """
        channels = [i * self.channel_multiplier for i in channels]
        in_type_1 = FieldType(
            self.gspace,
            [self.gspace.regular_repr]
            * (int(channels[0]) // self.gspace.fibergroup.order()),
        )
        in_type_2 = FieldType(
            self.gspace,
            [self.gspace.regular_repr]
            * (int(channels[1]) // self.gspace.fibergroup.order()),
        )
        out_type = FieldType(
            self.gspace,
            [self.gspace.regular_repr]
            * (int(channels[2]) // self.gspace.fibergroup.order()),
        )
        final_type = FieldType(
            self.gspace,
            [self.gspace.regular_repr]
            * (int(channels[3]) // self.gspace.fibergroup.order()),
        )
        decoder_block = DecoderBlock(in_type_1, in_type_2, out_type, final_type)

        # Set as attribute with specific name
        setattr(self, name, decoder_block)

        return getattr(self, name), final_type

    def forward(
        self, 
        x: Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]], 
        test: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the network
        
        Args:
            x: Input tensor or (tensor, features) tuple in test mode
            test: Whether in test mode
            
        Returns:
            Dictionary containing network outputs
        """
        # Store original input size
        original_size = x.shape[-2:]  # (H, W)

        # Get features from backbone and decoder
        decoder_feature, encoder_feature, rot_feature = self._get_decoder_features(
            x, test)

        # Get input features based on feature choice
        input_features_ref, input_features_rot = self._get_input_features(
            decoder_feature, rot_feature)

        # Get predictions from branches
        predictions = self._get_branch_predictions(
            input_features_ref, input_features_rot)

        # Process and format outputs
        return self._format_outputs(
            predictions, decoder_feature, encoder_feature, original_size, input_features_ref, input_features_rot
        )

    def _get_decoder_features(
        self, 
        x: Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]], 
        test: bool
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        """Extract and process features through decoder blocks
        
        Args:
            x: Input tensor or (tensor, features) tuple in test mode
            test: Whether in test mode
            
        Returns:
            Tuple containing:
                - decoder_features: List of features from each decoder block
                - encoder_feature: List of backbone and low-level features
        """
        decoder_feature = []
        encoder_feature = []

        if test:
            x, low_level_feature = x
        else:
            x, low_level_feature = self.resnet(x)
            encoder_feature.extend([x, low_level_feature])

        # Process through decoder blocks
        x = self.up1(x, low_level_feature[3])
        decoder_feature.append(x)
        x = self.up2(x, low_level_feature[2])
        decoder_feature.append(x)

        if self.include_rot:
            rot_feature = self.up3_1(x, low_level_feature[1])
        else:
            rot_feature = None
            

        x = self.up3(x, low_level_feature[1])
        decoder_feature.append(x)

        return decoder_feature, encoder_feature, rot_feature

    def _get_input_features(
        self, 
        decoder_feature: List[torch.Tensor],
        rot_feature: torch.Tensor
    ) -> torch.Tensor:
        """Process features based on feature choice
        
        Args:
            decoder_feature: List of decoder features
            
        Returns:
            Processed input features
        """
        shared_feature = decoder_feature[-1]
        b, c, h, w = shared_feature.tensor.shape
        if self.include_ref:
            if self.ref_feature_choice in ["sim", "cat", "sum", "sep"]:
                ref_similarity = self.ref_matching(shared_feature)

            if self.ref_feature_choice == "cat":
                ref_feature = torch.cat(
                    [shared_feature.tensor, ref_similarity], dim=1)
            elif self.ref_feature_choice == "sum":
                ref_feature = shared_feature.tensor + ref_similarity
            elif self.ref_feature_choice == "sim":
                ref_feature = ref_similarity
            elif self.ref_feature_choice == "apr":
                ref_feature = shared_feature.tensor
        else:
            ref_feature = None  

        if rot_feature is not None:
            if self.rot_feature_choice in ["sim", "cat", "sum", "sep"]:
                rot_similarity = self.rot_matching(shared_feature)

            rot_feature_pool = torch.max(rot_feature.tensor.reshape(
                b, -1, self.rot_group*2, h, w), dim=2)[0]

            if self.rot_feature_choice == "cat":
                rot_feature_output = torch.cat(
                    [rot_feature_pool, rot_similarity], dim=1)
            elif self.rot_feature_choice == "sum":
                rot_feature_output = rot_feature_pool + rot_similarity
            elif self.rot_feature_choice == "sim":
                rot_feature_output = rot_similarity
            elif self.rot_feature_choice == "apr":
                rot_feature_output = rot_feature_pool
        else:
            rot_feature_output = None

        return ref_feature, rot_feature_output

    def _get_branch_predictions(
        self, 
        input_feature_ref: torch.Tensor,
        input_feature_rot: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Get predictions from all branches
        
        Args:
            input_features: Input feature tensor
            
        Returns:
            Dictionary containing predictions for each branch:
                - 'midpoint': Midpoint confidence map
                - 'rot_center': Rotation center predictions
                - 'rho': Rho predictions
                - 'theta': Theta predictions
        """
        # rot_center = self.rot_center(input_feature_rot) if self.rot_feature_pooling else self.rot_center(input_feature_ref)
        if self.include_rot and self.include_ref:
            rot_center = self.rot_center_fold(input_feature_rot)
            return {
                "midpoint": self.midpoint(input_feature_ref),
                "rot_center": rot_center,
                # "rot_fold": self.rot_fold(input_feature_ref),
                "rho": self.rho(input_feature_ref),
                "theta": self.theta(input_feature_ref)
            }
        elif self.include_ref and not self.include_rot:
            return {
                "midpoint": self.midpoint(input_feature_ref),
                "rho": self.rho(input_feature_ref),
                "theta": self.theta(input_feature_ref)
            }
        elif self.include_rot and not self.include_ref:
            rot_center = self.rot_center_fold(input_feature_rot)
            return {
                "rot_center": rot_center,
            }

    def _format_outputs(
        self,
        predictions: Dict[str, torch.Tensor],
        decoder_feature: List[torch.Tensor],
        encoder_feature: List[torch.Tensor],
        original_size: Tuple[int, int],
        input_features_ref: torch.Tensor,
        input_features_rot: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Format network outputs

        Args:
            predictions: Dictionary of branch predictions
            decoder_feature: List of decoder features
            encoder_feature: List of backbone features
            original_size: Original input size (H, W)

        Returns:
            Dictionary containing formatted outputs:
                - 'midpoint_confidence_map': Confidence map
                - 'geometric_map': Geometric parameters
                - 'decoder_feat': Decoder features
                - 'encoder_feature': Backbone features
        """
        if self.split == "test":
            if self.include_ref:
                predictions["midpoint"] = F.interpolate(
                    predictions["midpoint"], 
                    size=original_size,
                    mode="bilinear"
                )
                geometric_map = torch.cat(
                    [predictions["rho"], predictions["theta"]], dim=1)
                geometric_map = F.interpolate(
                    geometric_map,
                    size=original_size,
                    mode="bilinear"
                )
            else:
                geometric_map = torch.cat([predictions["rho"], predictions["theta"]], dim=1)
            if self.include_rot:
                predictions["rot_center"] = F.interpolate(
                    predictions["rot_center"],
                    size=original_size,
                    mode="bilinear"
                )
        if self.include_ref:
            geometric_map = torch.cat(
                    [predictions["rho"], predictions["theta"]], dim=1)



        if self.include_rot and self.include_ref:
            return {
                "midpoint_confidence_map": predictions["midpoint"],
                "rot_center_map": predictions["rot_center"],
                "geometric_map": geometric_map,
                "decoder_feat": decoder_feature,
                "encoder_feature": encoder_feature,
                "input_features_ref": input_features_ref,
                "input_features_rot": input_features_rot,
            }
        elif self.include_ref and not self.include_rot:
            return {
                "midpoint_confidence_map": predictions["midpoint"],
                "geometric_map": geometric_map,
                "decoder_feat": decoder_feature,
                "encoder_feature": encoder_feature,
                "input_features_ref": input_features_ref,
                "input_features_rot": input_features_rot,
            }
        elif self.include_rot and not self.include_ref:
            return {
                "rot_center_map": predictions["rot_center"],
                "decoder_feat": decoder_feature,
                "encoder_feature": encoder_feature,
                "input_features_rot": input_features_rot,
            }