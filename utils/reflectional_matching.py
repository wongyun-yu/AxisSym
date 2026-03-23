"""Reflective matching module for equivariant feature processing.

This module implements single and group-based reflective matching operations,
supporting both single and multiple patch size configurations.

Key Components:
    - SingleMatching: Basic reflective matching for single patch
    - SingleMatchingGroup: Extended matching supporting multiple patch sizes
    - get_permutation_matrices: Helper function for rotation/reflection matrices
"""

import torch
import torch.nn as nn
import e2cnn.nn as enn
from e2cnn import gspaces
from e2cnn.nn import FieldType
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import numpy as np
from typing import List, Union, Optional


# Get 4 pemutation matrices list consists with 4 rotations and 1 reflection
def get_permutation_matrices(
    device: torch.device,
    num_channels: int,
    num_rotations: int,
    disable_permutation: bool = False
) -> List[torch.Tensor]:
    """Generate rotation and reflection permutation matrices.
    
    Args:
        device: Computation device
        num_channels: Number of feature channels
        num_rotations: Number of rotation groups (typically 4 or 8)
        disable_permutation: If True, returns identity matrices
        
    Returns:
        List of transformation matrices [num_rotations * num_channels * 2, num_rotations * num_channels * 2]
    """
    rotation_matrices = []
    if disable_permutation:
        # Identity matrices for disabled permutations
        rotation_matrices = [
            torch.eye(int(num_rotations * num_channels * 2)).to(device)
            for _ in range(num_rotations // 2 + 1)
        ]
    else:
        # Rotation matrices
        for i in range(num_rotations // 2):  # 0, 45, 90, 135
            block = torch.eye(num_rotations).roll(-i, 0) # For the clockwise rotation(usually 8)
            blocks = [block for _ in range(int(num_channels * 2))] # (* 2) is for the dihedral group
            rotation_matrices.append(torch.block_diag(*blocks).to(device))  # (C x D_8) x (C x D_8)
            # Add 4 permuation matrices for 0, 45, 90, 135

        # Reflection matrix
        block = torch.eye(num_rotations).flip(dims=[1]).roll(1, 0)
        zero_block = torch.zeros_like(block)
        unit = torch.hstack([
            torch.vstack([zero_block, block]),
            torch.vstack([block, zero_block]),
        ])
        units = [unit for _ in range(num_channels)]
        rotation_matrices.append(torch.block_diag(*units).to(device))
        # Add 1 permutation matrix for reflection

    return rotation_matrices

# nun_rot: How many rotations to consider
class SingleMatching(nn.Module):
    """Single patch reflective matching module.
    
    Applies rotation and reflection operations on feature maps using a single patch size
    to compute similarity scores between transformed features.
    
    Args:
        device: Device to run computations on
        in_type: Input field type from e2cnn
        num_channels: Number of feature channels
        feature_resolution: Resolution of feature maps (32 or 64)
        num_rotations: Number of rotation groups (typically 4 or 8)
        fix_seed: Whether to fix random seeds
    """
    def __init__(self, device, in_type, num_channels, feature_resolution, num_rotations, fix_seed):
        super(SingleMatching, self).__init__()
        
        # Basic configurations
        self.device = device
        self.feature_resolution = feature_resolution
        self.num_rotations = num_rotations
        self.num_channels = num_channels[0] if isinstance(num_channels, list) else num_channels

        # Set random seeds if required
        self._set_random_seeds(fix_seed)
        
        # Initialize transformation matrices
        self._init_transformation_matrices()
        
        # Initialize network layers
        self._init_network_layers(in_type)

    def _set_random_seeds(self, fix_seed: bool) -> None:
        """Set random seeds for reproducibility"""
        if fix_seed:
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)

    def _init_transformation_matrices(self) -> None:
        """Initialize rotation and reflection matrices"""
        self.permutation_matrices = get_permutation_matrices(
            self.device, self.num_channels, self.num_rotations
        )
        self.stacked_rotation_matrices = torch.stack(
            [self.permutation_matrices[int(i)] for i in range(self.num_rotations // 2)]
        )  # 4 x (C x D_8) x (C x D_8)  
        self.cosine_similarity = nn.CosineSimilarity(dim=3, eps=1e-6)

    def _init_network_layers(self, in_type: FieldType) -> None:
        """Initialize network layers for feature processing
        
        Args:
            in_type: Input field type from e2cnn
        """
        self.gspace = gspaces.FlipRot2dOnR2(self.num_rotations)
        out_type = FieldType(self.gspace, [self.gspace.regular_repr] * self.num_channels)
        
        # Bottleneck layers
        self.conv1 = enn.R2Conv(in_type=in_type, out_type=in_type, 
                               kernel_size=3, bias=False, padding=1)
        self.bn1 = enn.InnerBatchNorm(in_type)
        self.relu = enn.ReLU(in_type, inplace=True)
        self.conv2 = enn.R2Conv(in_type=in_type, out_type=out_type, 
                               kernel_size=1, bias=False, padding=0)
        
        # Optional maxpool layer
        if self.feature_resolution == 64:
            self.maxpool = enn.PointwiseMaxPool(
                self.conv1.out_type, kernel_size=2, stride=2, padding=0
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass for single patch matching.
        
        Args:
            input: Input tensor [B, C_in, H, W]
            
        Returns:
            Similarity map tensor [B, C*num_rotations, H, W]
        """
        # [B, C*|D_8|, H, W]
        conv_features = self.relu(self.bn1(self.conv1(input)))
        
        # [B, C*|D_8|, H/2, W/2]
        projected_features = self.conv2(conv_features)
        b, c, h, w = projected_features.shape
        
        # [B, C*|D_8|, H/2*W/2]
        input_tensor = projected_features.tensor.squeeze().reshape(b, c, h * w)
        
        # [B, num_rotations//2, num_channels, H*W]
        # [4, (C x D_8), (C x D_8)], [B, (C x D_8), H*W] -> [B, 4, (C x D_8), H*W]
        feature_rot = torch.einsum("rmn, bns -> brms", self.stacked_rotation_matrices, input_tensor)
        # [(C x D_8), (C x D_8)], [B, 4, (C x D_8), H*W] -> [B, 4, (C x D_8), H*W]
        feature_ref = torch.einsum("mn, brns -> brms", self.permutation_matrices[-1], feature_rot)
        
        # [B, 4, C, |D_8|, H, W]
        feature_rot = feature_rot.reshape(b, self.num_rotations // 2, self.num_channels, -1, h, w)
        feature_ref = feature_ref.reshape(b, self.num_rotations // 2, self.num_channels, -1, h, w)
        
        # Final output: [B, C*num_rotations, H, W] 
        # dim=3 cosine similarity
        cos_feat = self.cosine_similarity(feature_rot, feature_ref)
        cos_feat = cos_feat.repeat(1, 2, 1, 1, 1).transpose(1, 2)
        cos_feat = cos_feat.reshape(b, -1, h, w)

        if self.feature_resolution == 64:
            cos_feat = F.interpolate(cos_feat, mode="bilinear", size=[h, w])
        return cos_feat

class SingleMatchingGroup(nn.Module):
    """Group-based reflective matching module for multiple patch sizes
    
    Args:
        device: Device to run computations on
        in_type: Input field type from e2cnn
        num_channels: Number of channels for each patch size
        feature_resolution: Resolution of feature maps (32 or 64)
        num_rotations: Number of rotation groups
        fix_seed: Whether to fix random seeds
        patch_size: Size of patches to process (int or list)
        disable_permutation: Whether to disable permutation matrices
    """
    def __init__(
        self,
        device,
        in_type,
        num_channels, 
        feature_resolution, 
        num_rotations, 
        fix_seed,
        patch_size,
        disable_permutation, 
    ):
        super(SingleMatchingGroup, self).__init__()

        # Store basic configurations
        self._init_basic_configs(device, num_channels, feature_resolution, 
                               num_rotations, patch_size, disable_permutation)

        # Set random seeds if required
        self._set_random_seeds(fix_seed)

        # Initialize transformation matrices
        self._init_transformation_matrices()

        # Initialize similarity metrics
        self._init_similarity_metrics()

        # Initialize network layers
        self._init_network_layers(in_type)

        # Initialize unfold operations
        self._init_unfold_operations()

    def _init_basic_configs(self, device, num_channels, feature_resolution, 
                          num_rotations, patch_size, disable_permutation):
        """Initialize basic configuration parameters"""
        self.device = device
        self.num_rotations = num_rotations
        self.disable_permutation = disable_permutation
        self.feature_resolution = feature_resolution
        self.patch_size = patch_size if len(patch_size) > 1 else patch_size[0]
        self.num_channels = num_channels[0] if len(num_channels) == 1 else num_channels

    def _set_random_seeds(self, fix_seed: bool) -> None:
        """Set random seeds for reproducibility"""
        if fix_seed:
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)

    def _init_transformation_matrices(self) -> None:
        """Initialize rotation and reflection matrices"""
        self.rotation_matrices = []
        channels = (self.num_channels 
                   if isinstance(self.num_channels, list) 
                   else [self.num_channels])

        for channel in channels:
            self.rotation_matrices.append(
                get_permutation_matrices(
                    self.device,
                    channel,
                    self.num_rotations,
                    self.disable_permutation,
                )
            )

    def _init_similarity_metrics(self) -> None:
        """Initialize cosine similarity metrics"""
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.cosine_sim_dim2 = nn.CosineSimilarity(dim=2, eps=1e-6)

    def _init_network_layers(self, in_type: FieldType) -> None:
        """Initialize network layers based on patch size configuration"""
        self.gspace = gspaces.FlipRot2dOnR2(self.num_rotations)

        if isinstance(self.patch_size, int):
            self._init_single_patch_layers(in_type)
        else:
            self._init_multi_patch_layers(in_type)

        self._init_maxpool_layer()

    def _init_single_patch_layers(self, in_type: FieldType) -> None:
        """Initialize layers for single patch size case"""
        out_type = FieldType(self.gspace, [self.gspace.regular_repr] * self.num_channels)

        self.conv1 = enn.R2Conv(in_type=in_type, out_type=in_type, 
                               kernel_size=3, bias=False, padding=1)
        self.bn1 = enn.InnerBatchNorm(in_type)
        self.relu = enn.ReLU(in_type, inplace=True)
        self.conv2 = enn.R2Conv(in_type=in_type, out_type=out_type, 
                               kernel_size=1, bias=False, padding=0)

    def _init_multi_patch_layers(self, in_type: FieldType) -> None:
        """Initialize layers for multiple patch sizes case"""
        out_types = [
            FieldType(self.gspace, [self.gspace.regular_repr] * (self.num_channels[l]))
            for l in range(len(self.num_channels))
        ]

        self.conv1 = nn.ModuleList([
            enn.R2Conv(in_type=in_type, out_type=in_type, 
                        kernel_size=3, bias=False, padding=1)
            for _ in range(len(self.patch_size))
        ])
        self.bn1 = nn.ModuleList([
            enn.InnerBatchNorm(in_type) 
            for _ in range(len(self.patch_size))
        ])
        self.relu = enn.ReLU(in_type, inplace=True)
        self.conv2 = nn.ModuleList([
            enn.R2Conv(in_type=in_type, out_type=out_types[y], 
                        kernel_size=1, bias=False, padding=0)
            for y in range(len(self.patch_size))
        ])

    def _init_maxpool_layer(self) -> None:
        """Initialize maxpool layer based on patch size configuration"""
        if isinstance(self.patch_size, int):
            self.maxpool = enn.PointwiseMaxPool(
                self.conv1.out_type, kernel_size=2, stride=2, padding=0
            )
        else:
            self.maxpool = enn.PointwiseMaxPool(
                self.conv1[0].out_type, kernel_size=2, stride=2, padding=0
            )

    def _init_unfold_operations(self) -> None:
        """Initialize unfold operations based on patch size configuration"""
        if isinstance(self.patch_size, int):
            self.unfold = nn.Unfold(
                kernel_size=self.patch_size,
                padding=self.patch_size // 2,
                stride=1,
            )
        else:
            self.unfold = [
                nn.Unfold(
                    kernel_size=self.patch_size[i],
                    padding=self.patch_size[i] // 2,
                    stride=1,
                )
                for i in range(len(self.patch_size))
            ]

    def forward_single_patch(self, input: torch.Tensor):  # input: [B, C_in, H, W]
        # [B, C_in, H, W]
        conv_features = self.relu(self.bn1(self.conv1(input)))

        # [B, C*|D_8|, H, W]
        projected_features = self.conv2(conv_features)
        b, c, h, w = projected_features.shape

        # [B, C*|D_8|, patch_size, patch_size, H/2, W/2]
        unfolded_patches = self.unfold(projected_features.tensor).reshape(b, c, self.patch_size, self.patch_size, h, w)
        # [B*H*W, C*|D_8|, patch_size, patch_size]
        reshaped_patches = unfolded_patches.permute(0, 4, 5, 1, 2, 3).reshape(-1, c, self.patch_size, self.patch_size)

        similarity_maps = []
        # [0, 1, 2, 3] corresponds to 0, 45, 90, 135 degrees
        for i in range(self.num_rotations // 2): 
            angle = -i * 45

            # [B*H*W, c*|D_8|, patch_size, patch_size]
            if self.patch_size != 1:
                rotated_patches = TF.rotate(
                    reshaped_patches,
                    angle,
                    interpolation=TF.InterpolationMode.BILINEAR,
                )
            else:
                rotated_patches = reshaped_patches

            # Rotation permutation
            # [B*H*W, C*|D_8|, patch_size, patch_size]
            rotated_patches = torch.einsum(
                "lg, bghw -> blhw", self.rotation_matrices[0][i], rotated_patches
            )

            # [B*H*W, C*|D_8|*patch_size*patch_size]
            rotated_patches = rotated_patches.reshape(
                -1,
                self.num_channels,
                2 * self.num_rotations * self.patch_size * self.patch_size,
            )
            rotated_patches = F.normalize(rotated_patches, p=2, dim=2, eps=1e-6)

            reflected_patches = rotated_patches.reshape(-1,
                self.num_channels*2 * self.num_rotations, self.patch_size, self.patch_size)

            reflected_patches = torch.einsum(
                "lg, bghw -> blhw", self.rotation_matrices[0][-1], reflected_patches
            )

            # Apply reflection (horizontal flip)
            # [B*H*W, c*|D_8|, patch_size, patch_size]
            if self.patch_size != 1:
                reflected_patches = TF.hflip(reflected_patches)
            else:
                reflected_patches = reflected_patches

            # [B*H*W, C, num_rotations*2*patch_size*patch_size]
            rotated_patches = rotated_patches.reshape(
                -1,
                self.num_channels,
                2 * self.num_rotations * self.patch_size * self.patch_size,
            )
            # # [B*H*W, C, {|D_8|*patch_size*patch_size]
            reflected_patches = reflected_patches.reshape(
                -1,
                self.num_channels,
                2 * self.num_rotations * self.patch_size * self.patch_size,
            )

            similarity_map = torch.einsum('bmc, bmc -> bm', rotated_patches, reflected_patches)

            # [B, C, H, W]
            similarity_map = (
                similarity_map
                .reshape(b, h, w, self.num_channels)
                .permute(0, 3, 1, 2)
            )
            similarity_maps.append(similarity_map)

        # [B, C, num_rotations*2, H, W]
        similarity_maps = torch.stack(similarity_maps, dim=2).repeat(1, 1, 2, 1, 1)
        # [B, C*num_rotations, H, W]
        final_similarity_map = similarity_maps.reshape(b, -1, h, w)
        if self.feature_resolution == 64:
            final_similarity_map = F.interpolate(
                final_similarity_map, mode="bilinear", size=[h, w]
            )
        return final_similarity_map

    def forward_multi_patch(self, input):  # input: [b, 64, H, W]
        if isinstance(self.patch_size, int):
            conv_features = self.relu(self.bn1(self.conv1(input)))
            _, _, in_h, in_w = conv_features.shape
            if self.feature_resolution == 64:
                conv_features = self.maxpool(conv_features)
            projected_features = self.conv2(conv_features)  # [b, C|D_8|, H, W]
            b, c, h, w = projected_features.shape  # [b, C|D_8|, H, W]

        patch_outputs = []  # 'outs' -> 'patch_outputs'
        for patch_idx in range(len(self.patch_size)):  # 'j' -> 'patch_idx'
            if not isinstance(self.patch_size, int):
                conv_features = self.relu(self.bn1[patch_idx](self.conv1[patch_idx](input)))
                _, _, in_h, in_w = conv_features.shape
                if self.feature_resolution == 64:
                    conv_features = self.maxpool(conv_features)
                projected_features = self.conv2[patch_idx](conv_features)  # [b, C|D_8|, H, W]
                b, c, h, w = projected_features.shape  # [b, C|D_8|, H, W]

            # [B, c*|D_8|, H, W]
            unfolded_patches = self.unfold[patch_idx](projected_features.tensor).reshape(
                b, c, self.patch_size[patch_idx], self.patch_size[patch_idx], h, w
            )

            # [B*H*W, c*|D_8|, patch_size, patch_size]
            reshaped_patches = unfolded_patches.permute(0, 4, 5, 1, 2, 3).reshape(
                -1, c, self.patch_size[patch_idx], self.patch_size[patch_idx]
            )

            similarity_maps = []
            for rotation_idx in range(self.num_rotations // 2):  # 'i' -> 'rotation_idx'
                angle = -rotation_idx * 45
                # Apply rotation
                # [B*H*W, c*|D_8|, patch_size, patch_size]
                if self.patch_size[patch_idx] != 1:
                    rotated_patches = TF.rotate(
                        reshaped_patches,
                        angle,
                        interpolation=TF.InterpolationMode.BILINEAR,
                    )
                else:
                    rotated_patches = reshaped_patches

                # Rotation permutation
                # [B*H*W, C*|D_8|, patch_size, patch_size]
                rotated_patches = torch.einsum(
                    "lg, bghw -> blhw",
                    self.rotation_matrices[patch_idx][rotation_idx],  # indices updated
                    rotated_patches,
                )

                rotated_patches = rotated_patches.reshape(-1, 
                                                          self.num_channels[patch_idx],
                                                          2 * self.num_rotations * self.patch_size[patch_idx] * self.patch_size[patch_idx],)

                rotated_patches = F.normalize(rotated_patches, p=2, dim=2, eps=1e-6)

                # BHW, C|D8|, P, P
                reflected_patches = rotated_patches.reshape(-1,
                                                          self.num_channels[patch_idx]*2 * self.num_rotations,
                                                          self.patch_size[patch_idx],
                                                          self.patch_size[patch_idx])
                #  [C|D8|, C|D8|], [BHW, C|D8|, P, P] -> [BHW, C|D8|, P, P]
                reflected_patches = torch.einsum("lg, bghw -> blhw",
                                                self.rotation_matrices[patch_idx][-1],
                                                reflected_patches)
                
                if self.patch_size[patch_idx] != 1:
                    reflected_patches = TF.hflip(reflected_patches)
                else:
                    reflected_patches = reflected_patches

                rotated_patches = rotated_patches.reshape(-1,
                                                        self.num_channels[patch_idx],
                                                        2 * self.num_rotations * self.patch_size[patch_idx] * self.patch_size[patch_idx]
                                                        ) 
                reflected_patches = reflected_patches.reshape(-1,
                                                        self.num_channels[patch_idx],
                                                        2 * self.num_rotations * self.patch_size[patch_idx] * self.patch_size[patch_idx]
                                                        )

                similarity_map = torch.einsum('bmc, bmc -> bm', rotated_patches, reflected_patches)

                similarity_map = (
                    similarity_map
                    .reshape(b, h, w, self.num_channels[patch_idx])
                    .permute(0, 3, 1, 2)
                )
                similarity_maps.append(similarity_map)  

            # [B, C, num_rotations*2, H, W]
            stacked_similarities = torch.stack(similarity_maps, dim=2).repeat(1, 1, 2, 1, 1)  # 'similarity_maps' -> 'stacked_similarities'
            # [B, C*num_rotations, H, W]
            patch_similarity_map = stacked_similarities.reshape(b, -1, h, w)  # 'final_similarity_map' -> 'patch_similarity_map'
            patch_outputs.append(patch_similarity_map)

        combined_similarity = torch.cat(patch_outputs, dim=1)  # 'out_final' -> 'combined_similarity'

        if self.feature_resolution == 64:
            combined_similarity = F.interpolate(combined_similarity, mode="bilinear", size=[in_h, in_w])
        return combined_similarity

    def forward(self, input):
        if isinstance(self.patch_size, int):
            return self.forward_single_patch(input)
        else:
            return self.forward_multi_patch(input)