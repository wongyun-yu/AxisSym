import cv2
import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist
from skimage.draw import line
from typing import List, Optional, Tuple
import copy
# from configs.config import Config
from torch.utils.data import DataLoader

TARGET_SIZE: int = 128

def nms(heat: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    """Apply non-maximum suppression to heatmap
    
    Args:
        heat: Input heatmap tensor
        kernel: Size of max pooling kernel
        
    Returns:
        Tensor with non-maximum values suppressed
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def unnormalize_image(
    img: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> torch.Tensor:
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    return img


def gather_distributed_tensor(
    tensor: torch.Tensor,
    world_size: int,
    device: torch.device
) -> torch.Tensor:
    """Gather tensors from all distributed processes
    
    Args:
        tensor: Local tensor to gather
        world_size: Number of distributed processes
        device: Computation device
        
    Returns:
        Concatenated tensor from all processes
    """
    # Step 1: Gather tensor sizes
    local_size = torch.tensor([tensor.size(0)], dtype=torch.long, device=device)
    all_sizes = [
        torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)
    ]
    dist.all_gather(all_sizes, local_size)

    # Step 2: Pad tensors to maximum size
    max_size = max([size.item() for size in all_sizes])
    if max_size > tensor.size(0):
        padding = max_size - tensor.size(0)
        tensor_padded = F.pad(tensor, (0, padding))
    else:
        tensor_padded = tensor

    # Step 3: Gather all padded tensors
    gathered_tensors = [torch.zeros_like(tensor_padded) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor_padded)

    # Step 4: Remove padding and concatenate
    final_tensors = []
    for i, size in enumerate(all_sizes):
        if size.item() > 0:
            final_tensors.append(gathered_tensors[i][: size.item()])
    if not final_tensors:
        print("[Warning] gather_distributed_tensor: No non-empty tensors found across ranks.")
        return torch.empty((0,), dtype=gathered_tensors[0].dtype, device=gathered_tensors[0].device)
    
    return torch.cat(final_tensors, dim=0)


def get_pred_lines(
    midpoint_nms: torch.Tensor,
    midpoint_conf: torch.Tensor,
    geometry: torch.Tensor,
    num_anchor: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract line segments from network predictions
    
    Args:
        midpoint_nms: NMS applied midpoint predictions
        midpoint_conf: Midpoint confidence scores
        geometry: Geometric parameters (rho, theta)
        num_anchor: Number of anchor points
        
    Returns:
        Tuple containing:
            - Predicted line segments (N, 4) format: [start_x, start_y, end_x, end_y]
            - Confidence scores for each line segment
    """
    _, h, _ = midpoint_nms.shape
    coords = midpoint_nms.squeeze().nonzero()
    score = midpoint_conf[coords[:, 0], coords[:, 1], coords[:, 2]]

    coords_y = (coords[:, 1]).unsqueeze(1)
    coords_x = (coords[:, 2]).unsqueeze(1)
    coords_float = torch.cat([coords[:, 0][:, None], coords_y, coords_x], dim=1)

    rho = geometry[:num_anchor][coords[:, 0], coords[:, 1], coords[:, 2]]

    theta = geometry[num_anchor:][coords[:, 0], coords[:, 1], coords[:, 2]] + (
        (coords_float[:, 0]) * torch.pi / num_anchor
    )
    theta = (theta + torch.pi) * (theta < 0) + theta * (theta >= 0)
    theta -= torch.pi / 2

    start_x = coords_float[:, 2] + (rho / 2) * torch.cos(theta)
    start_y = coords_float[:, 1] + (rho / 2) * torch.sin(theta)
    end_x = coords_float[:, 2] - (rho / 2) * torch.cos(theta)
    end_y = coords_float[:, 1] - (rho / 2) * torch.sin(theta)

    min = torch.zeros_like(start_x)
    max = torch.ones_like(start_x) * (h - 1)
    start_x = torch.max(torch.min(start_x, max), min)
    start_y = torch.max(torch.min(start_y, max), min)
    end_x = torch.max(torch.min(end_x, max), min)
    end_y = torch.max(torch.min(end_y, max), min)

    line_segments = torch.vstack([start_x, start_y, end_x, end_y]).T

    return line_segments, score


def get_pred_lines_single_anchor(
    midpoint_nms: torch.Tensor,
    midpoint_conf: torch.Tensor,
    geometry: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract line segments from network predictions for single anchor case
    
    Args:
        midpoint_nms: NMS applied midpoint predictions [H, W]
        midpoint_conf: Midpoint confidence scores [H, W]
        geometry: Geometric parameters [2, H, W] containing:
                 - geometry[0]: rho values
                 - geometry[1]: theta values
        
    Returns:
        Tuple containing:
            - line_segments: Predicted line segments [N, 4] format: [start_x, start_y, end_x, end_y]
            - scores: Confidence scores [N] for each line segment
    """
    # Get image dimensions and find non-zero coordinates
    h, w = midpoint_nms.squeeze().shape
    coords = midpoint_nms.nonzero(as_tuple=False)
    scores = midpoint_conf[coords[:, 0], coords[:, 1]]

    # Extract y, x coordinates
    coords_y = coords[:, 0].float()
    coords_x = coords[:, 1].float()

    # Get geometric parameters
    rho = geometry[0, coords[:, 0], coords[:, 1]]
    theta = geometry[1, coords[:, 0], coords[:, 1]] - (torch.pi / 2)

    # Calculate line segment endpoints using polar coordinates
    half_rho = rho / 2
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Calculate start and end points
    start_x = coords_x + half_rho * cos_theta
    start_y = coords_y + half_rho * sin_theta
    end_x = coords_x - half_rho * cos_theta
    end_y = coords_y - half_rho * sin_theta

    # Clamp coordinates to image boundaries
    start_x = torch.clamp(start_x, 0, w - 1)
    start_y = torch.clamp(start_y, 0, h - 1)
    end_x = torch.clamp(end_x, 0, w - 1)
    end_y = torch.clamp(end_y, 0, h - 1)

    # Stack coordinates into line segments
    line_segments = torch.stack([start_x, start_y, end_x, end_y], dim=1)

    return line_segments, scores


class PermMatrix:
    """Generates permutation matrices for cyclic and dihedral group branch operations

    Args:
        device: Computation device
        in_ch: Number of input channels
        num_anchor: Number of anchor points
        branch: Type of branch ("cyclic" or "dihedral")
    """

    def __init__(
        self,
        device: torch.device,
        in_ch: int,
        num_anchor: int = 4,
        branch: str = "cyclic",
        feature_choice: str = "cat",
    ) -> None:
        self.device = device
        self.in_ch = in_ch
        self.num_anchor = num_anchor
        self.branch = branch
        self.feature_choice = feature_choice

        # Generate cyclic permutations
        self.perms = self._generate_cyclic_perms()

        # Generate dihedral permutations if needed
        if self.branch == "dihedral":
            self.perms_d = self._generate_dihedral_perms()

    def _generate_cyclic_perms(self) -> List[torch.Tensor]:
        """Generate cyclic permutation matrices"""
        perms = []
        for i in range(int(self.num_anchor * 2)):
            block = torch.eye(int(self.num_anchor * 2)).roll(i, 0)
            num_blocks = self.in_ch // (self.num_anchor * 2)
            blocks = [block for _ in range(num_blocks)]
            perm = torch.block_diag(*blocks).to(self.device)
            perms.append(perm)
        return perms

    def _generate_dihedral_perms(self) -> List[torch.Tensor]:
        """Generate dihedral permutation matrices"""
        perms_d_cat = []
        block = torch.eye(int(self.num_anchor * 2)).roll(0, 0)  # Reference block
        ref_block = torch.eye(8).flip(dims=[1]).roll(1, 0)
        ref_zero_block = torch.zeros_like(block)
        ref_unit = torch.hstack(
            [
                torch.vstack([ref_zero_block, ref_block]),
                torch.vstack([ref_block, ref_zero_block]),
            ]
        )
        for i in range(int(self.num_anchor * 2)):
            r_block = torch.eye(int(self.num_anchor * 2)).roll(i, 0)
            rot_unit = torch.block_diag(*[r_block] * 2)
            d_block = torch.matmul(ref_unit, rot_unit)  # 16 x 16
            if self.feature_choice == "cat":
                d_blocks = [d_block] * 4 + [r_block] * (
                    (self.in_ch - 64) // (self.num_anchor * 2)
                )
            elif self.feature_choice == "apr":
                d_blocks = [d_block] * 4
            elif self.feature_choice == "sim":
                d_blocks = [r_block] * (self.in_ch // (self.num_anchor * 2))

            perm_d_cat = torch.block_diag(*d_blocks).to(self.device)
            perms_d_cat.append(perm_d_cat)
        return perms_d_cat

    def get_perms(self):
        """Return appropriate permutation matrices based on branch type"""
        if self.branch == "cyclic":
            return self.perms
        elif self.branch == "dihedral":
            return self.perms, self.perms_d


# AP calculation
class APAccumulator:
    """Accumulator for computing Average Precision metrics
    
    Handles distributed computation of true/false positives and
    calculates AP values at multiple thresholds.
    """
    
    def __init__(self, num_thresholds: int = 3, device: Optional[torch.device] = None, distributed: bool = True):
        """Initialize AP accumulator

        Args:
            num_thresholds: Number of threshold values for AP calculation
            device: Computation device
            distributed: Whether running in distributed mode
        """
        self.distributed = distributed
        self.world_size = dist.get_world_size() if distributed else 1
        self.num_thresholds = num_thresholds
        self.device = device
        self.tp = [torch.tensor([], device=device) for _ in range(num_thresholds)]
        self.fp = [torch.tensor([], device=device) for _ in range(num_thresholds)]
        self.scores = [torch.tensor([], device=device)]
        self.n_gt = 0

    def update(self, tp, fp, scores, n_gt):
        for i in range(self.num_thresholds):
            self.tp[i] = torch.cat([self.tp[i], tp[i]])
            self.fp[i] = torch.cat([self.fp[i], fp[i]])
        self.scores[0] = torch.cat([self.scores[0], scores])
        self.n_gt += n_gt

    def gather_tpfp(self):
        if not self.distributed:
            return
        for i in range(self.num_thresholds):
            self.tp[i] = gather_distributed_tensor(
                self.tp[i], self.world_size, self.device
            )
            self.fp[i] = gather_distributed_tensor(
                self.fp[i], self.world_size, self.device
            )
        self.scores[0] = gather_distributed_tensor(
            self.scores[0], self.world_size, self.device
        )

        # Aggregate n_gt across all processes
        local_n_gt = torch.tensor([self.n_gt], dtype=torch.long, device=self.device)
        dist.all_reduce(local_n_gt, op=dist.ReduceOp.SUM)
        self.n_gt = local_n_gt.item()

    def pr_to_ap(self, recall, precision):
        # Add a sentinel value at the end
        recall = torch.cat(
            [
                torch.tensor([0.0], device=recall.device),
                recall,
                torch.tensor([1.0], device=recall.device),
            ]
        )
        precision = torch.cat(
            [
                torch.tensor([1.0], device=precision.device),
                precision,
                torch.tensor([0.0], device=precision.device),
            ]
        )

        # Compute the precision envelope
        for i in range(precision.shape[0] - 1, 0, -1):
            precision[i - 1] = torch.max(precision[i - 1], precision[i])

        # Find indices where recall changes
        i = torch.where(recall[1:] != recall[:-1])[0]

        # Compute average precision
        ap = torch.sum((recall[i + 1] - recall[i]) * precision[i + 1])
        return ap.item()

    def compute_ap(self):
        ap_results = []
        for i in range(len(self.tp)):
            tp = self.tp[i]
            fp = self.fp[i]
            scores = self.scores[0]

            sorted_indices = torch.argsort(-scores)
            tp_cumsum = torch.cumsum(tp[sorted_indices], dim=0)
            fp_cumsum = torch.cumsum(fp[sorted_indices], dim=0)

            recall = tp_cumsum / self.n_gt
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)

            ap = self.pr_to_ap(recall, precision)
            ap_results.append(ap)

        return ap_results

    def reset(self):
        self.__init__(num_thresholds=self.num_thresholds, device=self.device)


def TPFP(
    line_pred: torch.Tensor,
    line_gt: torch.Tensor,
    threshold: float,
    ellipse_mask: torch.Tensor,
    ellipse_center: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate True Positives and False Positives for line segment detection
    
    Args:
        line_pred: Predicted line segments
        line_gt: Ground truth line segments
        threshold: Distance threshold for matching
        ellipse_mask: Binary mask for valid regions
        ellipse_center: Center points of ellipses
        
    Returns:
        Tuple containing:
            - True positive indicators
            - False positive indicators
    """

    if len(line_gt) == 0:
        # All predictions are considered false positives
        tp = torch.zeros(size=(len(line_pred),), dtype=torch.float, device=line_pred.device)
        fp = torch.ones(size=(len(line_pred),), dtype=torch.float, device=line_pred.device)
        return tp, fp

    diff = ((line_pred[:, None, :, None] - line_gt[:, None]) ** 2).sum(-1)
    diff = torch.minimum(
        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
    )

    line_pred_mid = (line_pred[:, 0] + line_pred[:, 1]) / 2
    diff_centers = ((line_pred_mid[:, None, :] - ellipse_center[None, :, :]) ** 2).sum(
        -1
    )

    choice = torch.argmin(diff, 1)
    dist, _ = torch.min(diff, 1)
    hit = torch.zeros(size=(len(line_gt),), dtype=torch.bool, device=line_pred.device)
    tp = torch.zeros(size=(len(line_pred),), dtype=torch.float, device=line_pred.device)
    fp = torch.zeros(size=(len(line_pred),), dtype=torch.float, device=line_pred.device)

    ellipse_mask_np = ellipse_mask.cpu().numpy()

    for i in range(len(line_pred)):
        y1, x1 = line_pred[i, 0].tolist()
        y2, x2 = line_pred[i, 1].tolist()

        y1_int, x1_int = int(round(y1)), int(round(x1))
        y2_int, x2_int = int(round(y2)), int(round(x2))

        rr, cc = line(y1_int, x1_int, y2_int, x2_int)

        rr_clipped = np.clip(rr, 0, ellipse_mask_np.shape[0] - 1)
        cc_clipped = np.clip(cc, 0, ellipse_mask_np.shape[1] - 1)

        inside_mask = ellipse_mask_np[rr_clipped, cc_clipped]
        total_pixels = len(inside_mask)
        inside_ratio = inside_mask.sum() / total_pixels if total_pixels > 0 else 0.0

        condition = inside_ratio >= 0.7

        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = 1
        elif len((diff_centers[i] < threshold/2)) == 0:
            fp[i] = 1
        elif (diff_centers[i] < threshold/2)[0] and condition:
            tp[i] = 1
        else:
            fp[i] = 1
    return tp, fp


def normalize_coordinates(
    coords: torch.Tensor,
    h: int,
    w: int,
    target_size: int = TARGET_SIZE
) -> torch.Tensor:
    """Normalize coordinates to target size
    
    Args:
        coords: Input coordinates
        h: Original height
        w: Original width
        target_size: Target size for normalization
        
    Returns:
        Normalized coordinates
    """
    normalized = coords.clone()

    if len(coords.shape) == 3:
        normalized[:, :, 0] = coords[:, :, 1] / h * target_size
        normalized[:, :, 1] = coords[:, :, 0] / w * target_size
        return normalized

    else:
        normalized[:, 0] = coords[:, 1] / h * target_size
        normalized[:, 1] = coords[:, 0] / w * target_size
        return normalized


def calculate_rot_center_ap(
    model: nn.Module,
    loader: DataLoader,
    cfg,
    local_gpu_id: int,
    binary_center: bool = False
) -> List[float]:
    """Calculate Average Precision for rotation center detection
    
    Args:
        model: Neural network model
        loader: DataLoader for test data
        cfg: Configuration object
        local_gpu_id: Local GPU device ID
        
    Returns:
        List of AP values at different thresholds
    """
    model.eval()
    ap_accumulator = APAccumulator(
        num_thresholds=len(cfg.threshold), device=local_gpu_id, distributed=cfg.distributed
    )

    with torch.no_grad():
        for data in tqdm(
            loader, desc="Calculating Rotation Center AP", leave=False, disable=(cfg.rank != 0)
        ):
            inputs = data["img"].to(local_gpu_id)
            outputs = model(inputs)
            
            # Get rotation center predictions, max over the channel dimension
            if not binary_center:
                rot_center_map = outputs["rot_center_map"][:, :7].max(dim=1, keepdim=True)[0]
            else:
                rot_center_map = outputs["rot_center_map"]
            # rot_center_map = data['rot_center_map'].to(local_gpu_id)
            
            # Apply NMS to rotation center predictions
            rot_center_nms = nms(rot_center_map, kernel=5)
            rot_center_nms = rot_center_nms * (rot_center_nms > 0.01)
            
            batch_size = inputs.size(0)
            _, _, height, width = rot_center_map.shape


            for b in range(batch_size):
                if len(data['rot_centers'][b]) == 0:
                    continue
                # Get predicted center coordinates and scores
                rot_center_coords = rot_center_nms[b].squeeze().nonzero() # y, x
                scores = rot_center_map[b].squeeze()[rot_center_coords[:, 0], rot_center_coords[:, 1]] # y, x
                
                # Convert coordinates to normalized space
                pred_rot_centers = rot_center_coords.float()  # y, x

                normalized = pred_rot_centers.clone()  # [N, 2]
                normalized[:, 0] = pred_rot_centers[:, 1] / width * TARGET_SIZE  
                normalized[:, 1] = pred_rot_centers[:, 0] / height * TARGET_SIZE  
                pred_rot_centers = normalized  # x, y, 128 scale, [N, 2]

                # ================================================

                # Get ground truth centers and re-normalize
                gt_rot_centers = data['rot_centers'][b].to(local_gpu_id)
                gt_rot_centers[:, 0] = gt_rot_centers[:, 0] * \
                    width * (TARGET_SIZE / width)
                gt_rot_centers[:, 1] = gt_rot_centers[:, 1] * \
                    height * (TARGET_SIZE / height)

                # original_h, original_w = data['original_shape'][b][:2]
                # gt_rot_centers = gt_rot_centers * torch.tensor([[original_w, original_h]]).to(local_gpu_id) # x, y

                # margin = (max(original_w, original_h) - min(original_w, original_h)) / 2

                # if max(original_w, original_h) == original_w:
                #     gt_rot_centers[:, 1] = gt_rot_centers[:, 1] + margin
                # else:
                #     gt_rot_centers[:, 0] = gt_rot_centers[:, 0] + margin
                # gt_rot_centers = gt_rot_centers * TARGET_SIZE / max(original_w, original_h)

                n_gt = len(gt_rot_centers)

                # Initialize lists for true positives and false positives
                tp_list = []
                fp_list = []
                
                for threshold in cfg.threshold:
                    # Calculate squared distances between predictions and ground truths
                    if len(pred_rot_centers) > 0 and n_gt > 0:
                        distances = ((pred_rot_centers[:, None, :] - gt_rot_centers[None, :, :]) ** 2).sum(-1)
                        min_dists, matched_gt_idx = distances.min(dim=1)
                        
                        # Initialize TP and FP arrays
                        tp = torch.zeros(len(pred_rot_centers), dtype=torch.float, device=local_gpu_id)
                        fp = torch.zeros(len(pred_rot_centers), dtype=torch.float, device=local_gpu_id)
                        
                        # Mark matched predictions
                        matched_gt = torch.zeros(n_gt, dtype=torch.bool, device=local_gpu_id)
                        
                        # Assign TP/FP based on distance threshold
                        for pred_idx, (dist, gt_idx) in enumerate(zip(min_dists, matched_gt_idx)):
                            # if dist < (threshold/128) ** 2 and not matched_gt[gt_idx]:  # Normalize threshold by image size
                            if dist < (threshold/2) and not matched_gt[gt_idx]:     
                                tp[pred_idx] = 1
                                matched_gt[gt_idx] = True
                            else:
                                fp[pred_idx] = 1
                    else:
                        tp = torch.zeros(len(pred_rot_centers), dtype=torch.float, device=local_gpu_id)
                        fp = torch.ones(len(pred_rot_centers), dtype=torch.float, device=local_gpu_id)
                    
                    tp_list.append(tp)
                    fp_list.append(fp)
                # Update accumulator
                if len(scores) > 0:
                    ap_accumulator.update(tp_list, fp_list, scores, n_gt)
                else:
                    ap_accumulator.update(
                        [torch.zeros(0, device=local_gpu_id) for _ in cfg.threshold],
                        [torch.zeros(0, device=local_gpu_id) for _ in cfg.threshold],
                        torch.zeros(0, device=local_gpu_id),
                        n_gt
                    )

    # Gather results from all processes
    ap_accumulator.gather_tpfp()
    
    # Compute AP values
    ap_results = ap_accumulator.compute_ap()
    
    return ap_results


def calculate_ap(
    model: nn.Module,
    loader: DataLoader,
    cfg,
    local_gpu_id: int
) -> List[float]:
    """Calculate Average Precision for line segment detection
    
    Args:
        model: Neural network model
        loader: DataLoader for test data
        cfg: Configuration object
        local_gpu_id: Local GPU device ID
        
    Returns:
        List of AP values at different thresholds
    """
    model.eval()
    ap_accumulator = APAccumulator(
        num_thresholds=len(cfg.threshold), device=local_gpu_id, distributed=cfg.distributed
    )
    total_ground_truths = 0

    with torch.no_grad():
        for data in tqdm(
            loader, desc="Calculating AP", leave=False, disable=(cfg.rank != 0)
        ):  
            
            inputs = data["img"].to(local_gpu_id)
            outputs = model(inputs)

            midpoint_conf_map = outputs["midpoint_confidence_map"]
            _, _, height, width = midpoint_conf_map.shape
            geometric_map = outputs["geometric_map"]

            # Apply Non-Max Suppression and thresholding
            midpoint_map_nms = nms(midpoint_conf_map, kernel=5)
            midpoint_map_nms = midpoint_map_nms * (midpoint_map_nms > 0.01)

            # Prepare ground truth data
            gt_lines = data["gt_lines"].to(local_gpu_id) * height
            num_lines = data["n_lines"]
            ellipse_centers = data["ellipse_center"].to(local_gpu_id)
            ellipse_masks = data["gt_ellipses"].to(local_gpu_id)

            # Adjust ellipse masks dimensions and interpolate
            if ellipse_masks.dim() == 3:
                ellipse_masks = ellipse_masks.unsqueeze(1)
            ellipse_masks = F.interpolate(
                ellipse_masks, scale_factor=0.5, mode="bilinear"
            )

            batch_size = inputs.size(0)
            for b in range(batch_size):
                # Extract and clamp ground truth lines for the current batch

                gt_lines_b = gt_lines[b][: num_lines[b]].reshape(-1, 2, 2)
                gt_lines_b = torch.clamp(gt_lines_b, min=0, max=height - 1)
                n_gt_b = len(gt_lines_b)
                total_ground_truths += n_gt_b

                # Get predicted lines based on the number of anchors
                if cfg.num_anchor == 1 or (cfg.orientational_anchor is False):
                    line_pred, scores = get_pred_lines_single_anchor(
                        midpoint_map_nms[b].squeeze(),
                        midpoint_conf_map[b].squeeze(),
                        geometric_map[b],
                    )
                else:
                    line_pred, scores = get_pred_lines(
                        midpoint_map_nms[b],
                        midpoint_conf_map[b],
                        geometric_map[b],
                        cfg.num_anchor,
                    )
                # Reshape and normalize predicted and ground truth lines
                line_pred = line_pred.reshape(-1, 2, 2)
                line_pred = normalize_coordinates(line_pred, height, width)
                gt_lines_b = normalize_coordinates(gt_lines_b, height, width)
                # Initialize lists for true positives and false positives
                tp_list = []
                fp_list = []

                for threshold in cfg.threshold:
                    tp, fp = TPFP(
                        line_pred,
                        gt_lines_b,
                        threshold,
                        ellipse_masks[b].squeeze(),
                        ellipse_centers[b][:, [1, 0]] * 128,
                    )
                    tp_list.append(tp)
                    fp_list.append(fp)

                # Update the accumulator with TP, FP, scores, and ground truth count
                ap_accumulator.update(tp_list, fp_list, scores, n_gt_b)
    # Gather all true positives and false positives from distributed processes
    ap_accumulator.gather_tpfp()
    # Compute Average Precision for each threshold
    ap_results = ap_accumulator.compute_ap()

    return ap_results


def calculate_normalized_ap(
    model: nn.Module, loader: DataLoader, cfg, local_gpu_id: int
) -> List[float]:
    """Calculate Normalized Average Precision for line segment detection

    Normalized sAP calculates AP for each individual image across all thresholds
    and then computes the mean AP for each threshold.

    Args:
        model: Neural network model
        loader: DataLoader for test data
        cfg: Configuration object
        local_gpu_id: Local GPU device ID

    Returns:
        List of mean Average Precision values for each threshold
    """
    model.eval()
    sum_ap = [0.0 for _ in cfg.threshold]
    total_images = 0

    with torch.no_grad():
        for data in tqdm(
            loader,
            desc="Calculating Normalized sAP",
            leave=False,
            disable=(cfg.rank != 0),
        ):
            inputs = data["img"].to(local_gpu_id)
            outputs = model(inputs)

            midpoint_conf_map = outputs["midpoint_confidence_map"]
            _, _, height, width = midpoint_conf_map.shape
            geometric_map = outputs["geometric_map"]

            # Apply Non-Max Suppression and thresholding
            midpoint_map_nms = nms(midpoint_conf_map, kernel=5)
            midpoint_map_nms = midpoint_map_nms * (midpoint_map_nms > 0.01)

            # Prepare ground truth data
            gt_lines = data["gt_lines"].to(local_gpu_id) * height
            num_lines = data["n_lines"]
            ellipse_centers = data["ellipse_center"].to(local_gpu_id)
            ellipse_masks = data["gt_ellipses"].to(local_gpu_id)

            # Adjust ellipse masks dimensions and interpolate
            if ellipse_masks.dim() == 3:
                ellipse_masks = ellipse_masks.unsqueeze(1)
            ellipse_masks = F.interpolate(
                ellipse_masks, scale_factor=0.5, mode="bilinear"
            )

            batch_size = inputs.size(0)
            for b in range(batch_size):
                # Extract and clamp ground truth lines for the current batch
                gt_lines_b = gt_lines[b][: num_lines[b]].reshape(-1, 2, 2)
                gt_lines_b = torch.clamp(gt_lines_b, min=0, max=height - 1)
                n_gt_b = len(gt_lines_b)
                if n_gt_b == 0:
                    continue  # Skip images with no ground truth

                # Get predicted lines based on the number of anchors
                if cfg.num_anchor == 1:
                    line_pred, scores = get_pred_lines_single_anchor(
                        midpoint_map_nms[b].squeeze(),
                        midpoint_conf_map[b].squeeze(),
                        geometric_map[b],
                    )
                else:
                    line_pred, scores = get_pred_lines(
                        midpoint_map_nms[b],
                        midpoint_conf_map[b],
                        geometric_map[b],
                        cfg.num_anchor,
                    )

                # Reshape and normalize predicted and ground truth lines
                line_pred = line_pred.reshape(-1, 2, 2)
                line_pred = normalize_coordinates(line_pred, height, width)
                gt_lines_b = normalize_coordinates(gt_lines_b, height, width)

                for idx, threshold in enumerate(cfg.threshold):
                    tp, fp = TPFP(
                        line_pred,
                        gt_lines_b,
                        threshold,
                        ellipse_masks[b].squeeze(),
                        ellipse_centers[b][:, [1, 0]] * 128,
                    )

                    # Sort predictions by scores
                    sorted_indices = torch.argsort(-scores)
                    tp_sorted = tp[sorted_indices]
                    fp_sorted = fp[sorted_indices]
                    scores_sorted = scores[sorted_indices]

                    # Compute cumulative true positives and false positives
                    tp_cumsum = torch.cumsum(tp_sorted, dim=0)
                    fp_cumsum = torch.cumsum(fp_sorted, dim=0)

                    recall = tp_cumsum / n_gt_b
                    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

                    # Compute AP for the current threshold
                    ap = _compute_ap_single(recall, precision)
                    sum_ap[idx] += ap

                total_images += 1

    # Gather all APs from distributed processes
    if cfg.distributed and cfg.rank != -1:
        sum_ap_tensor = torch.tensor(sum_ap, device=local_gpu_id)
        total_images_tensor = torch.tensor([total_images], device=local_gpu_id)
        dist.all_reduce(sum_ap_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_images_tensor, op=dist.ReduceOp.SUM)
        sum_ap = sum_ap_tensor.tolist()
        total_images = total_images_tensor.item()

    # Calculate mean AP for each threshold
    mean_ap = [
        (ap_sum / total_images) if total_images > 0 else 0.0 for ap_sum in sum_ap
    ]
    return mean_ap


def _calculate_f1_score(model: nn.Module, loader: DataLoader, cfg, local_gpu_id: int) -> Tuple[float, float, float]:
    """Calculate F1 scores for rotation centers and midpoints
    
    Args:
        model: Neural network model
        loader: DataLoader for test data
        cfg: Configuration object
        local_gpu_id: Local GPU device ID
        
    Returns:
        Tuple containing:
            - Best F1 score for rotation centers
            - Best F1 score for midpoints
            - Optimal threshold
    """
    model.eval()
    thresholds = torch.linspace(0, 1, 100)

    max_dist = 2  # Radius of dilation
    ks = max_dist * 2 + 1  # Kernel size

    dilation_filter = torch.zeros(1, 1, ks, ks, device=local_gpu_id)
    center = max_dist
    for i in range(ks):
        for j in range(ks):
            # More precise circular distance calculation
            y_dist = (i - center) 
            x_dist = (j - center)
            dist = np.sqrt(x_dist**2 + y_dist**2)
            # Slightly tighter threshold for more circular shape
            if dist <= max_dist - 0.5:  # Adjust this threshold for circle precision
                dilation_filter[0, 0, i, j] = 1

    # dilation_filter = torch.zeros(1, 1, ks, ks, device=local_gpu_id)
    # for i in range(ks):
    #     for j in range(ks):
    #         dist = (i-max_dist) ** 2 + (j-max_dist) ** 2
    #         if dist <= max_dist**2:
    #             dilation_filter[0, 0, i, j] = 1
    
    dilation_filter = torch.zeros(1, 1, ks, ks, device=local_gpu_id)
    for i in range(ks):
        for j in range(ks):
            # Calculate exact Euclidean distance from center
            dist = np.sqrt((i-max_dist)**2 + (j-max_dist)**2)
            # Create sharp circular boundary
            if dist <= max_dist:
                dilation_filter[0, 0, i, j] = 1

    # Initialize storage for metrics at each threshold
    rot_metrics = {t.item(): {'tp': 0, 'fp': 0, 'fn': 0} for t in thresholds}
    mid_metrics = {t.item(): {'tp': 0, 'fp': 0, 'fn': 0} for t in thresholds}
    
    with torch.no_grad():
        cfg.distributed = False if not hasattr(cfg, 'rank') else True
        cfg.rank = 0 if not hasattr(cfg, 'rank') else cfg.rank


        for data in tqdm(loader, desc="Calculating F1 Score", leave=False, disable=(cfg.rank != 0)):
            inputs = data["img"].to(local_gpu_id)
            outputs = model(inputs)

            # Get rotation center and midpoint confidence maps
            rot_center_map = outputs["rot_center_map"]
            midpoint_conf_map = outputs["midpoint_confidence_map"]
            
            batch_size = inputs.size(0)
            _, _, height, width = rot_center_map.shape

            for threshold in thresholds:
                t = threshold.item()
                
                # Apply NMS and thresholding
                # rot_center_nms = nms(rot_center_map, kernel=1)
                # rot_center_pred = (rot_center_nms > threshold)
                # midpoint_map_nms, _ = nms(midpoint_conf_map, kernel=1).max(dim=1, keepdim=True)
                # midpoint_map_pred = (midpoint_map_nms > threshold)
                
                rot_center_pred = (rot_center_map > threshold)
                midpoint_map_pred = (midpoint_conf_map.max(dim=1, keepdim=True)[0] > threshold)


                for b in range(batch_size):

                    gt_midpoints = data['midpoints'][b][:data['n_lines'][b]].long()
                    gt_rot_centers = data['rot_centers_resized'][b].long()
                    
                    # Create initial binary maps
                    gt_mid_map = torch.zeros(1, 1, height, width, device=local_gpu_id)
                    gt_mid_map[0, 0, gt_midpoints[:, 1], gt_midpoints[:, 0]] = 1
                    
                    gt_rot_map = torch.zeros(1, 1, height, width, device=local_gpu_id)
                    gt_rot_map[0, 0, gt_rot_centers[:, 1], gt_rot_centers[:, 0]] = 1

                    # Dilate ground truth points
                    gt_mid_map = F.conv2d(gt_mid_map, dilation_filter, padding=max_dist)
                    gt_mid_map = (gt_mid_map > 0).float()
                    
                    gt_rot_map = F.conv2d(gt_rot_map, dilation_filter, padding=max_dist)
                    gt_rot_map = (gt_rot_map > 0).float()


                    # Calculate metrics for rotation centers
                    pred_rot = rot_center_pred[b].unsqueeze(0)
                    gt_rot = gt_rot_map > 0
                    
                    rot_metrics[t]['tp'] += torch.logical_and(pred_rot, gt_rot).sum().item()
                    rot_metrics[t]['fp'] += torch.logical_and(pred_rot, ~gt_rot).sum().item()
                    rot_metrics[t]['fn'] += torch.logical_and(~pred_rot, gt_rot).sum().item()

                    # Calculate metrics for midpoints
                    pred_mid = midpoint_map_pred[b].unsqueeze(0)
                    gt_mid = gt_mid_map > 0
                    
                    mid_metrics[t]['tp'] += torch.logical_and(pred_mid, gt_mid).sum().item()
                    mid_metrics[t]['fp'] += torch.logical_and(pred_mid, ~gt_mid).sum().item()
                    mid_metrics[t]['fn'] += torch.logical_and(~pred_mid, gt_mid).sum().item()

    # Calculate F1 scores for each threshold
    rot_f1_scores = {}
    mid_f1_scores = {}
    
    for t in thresholds:
        t = t.item()
        # Rotation centers F1
        rot_precision = rot_metrics[t]['tp'] / (rot_metrics[t]['tp'] + rot_metrics[t]['fp'] + 1e-10)
        rot_recall = rot_metrics[t]['tp'] / (rot_metrics[t]['tp'] + rot_metrics[t]['fn'] + 1e-10)
        rot_f1_scores[t] = 2 * (rot_precision * rot_recall) / (rot_precision + rot_recall + 1e-10)
        
        # Midpoints F1
        mid_precision = mid_metrics[t]['tp'] / (mid_metrics[t]['tp'] + mid_metrics[t]['fp'] + 1e-10)
        mid_recall = mid_metrics[t]['tp'] / (mid_metrics[t]['tp'] + mid_metrics[t]['fn'] + 1e-10)
        mid_f1_scores[t] = 2 * (mid_precision * mid_recall) / (mid_precision + mid_recall + 1e-10)

    # Find best F1 scores and corresponding threshold
    best_rot_f1 = max(rot_f1_scores.values())
    best_mid_f1 = max(mid_f1_scores.values())
    best_threshold_rot = max(rot_f1_scores.keys(), key=lambda t: rot_f1_scores[t])
    best_threshold_mid = max(mid_f1_scores.keys(), key=lambda t: mid_f1_scores[t])
    # If using distributed training, gather results
    if cfg.distributed and cfg.rank != -1:
        best_scores = torch.tensor([best_rot_f1, best_mid_f1, best_threshold_rot, best_threshold_mid], device=local_gpu_id)
        dist.all_reduce(best_scores, op=dist.ReduceOp.MAX)
        best_rot_f1, best_mid_f1, best_threshold_rot, best_threshold_mid = best_scores.tolist()

    return best_rot_f1, best_mid_f1, best_threshold_rot, best_threshold_mid, \
        gt_rot_map, gt_mid_map, gt_rot_centers, gt_midpoints, rot_center_map, midpoint_conf_map.max(dim=1, keepdim=True)[0]


def flip_tensor_vertical(input_tensor: torch.Tensor) -> torch.Tensor:
    """Flip tensor vertically (along width dimension)"""
    # Flip the tensor along the width dimension (last dimension)
    flipped_tensor = input_tensor.flip(
        dims=[-1]
    )  # Flips the tensor along the width axis
    return flipped_tensor


def rotate_tensor_ccw(input_tensor: torch.Tensor, i: int) -> torch.Tensor:
    """Rotate tensor counterclockwise by i*45 degrees"""
    angle = torch.tensor(
        i * 45 * torch.pi / 180
    )  # positive sign for counterclockwise rotation

    # Get the batch size, height, and width
    b, _, _, _ = input_tensor.shape

    # Define the affine transformation matrix for rotation
    theta = (
        torch.tensor(
            [
                [torch.cos(angle), -torch.sin(angle), 0],
                [torch.sin(angle), torch.cos(angle), 0],
            ],
            dtype=torch.float32,
        )
        .unsqueeze(0)
        .repeat(b, 1, 1)
    )

    # Create the affine grid
    grid = F.affine_grid(theta, input_tensor.size(), align_corners=False).to(
        input_tensor.device
    )

    # Apply the grid sampler
    rotated_tensor = F.grid_sample(input_tensor, grid, align_corners=False)

    return rotated_tensor


def rotate_tensor_cw(input_tensor: torch.Tensor, i: int) -> torch.Tensor:
    """Rotate tensor clockwise by i*45 degrees"""
    angle = -torch.tensor(
        i * 45 * torch.pi / 180
    )  # positive sign for counterclockwise rotation

    # Get the batch size, height, and width
    b, c, h, w = input_tensor.shape

    # Define the affine transformation matrix for rotation
    theta = (
        torch.tensor(
            [
                [torch.cos(angle), -torch.sin(angle), 0],
                [torch.sin(angle), torch.cos(angle), 0],
            ],
            dtype=torch.float32,
        )
        .unsqueeze(0)
        .repeat(b, 1, 1)
    )

    # Create the affine grid
    grid = F.affine_grid(theta, input_tensor.size(), align_corners=False).to(
        input_tensor.device
    )

    # Apply the grid sampler
    rotated_tensor = F.grid_sample(input_tensor, grid, align_corners=False)

    return rotated_tensor


def visualize_axes(
    img, original_img, midpoint_confidence_map, geometric_map, full_size=False
):

    img_show = unnormalize_image(img)
    original_img_show = unnormalize_image(original_img)

    _, original_h, original_w = original_img.shape

    nms_mid = nms(midpoint_confidence_map, kernel=5)
    nms_mid *= nms_mid > 0.1
    line, scores = get_pred_lines(
        nms_mid[0], midpoint_confidence_map[0], geometric_map[0], num_anchor=4
    )

    if full_size:
        pad_x = (
            round((256 - (256 * original_w / original_h)) / 2)
            if round((256 - (256 * original_w / original_h)) / 2) > 5
            else 0
        )
        pad_y = (
            round((256 - (256 * original_h / original_w)) / 2)
            if round((256 - (256 * original_h / original_w)) / 2) > 5
            else 0
        )
        
        line = (
            (line - torch.tensor([pad_x, pad_y, pad_x, pad_y]).cuda())
            / torch.tensor(
                [256 - 2 * pad_x, 256 - 2 * pad_y, 256 - 2 * pad_x, 256 - 2 * pad_y]
            ).cuda()
            * torch.tensor([original_w, original_h, original_w, original_h]).cuda()
        )
        img_pred = (original_img_show.clone().permute(1, 2, 0) * 255).numpy()


    else:
        img_pred = (img_show.permute(1, 2, 0) * 255).numpy()
        line *= 2

    thickness = 2
    line = line.reshape(-1, 2, 2)
    img_pred = img_pred[:, :, ::-1].astype(np.uint8).copy()
    for i in range(len(line)):
        cv2.line(
            img_pred,
            line[i][0].to(torch.int64).tolist(),
            line[i][1].to(torch.int64).tolist(),
            color=[0, 255, 0],
            thickness=thickness,
        )
    return img_pred[:, :, ::-1], line.reshape(-1, 4), scores


def visualize_symmetries(
    img, original_img, midpoint_confidence_map, geometric_map, rot_map=None, full_size=False, ref_threshold=0.1, rot_threshold=0.1
):
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    img_show = unnormalize_image(img)
    original_img_show = unnormalize_image(original_img)

    _, original_h, original_w = original_img.shape

    nms_mid = nms(midpoint_confidence_map, kernel=5)
    nms_mid_vis = (nms_mid > ref_threshold) * nms_mid
    # nms_mid = (nms_mid > 0.01) * nms_mid

    # ref
    line_vis, scores_vis = get_pred_lines(
        nms_mid_vis[0], midpoint_confidence_map[0], geometric_map[0], num_anchor=4
    )
    # line, scores = get_pred_lines(
    #     nms_mid[0], midpoint_confidence_map[0], geometric_map[0], num_anchor=4
    # )

    # rot
    if rot_map is not None:
        rot_map = rot_map[:, :7]
        rot_center_map_nms = nms(rot_map, kernel=5)
        rot_center_map_nms_vis = (rot_center_map_nms > rot_threshold) * rot_center_map_nms 
        # rot_center_map_nms = (rot_center_map_nms > 0.01) * rot_center_map_nms 

        rot_center_score_map_vis, rot_center_fold_class_vis = rot_center_map_nms_vis[0].max(dim=0)
        rot_center_idx_vis = rot_center_score_map_vis.nonzero()
        # rot_center_score_vis = rot_center_score_map_vis[rot_center_idx_vis[:,0], rot_center_idx_vis[:, 1]]
        # rot_center_fold_vis = rot_center_fold_class_vis[rot_center_idx_vis[:,0], rot_center_idx_vis[:, 1]]
        rot_center_idx_vis = rot_center_idx_vis[:, [1, 0]]
    
    # rot_center_score_map, rot_center_fold_class = rot_center_map_nms[0].max(dim=0)
    # rot_center_idx = rot_center_score_map.nonzero()
    # rot_center_score = rot_center_score_map[rot_center_idx[:,0], rot_center_idx[:, 1]]
    # rot_center_fold = rot_center_fold_class[rot_center_idx[:,0], rot_center_idx[:, 1]]
    # rot_center_idx = rot_center_idx[:, [1, 0]]

    if full_size:
        pad_x = (
            round((256 - (256 * original_w / original_h)) / 2)
            if round((256 - (256 * original_w / original_h)) / 2) > 5
            else 0
        )
        pad_y = (
            round((256 - (256 * original_h / original_w)) / 2)
            if round((256 - (256 * original_h / original_w)) / 2) > 5
            else 0
        )
        # line = (
        #     (line - torch.tensor([pad_x, pad_y, pad_x, pad_y]).cuda())
        #     / torch.tensor(
        #         [256 - 2 * pad_x, 256 - 2 * pad_y,
        #             256 - 2 * pad_x, 256 - 2 * pad_y]
        #     ).cuda()
        #     * torch.tensor([original_w, original_h, original_w, original_h]).cuda()
        # )
        line_vis = (
            (line_vis - torch.tensor([pad_x, pad_y, pad_x, pad_y]).cuda())
            / torch.tensor(
                [256 - 2 * pad_x, 256 - 2 * pad_y,
                    256 - 2 * pad_x, 256 - 2 * pad_y]
            ).cuda()
            * torch.tensor([original_w, original_h, original_w, original_h]).cuda()
        )
        # rot_center = (
        #     (rot_center_idx - torch.tensor([pad_x, pad_y]).cuda())
        #     / torch.tensor(
        #         [256 - 2 * pad_x, 256 - 2 * pad_y]
        #     ).cuda()
        #     * torch.tensor([original_w, original_h]).cuda()
        # )
        if rot_map is not None:
            rot_center_vis = (
                (rot_center_idx_vis - torch.tensor([pad_x, pad_y]).cuda())
                / torch.tensor(
                    [256 - 2 * pad_x, 256 - 2 * pad_y]
                ).cuda()
                * torch.tensor([original_w, original_h]).cuda()
            )
        img_pred = (original_img_show.clone().permute(1, 2, 0) * 255).numpy()

    else:
        img_pred = (img_show.permute(1, 2, 0) * 255).numpy()
        line_vis *= 2
        if rot_map is not None:
            rot_center_vis = rot_center_idx_vis * 2
            rot_center_idx_vis *= 2

    line_thickness = 3
    inner_line_thickness = 2
    circle_thickness = -1
    inner_circle_thickness = -1
    circle_radius = 4
    inner_circle_radius = 3
    line_vis = line_vis.reshape(-1, 2, 2)
    # line = line.reshape(-1, 2, 2)
    img_pred = img_pred[:, :, ::-1].astype(np.uint8).copy()
    img_pred_ref = copy.deepcopy(img_pred)
    img_pred_rot = copy.deepcopy(img_pred)
    img_pred_all = copy.deepcopy(img_pred)

    for i in range(len(line_vis)):
        cv2.line(
            img_pred_ref,
            line_vis[i][0].to(torch.int64).tolist(),
            line_vis[i][1].to(torch.int64).tolist(),
            color=[0, 255, 0],
            thickness=line_thickness,
        )
        cv2.line(
            img_pred_all,
            line_vis[i][0].to(torch.int64).tolist(),
            line_vis[i][1].to(torch.int64).tolist(),
            color=[0, 0, 0],
            thickness=line_thickness,
        )
        cv2.line(
            img_pred_ref,
            line_vis[i][0].to(torch.int64).tolist(),
            line_vis[i][1].to(torch.int64).tolist(),
            color=[0, 0, 0],
            thickness=inner_line_thickness,
        )
        cv2.line(
            img_pred_all,
            line_vis[i][0].to(torch.int64).tolist(),
            line_vis[i][1].to(torch.int64).tolist(),
            color=[0, 255, 0],
            thickness=inner_line_thickness,
        )
    if rot_map is not None:
        for j in range(len(rot_center_vis)):
            cv2.circle(
                img_pred_rot,
                rot_center_vis[j].to(torch.int64).tolist(),
                radius=circle_radius,
                color=(0, 0, 0),
                thickness=circle_thickness,
            )
            cv2.circle(
                img_pred_all,
                rot_center_vis[j].to(torch.int64).tolist(),
                radius=circle_radius,
                color=(0, 0, 0),
                thickness=circle_thickness,
            )
            cv2.circle(
                img_pred_rot,
                rot_center_vis[j].to(torch.int64).tolist(),
                radius=inner_circle_radius,
                color=(0, 0, 255),
                thickness=inner_circle_thickness,
            )
            cv2.circle(
                img_pred_all,
                rot_center_vis[j].to(torch.int64).tolist(),
                radius=inner_circle_radius,
                color=(0, 0, 255),
                thickness=inner_circle_thickness,
            )

    if rot_map is not None:
        return img_pred_all[:, :, ::-1], img_pred_ref[:, :, ::-1], img_pred_rot[:, :, ::-1]
    else:
        return img_pred_all[:, :, ::-1], img_pred_ref[:, :, ::-1]
    # return img_pred_ref[:, :, ::-1], img_pred_rot[:, :, ::-1], img_pred_all[:, :, ::-1], \
    #     line.reshape(-1, 4), scores, rot_center.reshape(-1,2), rot_center_score, rot_center_fold


def transform_json(input_data):
    transformed_data = []
    
    for item in input_data:
        transformed_item = {
            "filename": f"images/{item['filename']}" if 'filename' in item else "",
            "filename_ellipse": "",
            "filename_axis": "",
            "filename_reflection_mask": "",
            "width": item.get('width', 0),
            "height": item.get('height', 0),
            "ann": {
                "line": item.get('lines', []),
                "ellipse": []
            }
        }
        transformed_data.append(transformed_item)
    
    return transformed_data


class CustomCenterCrop(nn.Module):
    def __init__(self, size):
        super().__init__()
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
            
    def forward(self, img):
        """
        Args:
            img (Tensor): (B, C, H, W) 또는 (C, H, W) 형태의 이미지 텐서
        Returns:
            Tensor: Center crop된 이미지
        """
        if img.dim() == 3:
            c, h, w = img.shape
            th, tw = self.size
            
            # float를 텐서로 변환
            i = torch.round(torch.tensor((h - th) / 2, device=img.device)).long()
            j = torch.round(torch.tensor((w - tw) / 2, device=img.device)).long()
            
            return img[:, i:i + th, j:j + tw]
        else:  # img.dim() == 4 (배치 처리)
            b, c, h, w = img.shape
            th, tw = self.size
            
            # float를 텐서로 변환
            i = torch.round(torch.tensor((h - th) / 2, device=img.device)).long()
            j = torch.round(torch.tensor((w - tw) / 2, device=img.device)).long()
            
            return img[:, :, i:i + th, j:j + tw]
    
    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"


def _compute_ap_single(recall: torch.Tensor, precision: torch.Tensor) -> float:
    """Compute Average Precision for a single image

    Args:
        recall: Tensor of recall values
        precision: Tensor of precision values

    Returns:
        Average Precision score
    """
    # Ensure the recall and precision arrays start and end with 0 and 1
    recall = torch.cat(
        [
            torch.tensor([0.0], device=recall.device),
            recall,
            torch.tensor([1.0], device=recall.device),
        ]
    )
    precision = torch.cat(
        [
            torch.tensor([1.0], device=precision.device),
            precision,
            torch.tensor([0.0], device=precision.device),
        ]
    )

    # Compute the precision envelope
    for i in range(precision.shape[0] - 1, 0, -1):
        precision[i - 1] = torch.max(precision[i - 1], precision[i])

    # Calculate the area under the precision-recall curve
    indices = torch.where(recall[1:] != recall[:-1])[0]
    ap = torch.sum(
        (recall[indices + 1] - recall[indices]) * precision[indices + 1]
    ).item()
    return ap


def calculate_rot_center_ap_single(
        
    model: nn.Module,
    loader: DataLoader,
    thresholds: List[float],
    device: torch.device
) -> List[float]:
    """
    Calculate Average Precision for rotation center detection on a single GPU.

    Args:
        model (nn.Module): Neural network model producing "rot_center_map".
        loader (DataLoader): DataLoader for test/validation data.
        thresholds (List[float]): List of distance thresholds (squared distance) for AP calculation.
        device (torch.device): Device on which to run the computation.
        
    Returns:
        List[float]: AP values at each distance threshold in 'thresholds'.
    """
    model.eval()
    
    # Lists to accumulate true positives, false positives, and scores across the dataset
    all_tp = [[] for _ in range(len(thresholds))]  # each threshold has its own TP list
    all_fp = [[] for _ in range(len(thresholds))]  # each threshold has its own FP list
    all_scores = []  # all predicted scores (for sorting)
    total_gt = 0      # total number of ground-truth centers

    with torch.no_grad():
        for data in tqdm(loader, desc="Calculating Rotation Center AP", leave=True):
            # 1) Forward pass
            inputs = data["img"].to(device)
            outputs = model(inputs)
            
            # 2) Get and process rotation center predictions
            # shape: (B, 1, H, W) or similar
            rot_center_map = outputs["rot_center_map"]
            rot_center_nms = nms(rot_center_map, kernel=5)
            rot_center_nms = rot_center_nms * (rot_center_nms > 0.01)
            
            batch_size = inputs.size(0)
            _, _, height, width = rot_center_map.shape

            # 3) For each image in the batch, gather predictions and ground truths
            for b in range(batch_size):
                # Predicted coordinates: rot_center_nms[b] has shape (1, H, W) or (H, W)
                # 'nonzero()' returns [y, x]
                rot_center_coords = rot_center_nms[b].squeeze().nonzero(
                    as_tuple=False)
                scores = rot_center_map[b].squeeze()[rot_center_coords[:, 0], rot_center_coords[:, 1]]

                # Convert (y, x) to float so we can do scaling
                pred_centers = rot_center_coords.float()

                # Normalize predicted centers to the 128x128 space
                # Note: pred_centers[:, 0] is 'y', pred_centers[:, 1] is 'x'
                normalized = pred_centers.clone()
                normalized[:, 0] = pred_centers[:, 1] / width * TARGET_SIZE   # x in [0, TARGET_SIZE]
                normalized[:, 1] = pred_centers[:, 0] / height * TARGET_SIZE  # y in [0, TARGET_SIZE]
                pred_centers = normalized  # shape (N_pred, 2)

                # 4) Process ground-truth centers to the same 128x128 scale
                gt_centers = data['rot_centers'][b].to(device)  # shape: (N_gt, 2) (x, y in [0,1]?)
                original_h, original_w = data['original_shape'][b][:2]

                # Convert from normalized [0,1] to pixel coords
                gt_centers = gt_centers * torch.tensor(
                    [[original_w, original_h]], device=device
                )

                # Scale to TARGET_SIZE with margins (letterbox logic)
                max_length = max(original_w, original_h)
                scale_ratio = TARGET_SIZE / max_length
                margin = (TARGET_SIZE - (min(original_w, original_h) * scale_ratio)) / 2

                # If the height is the limiting dimension
                if max_length == original_h:
                    # x dimension = (x * scale_ratio) + margin
                    gt_centers[:, 0] = gt_centers[:, 0] * scale_ratio + margin
                    gt_centers[:, 1] = gt_centers[:, 1] * scale_ratio
                else:
                    # y dimension = (y * scale_ratio) + margin
                    gt_centers[:, 0] = gt_centers[:, 0] * scale_ratio
                    gt_centers[:, 1] = gt_centers[:, 1] * scale_ratio + margin

                n_gt = len(gt_centers)
                total_gt += n_gt  # accumulate total number of ground-truth points

                # 5) For each distance threshold, compute TP/FP
                for t_idx, threshold in enumerate(thresholds):
                    if len(pred_centers) > 0 and n_gt > 0:
                        # distances.shape = (N_pred, N_gt), squared distances
                        distances = ((pred_centers[:, None, :] - gt_centers[None, :, :]) ** 2).sum(dim=-1)
                        min_dists, matched_gt_idx = distances.min(dim=1)

                        # Initialize T/F positives
                        tp = torch.zeros(len(pred_centers), dtype=torch.float, device=device)
                        fp = torch.zeros(len(pred_centers), dtype=torch.float, device=device)

                        # Track which GTs are matched
                        matched_gt = torch.zeros(n_gt, dtype=torch.bool, device=device)

                        # Assign TPs/FPs based on threshold
                        for pred_idx, (dist, gt_idx) in enumerate(zip(min_dists, matched_gt_idx)):
                            # If you want to treat 'threshold' as a *squared distance*, keep this:
                            # if dist < threshold and not matched_gt[gt_idx]:
                            #
                            # If 'threshold' is the radial (Euclidean) distance in 128 space, do:
                            # if dist < threshold**2 and not matched_gt[gt_idx]:
                            if dist < threshold and not matched_gt[gt_idx]:
                                tp[pred_idx] = 1.0
                                matched_gt[gt_idx] = True
                            else:
                                fp[pred_idx] = 1.0
                    else:
                        # If no predictions, then there's no TP; if we do have predictions but no GT, all are FP
                        tp = torch.zeros(len(pred_centers), dtype=torch.float, device=device)
                        fp = torch.ones(len(pred_centers), dtype=torch.float, device=device) if len(pred_centers) > 0 else tp

                    # Accumulate for this threshold
                    all_tp[t_idx].append(tp)
                    all_fp[t_idx].append(fp)

                # We only store the scores once per image (not once per threshold)
                if len(scores) > 0:
                    all_scores.append(scores)

    # 6) After processing all images, compute AP for each threshold
    ap_results = []
    # If there were never any predicted scores, no predictions were made
    found_any_predictions = len(all_scores) > 0
    
    for t_idx, _ in enumerate(thresholds):
        if found_any_predictions:
            # Concatenate all TPs, FPs, and scores across the entire dataset
            tp = torch.cat(all_tp[t_idx], dim=0)
            fp = torch.cat(all_fp[t_idx], dim=0)
            scores = torch.cat(all_scores, dim=0)

            # Sort by descending confidence
            sorted_indices = torch.argsort(scores, descending=True)
            tp = tp[sorted_indices]
            fp = fp[sorted_indices]

            # Compute precision/recall
            tp_cumsum = torch.cumsum(tp, dim=0)
            fp_cumsum = torch.cumsum(fp, dim=0)

            recall = tp_cumsum / max(total_gt, 1)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)  # small eps to avoid /0

            # Calculate AP using your provided method
            ap = _compute_ap_single(recall, precision)
        else:
            ap = 0.0  # no predictions at all

        ap_results.append(ap)

    return ap_results


def calculate_mid_ap_single(
    model: nn.Module,
    loader: DataLoader,
    thresholds: List[float],
    device: torch.device
) -> List[float]:
    """
    Calculate Average Precision for midpoint detection on a single GPU.

    Requirements / Assumptions:
    - The model output dict has key 'midpoint_confidence_map' with shape (B, M, H, W).
    - We take max over the M dimension => shape (B, 1, H, W).
    - We apply NMS (or any local-max method) and filter out low confidence peaks.
    - GT midpoints are stored in data['midpoints'] with shape (max_point, 2) in the same H,W space.
      (Padding rows are typically zero, so we filter them out.)

    Args:
        model: Neural network model returning "midpoint_confidence_map".
        loader: DataLoader for the validation/test set.
        thresholds: Distance thresholds (interpreted as *squared* distances or direct pixel-distances).
        device: Computation device (e.g., torch.device('cuda:0')).

    Returns:
        List[float]: AP values for each threshold in `thresholds`.
    """
    model.eval()

    # Accumulators for TPs, FPs, scores, and total GT count
    all_tp = [[] for _ in range(len(thresholds))]
    all_fp = [[] for _ in range(len(thresholds))]
    all_scores = []
    total_gt = 0

    with torch.no_grad():
        for data in tqdm(loader, desc="Calculating Midpoint AP", leave=True):
            # 1) Forward pass
            inputs = data["img"].to(device)  # shape (B, C, H, W)
            outputs = model(inputs)

            # 2) Extract single-channel midpoint confidence by taking max across M
            # (B, M, H, W)
            # midpoint_conf_map = outputs["midpoint_confidence_map"]
            midpoint_conf_map = data['midpoint_confidence_map'].to(device)

            mid_conf_nms = nms(midpoint_conf_map, kernel=5)
            mid_conf_nms = mid_conf_nms * (mid_conf_nms > 0.01)
            batch_size, _, height, width = mid_conf_nms.shape

            num_lines = data["n_lines"]

            for b in range(batch_size):
                # 4) Find predicted midpoint coordinates (y, x)
                # shape (N_pred, 3) (Include Anchor coordinate), [256, 256]

                pred_coords_yx = mid_conf_nms[b].squeeze().nonzero(
                    as_tuple=False)
                # pred_coords_yx = mid_conf_nms[b].nonzero(
                #     as_tuple=False)
                scores = midpoint_conf_map[b].squeeze(
                )[pred_coords_yx[:, 0], pred_coords_yx[:, 1], pred_coords_yx[:, 2]]
                # scores = midpoint_conf_map[b][pred_coords_yx[:, 0],
                #                               pred_coords_yx[:, 1], pred_coords_yx[:, 2]]

                # Convert from (y, x) -> (x, y) in float
                pred_coords = pred_coords_yx[:, [2, 1]].float()  # (N_pred, 2)
                gt_mids = data["midpoints"][b][:num_lines[b]].to(
                    device).float()  # (N_gt, 2)

                # Normalize
                # Both based on map size
                pred_coords = normalize_coordinates(pred_coords, height, width)
                gt_mids = normalize_coordinates(gt_mids, height, width)
   
                # Update total GT count
                n_gt = len(gt_mids)
                total_gt += n_gt

                for t_idx, threshold in enumerate(thresholds):
                    if len(pred_coords) > 0 and n_gt > 0:
                        distances = ((pred_coords[:, None, :] - gt_mids[None, :, :]) ** 2).sum(dim=-1)
                        min_dists, matched_gt_idx = distances.min(dim=1)

                        tp = torch.zeros(len(pred_coords), dtype=torch.float, device=device)
                        fp = torch.zeros(len(pred_coords), dtype=torch.float, device=device)
                        matched_gt_flags = torch.zeros(n_gt, dtype=torch.bool, device=device)

                        for pred_idx, (dist, gt_idx) in enumerate(zip(min_dists, matched_gt_idx)):
                            if dist < threshold and not matched_gt_flags[gt_idx]:
                                tp[pred_idx] = 1.0
                                matched_gt_flags[gt_idx] = True
                            else:
                                fp[pred_idx] = 1.0
                    else:
                        # No GT => all predictions are FP (if any predictions exist)
                        tp = torch.zeros(len(pred_coords), dtype=torch.float, device=device)
                        fp = torch.ones(len(pred_coords), dtype=torch.float, device=device) if len(pred_coords) > 0 else tp

                    all_tp[t_idx].append(tp)
                    all_fp[t_idx].append(fp)

                # Collect the scores for sorting later
                if len(scores) > 0:
                    all_scores.append(scores)

    # 7) After processing the entire loader, compute AP for each threshold
    ap_results = []
    found_any_predictions = (len(all_scores) > 0)

    for t_idx in range(len(thresholds)):
        if found_any_predictions:
            # Concatenate TPs, FPs, and Scores
            tp_cat = torch.cat(all_tp[t_idx], dim=0)
            fp_cat = torch.cat(all_fp[t_idx], dim=0)
            scores_cat = torch.cat(all_scores, dim=0)

            # Sort by confidence descending
            sorted_indices = torch.argsort(scores_cat, descending=True)
            tp_sorted = tp_cat[sorted_indices]
            fp_sorted = fp_cat[sorted_indices]

            # Cumulative sums
            tp_cumsum = torch.cumsum(tp_sorted, dim=0)
            fp_cumsum = torch.cumsum(fp_sorted, dim=0)

            recall = tp_cumsum / max(total_gt, 1)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

            # Compute AP via your helper
            ap = _compute_ap_single(recall, precision)
        else:
            ap = 0.0

        ap_results.append(ap)

    return ap_results


def calculate_rot_fold_ap(
    model: nn.Module,
    loader: DataLoader,
    cfg,
    local_gpu_id: int
) -> List[float]:
    """Calculate Average Precision for rotation fold detection.
    
    This function uses the predicted rotation center map (outputs['rot_center_map']),
    but now it also considers the predicted fold class. A candidate prediction is considered 
    a true positive only if (a) its spatial distance is below threshold and (b) its predicted 
    fold class (derived from the first 7 channels using the mapping) exactly matches the 
    ground truth fold label.
    
    Ground truth fold labels are extracted from data['rot_fold_map_onehot'] (which is generated by 
    your get_one_hot_fold_map function) at the ground truth center locations. Only those 
    ground truth centers with a valid fold (i.e. from one of the first 7 channels) are used.
    """
    model.eval()
    ap_accumulator = APAccumulator(
        num_thresholds=len(cfg.threshold), device=local_gpu_id, distributed=cfg.distributed
    )

    # Define reverse mapping to convert predicted channel index to actual fold value.
    # This follows fold_to_channel = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6}
    reverse_mapping = {0: 0, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 8}

    with torch.no_grad():
        for data in tqdm(loader, desc="Calculating Rotation Fold AP", leave=False, disable=(cfg.rank != 0)):
            inputs = data["img"].to(local_gpu_id)
            outputs = model(inputs)

            # ---------------------------
            # Prediction extraction:
            # ---------------------------
            # Use only the first 7 channels (valid fold classes) from outputs["rot_center_map"]
            # Shape: (batch, 7, H, W)
            pred_map = outputs["rot_center_map"][:, :7, :, :]
            # Compute predicted score and predicted fold index per pixel.
            pred_scores, pred_fold_idx = pred_map.max(
                dim=1, keepdim=True)  # (batch, 1, H, W)
            # Apply non-maximum suppression and thresholding.
            pred_nms = nms(pred_scores, kernel=5)
            pred_nms = pred_nms * (pred_nms > 0.01)

            batch_size, _, height, width = pred_map.shape

            for b in range(batch_size):
                # Get candidate predicted center coordinates (nonzero positions in NMS result)
                coords = pred_nms[b].squeeze().nonzero()  # Each row: [y, x]

                if len(data['rot_centers'][b]) == 0:
                    continue

                if coords.numel() == 0:
                    # No predictions; update accumulator with empty predictions.
                    tp_list = [torch.zeros(0, device=local_gpu_id)
                               for _ in cfg.threshold]
                    fp_list = [torch.zeros(0, device=local_gpu_id)
                               for _ in cfg.threshold]
                    ap_accumulator.update(
                        tp_list, fp_list, torch.zeros(0, device=local_gpu_id), 0)
                    continue

                # Extract predicted scores at candidate locations.
                scores = pred_scores[b].squeeze()[coords[:, 0], coords[:, 1]]
                # Extract predicted fold indices at candidate locations.
                pred_fold_idxs = pred_fold_idx[b].squeeze()[
                    coords[:, 0], coords[:, 1]]
                # Convert predicted fold indices to actual fold values.
                # (Note: since pred_fold_idxs is a tensor, we use a list comprehension to map each element.)
                pred_folds = torch.tensor(
                    [reverse_mapping[int(idx)] for idx in pred_fold_idxs],
                    device=local_gpu_id, dtype=torch.int64
                )
                # Convert candidate coordinates to normalized space.
                # (N, 2) where columns are [y, x]
                pred_centers = coords.float()
                normalized = pred_centers.clone()
                normalized[:, 0] = pred_centers[:, 1] / \
                    width * TARGET_SIZE  # x coordinate
                normalized[:, 1] = pred_centers[:, 0] / \
                    height * TARGET_SIZE  # y coordinate
                pred_centers = normalized  # (N, 2) in TARGET_SIZE scale

                # ---------------------------
                # Ground truth extraction:
                # ---------------------------
                # Get ground truth rotation centers (assumed provided in data["rot_centers"])
                gt_centers = data["rot_centers"][b].to(local_gpu_id)
                # Convert to TARGET_SIZE scale similarly to predicted centers.
                gt_centers[:, 0] = gt_centers[:, 0] * \
                    width * (TARGET_SIZE / width)
                gt_centers[:, 1] = gt_centers[:, 1] * \
                    height * (TARGET_SIZE / height)

                # Get the ground truth fold map for this image.
                # Shape: (8, H, W)
                gt_fold_map = data["rot_fold_map_onehot"][b].to(local_gpu_id)
                # For each ground truth center, extract the fold label.
                gt_folds = []
                valid_gt_idxs = []
                for i in range(gt_centers.shape[0]):
                    # Convert ground truth center coordinates to pixel indices (rounding)
                    y = int(
                        round(gt_centers[i, 1].item() * height / TARGET_SIZE))
                    x = int(
                        round(gt_centers[i, 0].item() * width / TARGET_SIZE))
                    if y < 0 or y >= height or x < 0 or x >= width:
                        continue
                    # Use only if there is a valid fold (i.e. one of the first 7 channels is active)
                    if gt_fold_map[:7, y, x].sum() < 0.5:
                        continue  # Skip ground truth with "no fold"
                    # Get ground truth fold index (0–6) then map to actual fold value.
                    fold_idx = gt_fold_map[:7, y, x].argmax().item()
                    gt_folds.append(reverse_mapping[fold_idx])
                    valid_gt_idxs.append(i)
                if len(valid_gt_idxs) == 0:
                    n_gt = 0
                else:
                    gt_centers = gt_centers[valid_gt_idxs]
                    gt_folds = torch.tensor(
                        gt_folds, device=local_gpu_id, dtype=torch.int64)
                    n_gt = len(gt_centers)

                # ---------------------------
                # Evaluate each threshold:
                # ---------------------------
                tp_list = []
                fp_list = []
                for threshold in cfg.threshold:
                    if pred_centers.shape[0] > 0 and n_gt > 0:
                        # Compute squared Euclidean distances between each predicted center and each ground truth center.
                        distances = (
                            (pred_centers[:, None, :] - gt_centers[None, :, :]) ** 2).sum(-1)
                        min_dists, matched_gt_idx = distances.min(dim=1)
                        # Initialize true positive (TP) and false positive (FP) flags.
                        tp = torch.zeros(
                            pred_centers.shape[0], dtype=torch.float, device=local_gpu_id)
                        fp = torch.zeros(
                            pred_centers.shape[0], dtype=torch.float, device=local_gpu_id)
                        matched_gt = torch.zeros(
                            n_gt, dtype=torch.bool, device=local_gpu_id)
                        for pred_idx, (dist, gt_idx) in enumerate(zip(min_dists, matched_gt_idx)):
                            # For a valid match, both the distance must be below threshold and the predicted
                            # fold must equal the ground truth fold.
                            if (dist < threshold/2) and (not matched_gt[gt_idx]) and (pred_folds[pred_idx] == gt_folds[gt_idx]):
                                tp[pred_idx] = 1
                                matched_gt[gt_idx] = True
                            else:
                                fp[pred_idx] = 1
                    else:
                        tp = torch.zeros(
                            pred_centers.shape[0], dtype=torch.float, device=local_gpu_id)
                        fp = torch.ones(
                            pred_centers.shape[0], dtype=torch.float, device=local_gpu_id)
                    tp_list.append(tp)
                    fp_list.append(fp)
                if pred_centers.shape[0] > 0:
                    ap_accumulator.update(tp_list, fp_list, scores, n_gt)
                else:
                    ap_accumulator.update(
                        [torch.zeros(0, device=local_gpu_id)
                         for _ in cfg.threshold],
                        [torch.zeros(0, device=local_gpu_id)
                         for _ in cfg.threshold],
                        torch.zeros(0, device=local_gpu_id),
                        n_gt,
                    )
    # Gather results from all processes.
    ap_accumulator.gather_tpfp()
    # Compute AP values.
    ap_results = ap_accumulator.compute_ap()
    return ap_results
