import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class EquivRefSymLoss(nn.Module):
    """Loss function for equivariant axis-level symmetry detection

    Combines wbce loss for midpoint detection and regression losses for geometric parameters.
    """

    def __init__(self, 
                 device: torch.device, 
                 bce_weight: float, 
                 mid_weight: float, 
                 rho_weight: float, 
                 theta_weight: float, 
                 rot_center_weight: float,
                 include_rot: bool,
                 include_ref: bool,
                 use_focal_loss: bool = True,
                 use_focal_loss_ref: bool = False,
                 use_focal_loss_rot: bool = True,
                 alpha: float = 0.95,
                 gamma: float = 3.0,
                 binary_center: bool = False):
        
        super(EquivRefSymLoss, self).__init__()
        self.device = device
        self.bce_weight = bce_weight
        self.use_focal_loss = use_focal_loss
        self.use_focal_loss_ref = use_focal_loss_ref
        self.use_focal_loss_rot = use_focal_loss_rot
        self.alpha = alpha
        self.gamma = gamma

        self.rho_loss_fn = nn.SmoothL1Loss()
        self.theta_loss_fn = nn.L1Loss()

        # Loss weights
        self.THETA_WEIGHT = theta_weight
        self.RHO_WEIGHT = rho_weight
        self.MIDPOINT_WEIGHT = mid_weight
        self.ROT_CENTER_WEIGHT = rot_center_weight
        self.include_rot = include_rot
        self.include_ref = include_ref
        self.binary_center = binary_center
    def _compute_focal_loss(
        self, pred: torch.Tensor, target: torch.Tensor, num_points: int
    ) -> torch.Tensor:
        """Compute focal loss for confidence map

        Args:
            pred: Predicted confidence map
            target: Ground truth confidence map
            num_points: Total number of points in the batch

        Returns:
            Focal loss value
        """
        num_points = max(torch.sum(num_points).item(), 1)  # Avoid division by zero
        
        # Focal loss formula: -alpha * (1-p)^gamma * y * log(p) - (1-alpha) * p^gamma * (1-y) * log(1-p)
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_factor = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal_weight = (1 - pt) ** self.gamma
        
        loss = -alpha_factor * focal_weight * torch.log(torch.clamp(pt, min=1e-8, max=1-1e-8))
        # loss = -alpha_factor * torch.log(torch.clamp(pt, min=1e-8, max=1-1e-8))

        return torch.sum(loss) / num_points
        # return loss.mean()
    
    def _compute_focal_loss_multi_class(self, pred: torch.Tensor, target: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-class focal loss for a confidence map.

        Args:
            pred: Predicted logits for each class (B, 7, H, W)
                (Remember: no activation has been applied; they are raw logits.)
            target: Ground truth one-hot vector (B, 7, H, W)
            num_points: Tensor or scalar indicating the total number of points in the batch.
                        (This is used to normalize the loss; ensure it is nonzero.)

        Returns:
            Focal loss value (a scalar tensor).
        """


        # Ensure we do not divide by zero
        num_points = max(torch.sum(num_points).item(), 1)

        weight_map = target * 0.9 + 0.1
        reduction_map = (1 - pred) ** self.gamma
        loss = - weight_map * reduction_map * target * \
            torch.log(torch.clamp(pred, min=1e-8, max=1-1e-8))

        # Average the loss over all points.
        return torch.sum(loss) / num_points

    def _compute_wbce_loss(
        self, pred: torch.Tensor, target: torch.Tensor, num_lines: int
    ) -> torch.Tensor:
        """Compute wbce loss for midpoint confidence map

        Args:
            pred: Predicted confidence map
            target: Ground truth confidence map
            num_lines: Total number of lines in the batch

        Returns:
            WBCE loss value
        """
        num_lines = max(torch.sum(num_lines).item(), 1)  # Avoid division by zero
        loss = -(
            target * torch.log(torch.clamp(pred, min=1e-8, max=1-1e-8)) * self.bce_weight
            + (1 - target) * torch.log(torch.clamp(1 - pred, min=1e-8, max=1-1e-8))
        )
        return torch.sum(loss) / num_lines

    def _compute_wbce_loss_multi_class(self, pred: torch.Tensor, target: torch.Tensor, num_lines: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted multi-class cross entropy loss for midpoint confidence map.

        Args:
            pred: Predicted logits for each class (B, 7, H, W)
                (No activation has been applied.)
            target: Ground truth one-hot vector (B, 7, H, W)
            num_lines: Tensor or scalar indicating the total number of lines in the batch

        Returns:
            Weighted cross entropy loss value (a scalar tensor).
        """
        # Ensure we do not divide by zero.
        num_lines = max(torch.sum(num_lines).item(), 1)

        weight_map = target * 0.9 + 0.1
        loss = - weight_map * target * torch.log(torch.clamp(pred, min=1e-8, max=1-1e-8))

        # Average the loss over all lines (or points).
        return torch.sum(loss) / num_lines

    def _compute_geometric_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute regression losses for rho and theta

        Args:
            pred: Predicted geometric parameters
            target: Ground truth geometric parameters

        Returns:
            Tuple of (rho_loss, theta_loss)
        """
        batch_size = len(target)
        num_anchor = target.shape[1] // 2 if target.shape[1] != 1 else 1

        rho_loss = torch.tensor(0.0, device=pred.device)
        theta_loss = torch.tensor(0.0, device=pred.device)
        for i in range(batch_size):
            # Split rho and theta maps
            gt_rho = target[i][:num_anchor]
            gt_theta = target[i][num_anchor:]
            pred_rho = pred[i][:num_anchor]
            pred_theta = pred[i][num_anchor:]

            # Get valid points
            mask = gt_rho > 0
            if not mask.any(): # If no valid points (All points are background)
                continue

            # Compute losses for valid points
            valid_gt_rho = gt_rho[mask].to(pred.device)
            valid_pred_rho = pred_rho[mask]
            valid_gt_theta = gt_theta[mask].to(pred.device)
            valid_pred_theta = pred_theta[mask]

            if not (
                torch.isnan(valid_pred_rho).any() or torch.isnan(valid_pred_theta).any()
            ):
                rho_loss += self.rho_loss_fn(valid_pred_rho, valid_gt_rho)
                theta_loss += self.theta_loss_fn(valid_pred_theta, valid_gt_theta)

        return rho_loss / batch_size, theta_loss / batch_size

    def _compute_fold_loss(
        self, pred: torch.Tensor, target: torch.Tensor, num_centers: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross entropy loss for rotation fold prediction
        
        Args:
            pred: Predicted fold probabilities [B, 9, H, W] (already softmaxed)
            target: Ground truth one-hot encodings [B, 9, H, W]
            num_centers: Number of rotation centers per batch
        """
        num_centers = max(torch.sum(num_centers).item(), 1)

        # Just take log of the already softmaxed predictions
        loss = -(target * torch.log(torch.clamp(pred, min=1e-8)))

        # Sum loss only for valid positions and normalize by number of centers
        # Valid positions are those that have at least one positive class
        valid_mask = target.sum(dim=1) > 0
        return torch.sum(loss[valid_mask]) / num_centers

    def _compute_nonequiv_geometric_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute regression losses for non-equivariant geometric parameters"""
        batch_size = target.shape[0]
        rho_loss = torch.tensor(0.0, device=pred.device)
        theta_loss = torch.tensor(0.0, device=pred.device)

        for i in range(batch_size):
            # Get valid points
            mask = target[i][0] > 0
            if not mask.any():
                continue

            # Compute losses for valid points
            valid_gt_rho = target[i][0][mask]
            valid_pred_rho = pred[i][0][mask]
            valid_gt_theta = target[i][1][mask]
            valid_pred_theta = pred[i][1][mask]

            rho_loss += self.rho_loss_fn(valid_pred_rho, valid_gt_rho)
            theta_loss += self.theta_loss_fn(valid_pred_theta, valid_gt_theta)

        return rho_loss / batch_size, theta_loss / batch_size

    def forward(
        self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        
        """Forward pass

        Args:
            pred: Dictionary containing predicted values
            target: Dictionary containing ground truth values

        Returns:
            Tuple of (total_loss, midpoint_loss, geometric_loss, rho_loss, theta_loss)
        """
        # print(pred["rot_center_map"].max(dim=1, keepdim=True)[0].shape)
        # Compute midpoint confidence loss

        if self.include_rot and not self.include_ref:
            num_rot_centers = torch.tensor(
                [len(rot_center) for rot_center in target['rot_centers']])
            if self.use_focal_loss_rot:
                if self.binary_center:
                    rot_center_loss = self._compute_focal_loss(
                        pred["rot_center_map"],
                        target["rot_center_map"].to(self.device),
                        num_rot_centers,
                    )
                else:
                    rot_center_loss = self._compute_focal_loss_multi_class(
                        pred["rot_center_map"],
                        target["rot_fold_map_onehot"].to(self.device),
                        num_rot_centers,
                    )
            else:
                if self.binary_center:
                    rot_center_loss = self._compute_wbce_loss(
                        pred["rot_center_map"],
                        target["rot_center_map"].to(self.device),
                        num_rot_centers,
                    )
                else:
                    rot_center_loss = self._compute_wbce_loss_multi_class(
                        pred["rot_center_map"],
                        target["rot_fold_map_onehot"].to(self.device),
                        num_rot_centers,
                    ) 
            return rot_center_loss, \
                rot_center_loss, \
                torch.tensor(0.0, device=self.device), \
                torch.tensor(0.0, device=self.device), \
                torch.tensor(0.0, device=self.device), \
                torch.tensor(0.0, device=self.device)
            

        if self.use_focal_loss:
            midpoint_loss = self._compute_focal_loss(
                pred["midpoint_confidence_map"],
                target["midpoint_confidence_map"].to(self.device),
                target["n_lines"],
            )
            if self.include_rot:
                num_rot_centers = torch.tensor([len(rot_center) for rot_center in target['rot_centers']])
                rot_center_loss = self._compute_focal_loss_multi_class(
                    pred["rot_center_map"],
                    target["rot_fold_map_onehot"].to(self.device),
                    num_rot_centers,
                )
        else:
            if self.use_focal_loss_ref:
                midpoint_loss = self._compute_focal_loss(
                    pred["midpoint_confidence_map"],
                    target["midpoint_confidence_map"].to(self.device),
                    target["n_lines"],
                )
            else:
                midpoint_loss = self._compute_wbce_loss(
                    pred["midpoint_confidence_map"],
                    target["midpoint_confidence_map"].to(self.device),
                    target["n_lines"],
                )
            if self.include_rot:
                num_rot_centers = torch.tensor([len(rot_center) for rot_center in target['rot_centers']])
                if self.use_focal_loss_rot:
                    rot_center_loss = self._compute_focal_loss_multi_class(
                        pred["rot_center_map"],
                        target["rot_fold_map_onehot"].to(self.device),
                        num_rot_centers,
                    )
                else:
                    rot_center_loss = self._compute_wbce_loss_multi_class(
                        pred["rot_center_map"],
                        target["rot_fold_map_onehot"].to(self.device),
                        num_rot_centers,
                    )


        # Compute geometric losses
        if target["geometric_map"].shape[1] != 2:
            rho_loss, theta_loss = self._compute_geometric_loss(
                pred["geometric_map"], target["geometric_map"]
            )
        else:
            rho_loss, theta_loss = self._compute_nonequiv_geometric_loss(
                pred["geometric_map"], target["geometric_map"].to(self.device)
            )

        # Compute weighted losses
        geometric_loss = self.THETA_WEIGHT * theta_loss + self.RHO_WEIGHT * rho_loss
        
        if self.include_rot:
            total_loss = self.MIDPOINT_WEIGHT * midpoint_loss + geometric_loss + self.ROT_CENTER_WEIGHT * rot_center_loss
        else:
            total_loss = self.MIDPOINT_WEIGHT * midpoint_loss + geometric_loss
            rot_center_loss = torch.tensor(0.0, device=self.device)

        return total_loss, rot_center_loss, midpoint_loss, geometric_loss, rho_loss, theta_loss
