"""
Main training script for axis-level symmetry detection model.
Handles distributed training, model management, and result logging.
"""

# Standard library imports
import os
import warnings
import argparse
from typing import Dict, List, Optional, Union

# Third-party imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Local imports
from utils.loss import EquivRefSymLoss
from utils.utils import calculate_ap, calculate_rot_center_ap, calculate_mid_ap_single, calculate_rot_fold_ap
from utils.dataset_factory import generate_dataset
from utils.model_factory import generate_model


# Constants (if any)
SAVE_INTERVAL = 50
LEARNING_RATE_DECAY_POINTS = [0.5, 0.8]  # Points at which to decay learning rate


def get_config():
    """Get training configuration from specified config file
    
    Returns:
        Config object containing all training parameters
    """
    parser = argparse.ArgumentParser(
        description='Training script for axis-level symmetry detection')
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to config file')
    args = parser.parse_args()

    # Import config from specified path
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_module", args.cfg)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    return config_module.config

def adjust_lr(lr: float, epoch: int, max_epoch: int) -> float:
    """Adjust learning rate based on epoch
    
    Args:
        lr: Current learning rate
        epoch: Current epoch number
        max_epoch: Maximum number of epochs
        
    Returns:
        Adjusted learning rate
    """
    if epoch in [int(max_epoch * LEARNING_RATE_DECAY_POINTS[0]), 
                 int(max_epoch * LEARNING_RATE_DECAY_POINTS[1])]:
        lr /= 10
    return lr

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_epoch: int
) -> Dict[str, float]:
    """Train model for one epoch
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Computation device
        epoch: Current epoch number
        max_epoch: Maximum number of epochs
        
    Returns:
        Dictionary containing average losses:
            - total: Total loss
            - midpoint: Midpoint detection loss
            - geometric: Geometric parameters loss
            - rho: Rho parameter loss
            - theta: Theta parameter loss
    """
    model.train()
    running_loss = {'total': 0.0, 'rot_center': 0.0, 'midpoint': 0.0, 'geometric': 0.0, 'rho': 0.0, 'theta': 0.0}

    lr = adjust_lr(optimizer.param_groups[0]['lr'], epoch, max_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    for data in tqdm(train_loader, desc="Train", leave=False, disable=(cfg.rank != 0)): 
        inputs = data['img'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss, rot_center, midpoint, geometric, rho, theta = criterion(outputs, data)  
        loss.backward()
        optimizer.step()

        running_loss['total'] += loss.item()
        running_loss['rot_center'] += rot_center.item()
        running_loss['midpoint'] += midpoint.item()
        running_loss['geometric'] += geometric.item()
        running_loss['rho'] += rho.item()
        running_loss['theta'] += theta.item()

    return {k: v / len(train_loader) for k, v in running_loss.items()}

def val(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg
) -> Dict[str, float]:
    """Validate model on validation dataset
    
    Args:
        model: Neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Computation device
        cfg: Configuration object
        
    Returns:
        Dictionary containing average validation losses
    """
    model.eval()
    running_loss = {'total': 0.0, 'rot_center': 0.0, 'midpoint': 0.0, 'geometric': 0.0, 'rho': 0.0, 'theta': 0.0}

    with torch.no_grad():
        for data in tqdm(val_loader, desc="Test", leave=False, disable=(cfg.rank != 0)):
            inputs = data['img'].to(device)
            outputs = model(inputs)
            loss, rot_center, midpoint, geometric, rho, theta = criterion(outputs, data)

            running_loss['total'] += loss.item()
            running_loss['rot_center'] += rot_center.item()
            running_loss['midpoint'] += midpoint.item()
            running_loss['geometric'] += geometric.item()
            running_loss['rho'] += rho.item()
            running_loss['theta'] += theta.item()

    return {k: v / len(val_loader) for k, v in running_loss.items()}

def train(cfg) -> None:
    """Main training function
    
    Manages the entire training process including:
        - Distributed training setup
        - Model initialization
        - Training loop
        - Validation
        - Model checkpointing
        - Logging
    
    Args:
        cfg: Configuration object containing all training parameters
    """
    # Initialize distributed training environment
    init_for_distributed(cfg)
    local_gpu_id = cfg.gpu

    # Initialize wandb (optional)
    if cfg.rank == 0 and HAS_WANDB:
        wandb.init(project='reflection symmetry detection', config=cfg)
        if cfg.run_name is not None:
            wandb.run.name = cfg.run_name
            wandb.run.save()

    # Generate datasets
    train_loader, train_sampler = generate_dataset(cfg, 'train')
    val_loader = generate_dataset(cfg, 'val', different_eval = cfg.dataset_val) 
    test_loader = generate_dataset(cfg, 'test', different_eval = cfg.dataset_test)

    # Generate model
    binary_center = True if cfg.dataset_test == 'symcoco_rot_final.json' else False
    model = generate_model(cfg, local_gpu_id, binary_center)
    if cfg.pretrained_weights is not None:
        print('loading pretrained weights')
        weight = torch.load(cfg.pretrained_weights)
        model.load_state_dict(weight)

    # Define loss function and optimizer
    criterion = EquivRefSymLoss(device=local_gpu_id, 
                                bce_weight=cfg.bce_weight,
                                mid_weight=cfg.mid_weight, 
                                rho_weight=cfg.rho_weight, 
                                theta_weight=cfg.theta_weight, 
                                rot_center_weight=cfg.rot_center_weight,
                                include_rot=cfg.include_rot,
                                include_ref=cfg.include_ref,
                                use_focal_loss=cfg.use_focal_loss,
                                use_focal_loss_ref=cfg.use_focal_loss_ref,
                                use_focal_loss_rot=cfg.use_focal_loss_rot,
                                alpha=cfg.alpha,
                                gamma=cfg.gamma,
                                binary_center=binary_center)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay)

    best_val_ap = 0
    best_test_ap = 0
    save_epoch = int(cfg.save_epoch)

    try:
        for epoch in range(cfg.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if epoch == save_epoch:
                best_val_ap = 0
                best_test_ap = 0

            # Train
            model.train()
            """Change to train_loader"""
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, local_gpu_id, epoch, cfg.epochs)

            # Validation
            test_loss = val(model, val_loader, criterion, local_gpu_id, cfg)

            # AP Calculation
            if cfg.include_ref:
                ap_results_val = calculate_ap(model, val_loader, cfg, local_gpu_id)
            else:
                ap_results_val = 0.0, 0.0, 0.0
            if cfg.include_rot:
                ap_result_val_rot_center = calculate_rot_center_ap(model, val_loader, cfg, local_gpu_id, binary_center)
                if not binary_center:
                    ap_result_val_rot_fold = calculate_rot_fold_ap(model, val_loader, cfg, local_gpu_id)
                else:
                    ap_result_val_rot_fold = 0.0, 0.0, 0.0
            else:
                ap_result_val_rot_center = None
                ap_result_val_rot_fold = None
            if cfg.dataset != 'wireframe':
                if cfg.include_ref:
                    ap_results_test = calculate_ap(
                        model, test_loader, cfg, local_gpu_id)
                else:
                    ap_results_test = ap_results_val
                if cfg.include_rot:
                    ap_result_test_rot_center = calculate_rot_center_ap(model, test_loader, cfg, local_gpu_id, binary_center)
                    if not binary_center:
                        ap_result_test_rot_fold = calculate_rot_fold_ap(model, test_loader, cfg, local_gpu_id)
                    else:
                        ap_result_test_rot_fold = 0.0, 0.0, 0.0
                else:
                    ap_result_test_rot_center = None
                    ap_result_test_rot_fold = None
            else:
                if cfg.include_ref:
                    ap_results_test = ap_results_val    
                else:
                    ap_results_test = 0.0, 0.0, 0.0
                if cfg.include_rot:
                    ap_result_test_rot_center = ap_result_val_rot_center
                    ap_result_test_rot_fold = ap_result_val_rot_fold
                else:
                    ap_result_test_rot_center = None
                    ap_result_test_rot_fold = None


            # Log and save results
            if cfg.rank == 0:
                log_results(epoch, 
                            train_loss, 
                            test_loss, 
                            ap_results_val, 
                            ap_results_test, 
                            ap_result_val_rot_center, 
                            ap_result_test_rot_center,
                            ap_result_val_rot_fold,
                            ap_result_test_rot_fold)

                save_model(model, 
                           epoch, 
                           save_epoch, 
                           ap_results_val, 
                           ap_results_test, 
                           best_val_ap, 
                           best_test_ap, 
                           cfg)

            best_val_ap = max(best_val_ap, ap_results_val[-1])
            best_test_ap = max(best_test_ap, ap_results_test[-1])

            torch.cuda.synchronize()
            if cfg.distributed:
                dist.barrier()
            torch.cuda.empty_cache()

        print("Training completed")
        if cfg.rank == 0 and HAS_WANDB:
            wandb.finish()
    except Exception as e:
        print(f"Error on rank {cfg.rank}: {str(e)}")
        raise e
    finally:
        if cfg.distributed:
            dist.destroy_process_group()

def log_results(
    epoch: int,
    train_loss: Dict[str, float],
    test_loss: Dict[str, float],
    ap_results_val: Optional[List[float]],
    ap_results_test: Optional[List[float]],
    ap_result_val_rot_center=None,
    ap_result_test_rot_center=None,
    ap_result_val_rot_fold=None,
    ap_result_test_rot_fold=None
) -> None:
    """Log training results to wandb and console
    
    Args:
        epoch: Current epoch number
        train_loss: Dictionary of training losses
        test_loss: Dictionary of validation losses
        ap_results_val: List of AP values for validation set
        ap_results_test: List of AP values for test set
        ap_result_val_rot_center: List of AP values for validation set (rotation center)
        ap_result_test_rot_center: List of AP values for test set (rotation center)
        ap_result_val_rot_fold: List of AP values for validation set (rotation fold)
        ap_result_test_rot_fold: List of AP values for test set (rotation fold)
    """
    if HAS_WANDB:
        wandb.log({f'Loss/train/{k}': v for k, v in train_loss.items()}, step=epoch)
        wandb.log({f'Loss/test/{k}': v for k, v in test_loss.items()}, step=epoch)

        if ap_results_val is not None:
            for i, ap in enumerate(ap_results_val):
                wandb.log({f'AP/val/{(i+1)*5}': ap}, step=epoch)
        if ap_results_test is not None:
            for i, ap in enumerate(ap_results_test):
                wandb.log({f'AP/test/{(i+1)*5}': ap}, step=epoch)
        if ap_result_val_rot_center is not None:
            for i, ap in enumerate(ap_result_val_rot_center):
                wandb.log({f'AP/val/rot_center/{(i+1)*5}': ap}, step=epoch)
        if ap_result_test_rot_center is not None:
            for i, ap in enumerate(ap_result_test_rot_center):
                wandb.log({f'AP/test/rot_center/{(i+1)*5}': ap}, step=epoch)
        if ap_result_val_rot_fold is not None:
            for i, ap in enumerate(ap_result_val_rot_fold):
                wandb.log({f'AP/val/rot_fold/{(i+1)*5}': ap}, step=epoch)
        if ap_result_test_rot_fold is not None:
            for i, ap in enumerate(ap_result_test_rot_fold):
                wandb.log({f'AP/test/rot_fold/{(i+1)*5}': ap}, step=epoch)

    print(f"\nEpoch {epoch + 1}/{cfg.epochs}")
    print(f"Train Loss: {train_loss['total']:.4f}, Test Loss: {test_loss['total']:.4f}")
    if ap_results_val is not None:
        print(f"AP(val) 5/10/15: {ap_results_val[0]:.3f}/{ap_results_val[1]:.3f}/{ap_results_val[2]:.3f}")
    if ap_results_test is not None:
        print(f"AP(test) 5/10/15: {ap_results_test[0]:.3f}/{ap_results_test[1]:.3f}/{ap_results_test[2]:.3f}")
    if ap_result_val_rot_center is not None:
        print(f"AP(val) rot_center 5/10/15: {ap_result_val_rot_center[0]:.3f}/{ap_result_val_rot_center[1]:.3f}/{ap_result_val_rot_center[2]:.3f}")
    if ap_result_test_rot_center is not None:
        print(f"AP(test) rot_center 5/10/15: {ap_result_test_rot_center[0]:.3f}/{ap_result_test_rot_center[1]:.3f}/{ap_result_test_rot_center[2]:.3f}")
    if ap_result_val_rot_fold is not None:
        print(f"AP(val) rot_fold 5/10/15: {ap_result_val_rot_fold[0]:.3f}/{ap_result_val_rot_fold[1]:.3f}/{ap_result_val_rot_fold[2]:.3f}")
    if ap_result_test_rot_fold is not None:
        print(f"AP(test) rot_fold 5/10/15: {ap_result_test_rot_fold[0]:.3f}/{ap_result_test_rot_fold[1]:.3f}/{ap_result_test_rot_fold[2]:.3f}")

def save_model(
    model: nn.Module,
    epoch: int,
    save_epoch: int,
    ap_results_val: List[float],
    ap_results_test: List[float],
    best_val_ap: float,
    best_test_ap: float,
    cfg
) -> None:
    """Save model checkpoint if performance improves
    
    Args:
        model: Neural network model
        epoch: Current epoch number
        ap_results_val: AP values for validation set
        ap_results_test: AP values for test set
        best_val_ap: Best validation AP so far
        best_test_ap: Best test AP so far
        cfg: Configuration object
    """
    save_model = False
    current_val_ap = ap_results_val[2] 
    current_test_ap = ap_results_test[2] 

    if (current_val_ap > best_val_ap) and (epoch > save_epoch - 1):
        save_model = True
        print(f"New best validation AP: {current_val_ap:.4f}")
    if (current_test_ap > best_test_ap) and (epoch > save_epoch - 1):
        save_model = True
        print(f"New best test AP: {current_test_ap:.4f}")

    if (
        save_model
        or ((epoch + 1) % SAVE_INTERVAL == 0)
    ):
        model_name = f'{cfg.run_name}_epoch{epoch + 1}_val{current_val_ap:.4f}_test{current_test_ap:.4f}.pt'
        model_save_pth = os.path.join('weights/', model_name)
        torch.save(model.state_dict(), model_save_pth)
        print(f'Model saved at epoch {epoch+1}')

def init_for_distributed(cfg) -> None:
    """Initialize distributed training environment
    
    Sets up distributed training parameters including:
        - Rank
        - World size
        - GPU device
        - Backend
        - Process group
    
    Args:
        cfg: Configuration object to store distributed training parameters
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cfg.rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ['WORLD_SIZE'])
        cfg.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        cfg.rank = int(os.environ['SLURM_PROCID'])
        cfg.gpu = cfg.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        cfg.distributed = False
        cfg.rank = 0
        cfg.gpu = 0
        return

    torch.cuda.set_device(cfg.gpu)
    cfg.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(cfg.rank, 'env://'), flush=True)
    torch.distributed.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
                                         world_size=cfg.world_size, rank=cfg.rank)
    torch.distributed.barrier()
    setup_for_distributed(cfg.rank == 0)

def setup_for_distributed(is_master: bool) -> None:
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # Suppress warnings
    cfg = get_config()  # Get training configuration
    train(cfg)  # Start training process
