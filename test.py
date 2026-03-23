"""
Test script for axis-level symmetry detection model.
Handles model evaluation and distributed testing setup.
"""

# Standard library imports
import os
import warnings
import argparse
import builtins as __builtin__

# Third-party imports
import torch

# Local imports
from utils.utils import calculate_ap, calculate_rot_center_ap, calculate_rot_fold_ap
from utils.dataset_factory import generate_dataset
from utils.model_factory import generate_model


def get_config():
    """Get testing configuration from specified config file

    Returns:
        Config object containing all testing parameters
    """
    parser = argparse.ArgumentParser(
        description='Test script for axis-level symmetry detection')
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--weight', type=str, default=None,
                        help='Path to model weight (overrides config)')
    args = parser.parse_args()

    # Import config from specified path
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_module", args.cfg)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    cfg = config_module.config
    if args.weight is not None:
        cfg.weight = args.weight
    return cfg


def test(cfg) -> None:
    """Main testing function
    
    Handles the complete testing process including:
        - Distributed setup
        - Model loading
        - Dataset preparation
        - Evaluation
        - Results reporting
    
    Args:
        cfg: Configuration object containing test parameters
    """
    # Initialize distributed testing environment
    init_for_distributed(cfg)
    local_gpu_id = 0

    # Prepare test dataset and model
    test_loader = generate_dataset(cfg, "test")
    model = generate_model(cfg, local_gpu_id)
    
    # Load pretrained weights
    weight = torch.load(cfg.weight, map_location=f'cuda:{local_gpu_id}')
    # Handle DDP-saved weights (with 'module.' prefix) when loading into non-DDP model
    if not cfg.distributed and any(k.startswith('module.') for k in weight.keys()):
        weight = {k.replace('module.', ''): v for k, v in weight.items()}
    model.load_state_dict(weight)

    # Calculate and print AP results
    if cfg.dataset != "symcoco_rot":
        ap_results_test = calculate_ap(model, test_loader, cfg, local_gpu_id)

        print(
            f"AP(test) 5/10/15: {ap_results_test[0]:.3f}/{ap_results_test[1]:.3f}/{ap_results_test[2]:.3f}"
        )
    if cfg.include_rot and cfg.dataset != "symcoco_ref":
        ap_result_test_rot_center = calculate_rot_center_ap(model, test_loader, cfg, local_gpu_id)
        ap_result_test_rot_fold = calculate_rot_fold_ap(model, test_loader, cfg, local_gpu_id)
        print(
            f"Rot Center AP(test) 5/10/15: {ap_result_test_rot_center[0]:.3f}/{ap_result_test_rot_center[1]:.3f}/{ap_result_test_rot_center[2]:.3f}"
        )
        print(
            f"Rot Fold AP(test) 5/10/15: {ap_result_test_rot_fold[0]:.3f}/{ap_result_test_rot_fold[1]:.3f}/{ap_result_test_rot_fold[2]:.3f}"
        )



def init_for_distributed(cfg) -> None:
    """Initialize distributed testing environment
    
    Sets up distributed testing parameters including:
        - Rank
        - World size
        - GPU device
        - Backend
        - Process group
    
    Args:
        cfg: Configuration object to store distributed parameters
    """
    # Check environment variables for distributed setup
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        cfg.rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ["WORLD_SIZE"])
        cfg.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        cfg.rank = int(os.environ["SLURM_PROCID"])
        cfg.gpu = cfg.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        cfg.distributed = False
        cfg.rank = 0
        cfg.gpu = 0
        return

    # Setup distributed environment
    torch.cuda.set_device(cfg.gpu)
    cfg.dist_backend = "nccl"
    print("| distributed init (rank {}): {}".format(cfg.rank, "env://"), flush=True)
    
    # Initialize process group
    torch.distributed.init_process_group(
        backend=cfg.dist_backend,
        init_method=cfg.dist_url,
        world_size=cfg.world_size,
        rank=cfg.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(cfg.rank == 0)


def setup_for_distributed(is_master: bool) -> None:
    """Configure printing behavior for distributed testing
    
    Disables printing for non-master processes to avoid duplicate output
    
    Args:
        is_master: Whether current process is the master process
    """
    builtin_print = __builtin__.print

    def print(*args, **kwargs) -> None:
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # Suppress warnings
    cfg = get_config()  # Get testing configuration
    test(cfg)  # Start testing process
