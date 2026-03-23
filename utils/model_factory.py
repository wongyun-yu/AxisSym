import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def generate_model(cfg, local_gpu_id, binary_center=False):
    """Generate model and wrap it with DDP"""
    if cfg.equiv:
        from network.equiv_ref_sym import EquivRefSym
        model = EquivRefSym(
            device=local_gpu_id,
            map_size=cfg.map_size,
            channel_multiplier=cfg.channel_multiplier,
            num_anchor=cfg.num_anchor,
            rot_group=cfg.rot_group,
            include_ref=cfg.include_ref,
            reg_conv=cfg.reg_conv,  
            loc_conv=cfg.loc_conv,
            include_rot=cfg.include_rot,
            rot_center_conv=cfg.rot_center_conv,
            rot_fold_conv=cfg.rot_fold_conv,
            dropout=cfg.dropout,
            ref_feature_choice=cfg.ref_feature,     
            rot_feature_choice=cfg.rot_feature,     
            backbone=cfg.backbone,
            ref_matching_channels=cfg.ref_matching_channels,  
            rot_matching_channels=cfg.rot_matching_channels,  
            matching_resolution=cfg.matching_resolution,
            ref_matching_patches_size=cfg.ref_matching_patches_size,  
            rot_matching_patches_size=cfg.rot_matching_patches_size,  
            fix_seed=cfg.fix_seed,
            disable_permutation=cfg.disable_permutation,  
            branch=cfg.branch,
            branch_pad=cfg.branch_padding,  
            freeze_backbone=cfg.freeze_backbone,
            rot_feature_pooling=cfg.rot_feature_pooling,
            orientational_anchor=cfg.orientational_anchor, 
            binary_center=binary_center,
        )
    else:
        raise ValueError("Only equivariant model (cfg.equiv=True) is supported.")

    model = model.cuda(local_gpu_id)
    if cfg.distributed:
        model = DDP(module=model, device_ids=[local_gpu_id], find_unused_parameters=True)
    return model