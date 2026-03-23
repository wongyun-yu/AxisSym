from dataset.dendi.dendi_dataset import AxisSymDataset

import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def worker_init_fn():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def generate_dataset(cfgs, mode, different_eval=None):

    if cfgs.dataset == 'dendi':
        train_data_root = 'data/dendi_synthetic_rot_654654_233455' if cfgs.synthetic else 'data/dendi'
        val_data_root = 'data/dendi'
        test_data_root = 'data/dendi'
        ann_file = {'train': 'train_test_val_rot_final.json',
                    'val': 'val_rot_final.json',
                    'test': 'test_rot_final.json'}

    elif cfgs.dataset == 'sdrw':
        train_data_root = 'data'
        val_data_root = 'data'
        test_data_root = 'data'
        ann_file = {
            'train': 'cat_LDRS_SDRW_annotations_train_revised.json',
            'val': 'SDRW_annotations_test.json',
            'test': 'SDRW_annotations_test.json'}
    elif cfgs.dataset == 'ldrs':
        train_data_root = 'data'
        val_data_root = 'data'
        test_data_root = 'data'
        ann_file = {
            'train': 'cat_LDRS_SDRW_annotations_train_revised.json',
            'val': 'LDRS_annotations_test.json',
            'test': 'LDRS_annotations_test.json'}
    if different_eval is not None:
        if mode == 'val':
            val_data_root = 'data'
            ann_file['val'] = different_eval
        elif mode == 'test':
            test_data_root = 'data'
            ann_file['test'] = different_eval


    if mode == 'train':
        train_dataset = AxisSymDataset(ann_file=ann_file['train'],
                                    split='train',
                                    data_root=train_data_root,
                                    input_size=cfgs.input_size,
                                    sigma=cfgs.sigma,
                                    rot_center_sigma=cfgs.rot_center_sigma,
                                    kernel_size=cfgs.kernel_size,
                                    map_size=cfgs.map_size,
                                    num_anchor=cfgs.num_anchor,
                                    fix_seed=cfgs.fix_seed,
                                    resize=cfgs.resize,
                                    orientational_anchor=cfgs.orientational_anchor,
                                    num_data=cfgs.num_data)

        if cfgs.distributed:
            train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True, seed=42)
            train_loader = DataLoader(train_dataset,
                                    batch_size=int(cfgs.batch_size / cfgs.world_size),
                                    shuffle=False,
                                    num_workers=int(cfgs.num_workers / cfgs.world_size),
                                    collate_fn=train_dataset.collate_fn,
                                    pin_memory=True,
                                    sampler=train_sampler,
                                    worker_init_fn=worker_init_fn if cfgs.fix_seed else None)
        else:
            train_sampler = None
            train_loader = DataLoader(train_dataset,
                                    batch_size=cfgs.batch_size,
                                    shuffle=True,
                                    num_workers=cfgs.num_workers,
                                    collate_fn=train_dataset.collate_fn,
                                    pin_memory=True,
                                    worker_init_fn=worker_init_fn if cfgs.fix_seed else None)
        return train_loader, train_sampler

    elif mode == 'val':
        valid_dataset = AxisSymDataset(ann_file=ann_file['val'],
                                    split='val',
                                    data_root=val_data_root,
                                    input_size=cfgs.input_size,
                                    sigma=cfgs.sigma,
                                    rot_center_sigma=cfgs.rot_center_sigma,
                                    kernel_size=cfgs.kernel_size,
                                    map_size=cfgs.map_size,
                                    num_anchor=cfgs.num_anchor,
                                    fix_seed=cfgs.fix_seed,
                                    resize=cfgs.resize,
                                    orientational_anchor=cfgs.orientational_anchor,)

        if cfgs.distributed:
            valid_sampler = DistributedSampler(dataset=valid_dataset, shuffle=False, seed=42)
            val_loader = DataLoader(valid_dataset,
                                        batch_size=int(cfgs.batch_size / cfgs.world_size),
                                        shuffle=False,
                                        num_workers=int(cfgs.num_workers / cfgs.world_size),
                                        collate_fn=valid_dataset.collate_fn,
                                        pin_memory=True,
                                        sampler=valid_sampler)
        else:
            val_loader = DataLoader(valid_dataset,
                                        batch_size=cfgs.batch_size,
                                        shuffle=False,
                                        num_workers=cfgs.num_workers,
                                        collate_fn=valid_dataset.collate_fn,
                                        pin_memory=True)
        return val_loader

    elif mode == 'test':
        test_dataset = AxisSymDataset(ann_file=ann_file['test'],
                                   split='val',
                                   data_root=test_data_root,
                                   input_size=cfgs.input_size,
                                   sigma=cfgs.sigma,
                                   rot_center_sigma=cfgs.rot_center_sigma,
                                   kernel_size=cfgs.kernel_size,
                                   map_size=cfgs.map_size,
                                   num_anchor=cfgs.num_anchor,
                                   fix_seed=cfgs.fix_seed,
                                   resize=cfgs.resize,
                                   orientational_anchor=cfgs.orientational_anchor,)

        if cfgs.distributed:
            test_sampler = DistributedSampler(dataset=test_dataset, shuffle=False, seed=42)
            test_loader = DataLoader(test_dataset,
                                     batch_size=int(cfgs.batch_size / cfgs.world_size),
                                     shuffle=False,
                                     num_workers=int(cfgs.num_workers / cfgs.world_size),
                                     collate_fn=test_dataset.collate_fn,
                                     pin_memory=True,
                                     sampler=test_sampler)
        else:
            test_loader = DataLoader(test_dataset,
                                     batch_size=cfgs.batch_size,
                                     shuffle=False,
                                     num_workers=cfgs.num_workers,
                                     collate_fn=test_dataset.collate_fn,
                                     pin_memory=True)

        return test_loader
