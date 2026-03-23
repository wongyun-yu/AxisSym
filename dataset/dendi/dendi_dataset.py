# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import mmcv
import mmengine
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import cv2
import warnings
import os
import random
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .loading import *
from .dendi_utils import *


def remove_out_of_range(bboxes, centers, orders=None, num_vertices=None, isEllipse=None):
    # return bboxes, centers, orders
    """Filter rotation annotations with out-of-range coordinates and adjust orders."""
    # Check validity for centers [N,]
    valid_centers = (centers >= -1e-1).all(dim=1) & (centers <
                                                     1+1e-1).all(dim=1)

    # Check validity for bboxes [N,]
    valid_bboxes = (bboxes >= -1e-1).all(dim=2).all(
        dim=1) & (bboxes < 1+1e-1).all(dim=2).all(dim=1)

    # Combined validity mask
    valid_mask = valid_centers & valid_bboxes

    # Apply filtering
    filtered_bboxes = bboxes[valid_mask]
    filtered_centers = centers[valid_mask]
    if orders is not None:
        orders = torch.tensor(orders)
        # Simply filter orders with the same mask
        filtered_orders = orders[valid_mask]
    else:
        filtered_orders = None

    if num_vertices is not None:
        num_vertices = torch.tensor(num_vertices)
        filtered_num_vertices = num_vertices[valid_mask]
    else:
        filtered_num_vertices = None

    if isEllipse is not None:
        isEllipse = torch.tensor(isEllipse)
        filtered_isEllipse = isEllipse[valid_mask]
    else:
        filtered_isEllipse = None


    return filtered_bboxes, filtered_centers, filtered_orders, filtered_num_vertices, filtered_isEllipse


class DendiDataset(Dataset):
    CLASSES = None
    PALETTE = None

    def __init__(self,
                 ann_file,
                 split,
                 input_size=(511, 511),
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 file_client_args=dict(backend='disk'),
                 sigma=0.6,
                 rot_center_sigma = 1,
                 map_size=[256, 256],
                 kernel_size=5,
                 num_anchor=8,
                 num_data=None,
                 fix_seed=False,
                 resize=False,
                 orientational_anchor=True,):
        
        self.fix_seed = fix_seed
        if self.fix_seed:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)

        self.ann_file = ann_file
        self.data_root = data_root
        self.split = split
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.file_client = mmengine.fileio.file_client.FileClient(**file_client_args)
        self.input_size = input_size
        self.im_flip = 'a'
        self.im_pad = True
        self.color_jitter = True
        self.random_rotate = True
        self.sigma = sigma
        self.rot_center_sigma = rot_center_sigma
        self.map_size = map_size
        self.kernel_size = kernel_size
        self.num_anchor = num_anchor
        self.num_data = num_data
        self.orientational_anchor = orientational_anchor
        if resize:
            self.im_pad = False
            
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)

        # load annotations (and proposals)
        # Remove data without lines
        self.data_infos = self.load_annotations(self.ann_file)

        # Remove samples without line
        for i in reversed(range(len(self.data_infos))):
            orders = [self.data_infos[i]['ann']['rot'][l]['order']
                      for l in range(len(self.data_infos[i]['ann']['rot']))]
            for j in range(len(orders) - 1, -1, -1):
                if self.data_infos[i]['ann']['rot'][j]['order'] > 8 or self.data_infos[i]['ann']['rot'][j]['order'] == 7:
                    self.data_infos[i]['ann']['rot'].pop(j)

        self.compose_pipeline()

    def compose_pipeline(self):
        self.loading = [LoadImageFromFile(channel_order='rgb'),
            LoadSymmetryAnnotations(),]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        at = {'gt_lines1': 'bboxes', 
              'gt_lines2': 'bboxes', 
              'gt_ellipses':'mask', 
              'gt_axis': 'mask', 
              'seg_mask': 'mask', 
              'seg_mask_rot': 'mask',
              'ellipse_center': 'bboxes',
              'ellipse_line': 'bboxes',
              'rot_centers': 'bboxes',
              'rot_vertices': 'bboxes',
              }
        
        if self.im_pad:
            augs = [
                A.LongestMaxSize(max_size=self.input_size[0]),  # 511, 511
                    A.PadIfNeeded(min_height=self.input_size[0], min_width=self.input_size[1], \
                            border_mode=cv2.BORDER_CONSTANT, mask_value=255),
                    ]
        else:
            augs = [
                    A.Resize(self.input_size[0], self.input_size[1]),
                    ]
        if self.im_flip in ['a']:
            flip = A.Flip()
            augs.append(flip)
        elif self.im_flip in ['h']:
            flip = A.HorizontalFlip()
            augs.append(flip)
        
        if self.color_jitter:
            augs.append(A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5))
        
        rotate = None
        if self.random_rotate in [1]:
            rotate = A.Rotate(border_mode=0, mask_value=255)
        self.rotate = rotate
        if rotate is not None:
            augs.append(rotate)

        transforms_train = A.Compose(
            augs + [
                A.Normalize(mean, std),
                ToTensorV2(),
            ],
            additional_targets=at,
        )

        if self.im_pad:
            val_aug = [ A.LongestMaxSize(max_size=self.input_size[0]),
                        A.PadIfNeeded(min_height=self.input_size[0], min_width=self.input_size[1], 
                                      border_mode=cv2.BORDER_CONSTANT, mask_value=255)]
        else:
            val_aug = [A.Resize(self.input_size[0], self.input_size[1]),]   

        transforms_val = A.Compose(
                        val_aug
                        +[A.Normalize(mean, std),
                        ToTensorV2(), 
                          ], additional_targets=at,)

        transforms_test = A.Compose([
                        A.Normalize(mean, std),
                        ToTensorV2(), 
        ], additional_targets=at,)

        self.transforms = {}
        self.transforms['train'] = transforms_train
        self.transforms['val'] = transforms_val
        self.transforms['test'] = transforms_test

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        return mmengine.load(ann_file)

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def __getitem__(self, idx):
        if self.fix_seed:
            random.seed(42 + idx)
            np.random.seed(42 + idx)
            torch.manual_seed(42 + idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42 + idx)
        
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        data = self.do_pipeline(results)
        return data

    def do_pipeline(self, x):
        for t in self.loading:
            x = t(x)
        img, gt_lines1, gt_lines2 = x['img'], x['gt_lines1'], x['gt_lines2']
        rot_centers, rot_vertices, rot_orders, num_vertices, isEllipse \
            = x['rot_centers'], x['rot_vertices'], x['rot_orders'], x['num_vertices'], x['isEllipse']
        h, w = img.shape[0], img.shape[1]

        if 'ellipse_center' in x.keys():
            ellipse_center = x['ellipse_center']
            ellipse_center = ellipse_center / torch.tensor([[w, h, w, h]])
        else:
            ellipse_center = torch.zeros(0, 2)  # Use shape [0, 2]

        if 'ellipse_line' in x.keys():
            ellipse_line = x['ellipse_line']
            ellipse_line = ellipse_line / torch.tensor([[w, h, w, h]])
        else:
            ellipse_line = torch.zeros(0, 4)  # Use shape [0, 4]

        if 'rot_centers' in x.keys():
            rot_centers = x['rot_centers']
            rot_centers = rot_centers / torch.tensor([[w, h, w, h]])
        else:
            rot_centers = torch.zeros(0, 2)  # Use shape [0, 2]


        if 'rot_vertices' in x.keys():
            rot_vertices = x['rot_vertices']
            if 0 not in rot_vertices.shape:
                rot_vertices = rot_vertices / torch.tensor([[w, h, w, h]])
            else:
                rot_vertices = torch.zeros(len(rot_orders), 0, 4)
        else:
            rot_vertices = torch.zeros(0, 4)  # Use shape [0, 4]

        seg_mask = x['gt_reflective_seg']
        seg_mask_rot = x['gt_rotational_seg']
        # filename = x['filename'].split('/')[-1].split('.')[0]
        filename = x['filename']
        original_shape = img.shape

        # ellipse train
        # Mask o -> 1
        # Mask x -> 0
        if self.split == 'train':
            mask_path = os.path.join(self.data_root, 'masks', f"{filename}.png")
            if os.path.exists(mask_path):
                gt_ellipses = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                gt_ellipses = (gt_ellipses == 255).astype(np.float32) * 100 # 100 으로 표시
            else:
                gt_ellipses = np.zeros_like(seg_mask, dtype=np.float32)
        # ellipse val
        else:
            gt_ellipses = (x['gt_ellipses'] == 255).astype(np.float32) * 100

        gt_axis = x['gt_axis']
        gt_axis = ((gt_axis > 0) * 100).astype(np.float32)

        gt_lines1 = gt_lines1 / torch.tensor([[w, h, w, h]])
        gt_lines2 = gt_lines2 / torch.tensor([[w, h, w, h]]) 

        num_objects = rot_centers.shape[0]

        t = self.transforms[self.split](image=img, 
                                        seg_mask=seg_mask,
                                        seg_mask_rot=seg_mask_rot,
                                        gt_lines1=gt_lines1,
                                        gt_lines2=gt_lines2,
                                        gt_ellipses=gt_ellipses,
                                        gt_axis=gt_axis,
                                        ellipse_center=torch.clamp(
                                            ellipse_center, min=1e-3, max=1-1e-3),
                                        ellipse_line=torch.clamp(
                                            ellipse_line, min=1e-3, max=1-1e-3),
                                        rot_centers=torch.clamp(
                                            rot_centers, min=1e-3, max=1-1e-3),
                                        rot_vertices=torch.clamp(
                                            rot_vertices, min=1e-3, max=1-1e-3).reshape(-1, 4),)

        # Process the rest of the data
        data = defaultdict(dict)

        if gt_lines1.shape[0] > 0:
            data['img'] = t['image']
            data['seg_mask'] = t['seg_mask']
            data['seg_mask_rot'] = t['seg_mask_rot']
            h, w = data['img'].shape[-2], data['img'].shape[-1]
            gt_lines1, gt_lines2 = t['gt_lines1'], t['gt_lines2']


            gt_lines = []
            for a, b in zip(gt_lines1, gt_lines2):
                line = a[0], a[1], b[0], b[1]
                new_line = calibrate_lines(line)
                if new_line is not None:
                    a, b, c, d = new_line
                    new_line = a.item(), b.item(), c.item(), d.item()
                    gt_lines.append(new_line)
            data['gt_lines'] = torch.tensor(gt_lines)

        else:
            data['img'] = t['image']
            data['seg_mask'] = t['seg_mask']
            data['seg_mask_rot'] = t['seg_mask_rot']
            data['gt_lines'] = torch.empty((0, 4))


        if len(ellipse_center) != 0:
            ellipse_center_new = t['ellipse_center']
            ellipse_center_new = torch.tensor(ellipse_center_new)[:,:2]
        else:
            ellipse_center_new = ellipse_center[:, :2]

        if len(ellipse_line) != 0:
            ellipse_line_new = t['ellipse_line']
            ellipse_line_new = torch.tensor(
                ellipse_line_new).reshape(-1, 5, 4)[:, :, :2]
        else:
            ellipse_line_new = ellipse_line[:, :2].reshape(0, 0, 2)

        ellipse_line_new, ellipse_center_new, _, _, _ \
            = remove_out_of_range(ellipse_line_new, ellipse_center_new)

        if len(rot_centers) != 0:
            rot_centers_new = t['rot_centers']
            rot_centers_new = torch.tensor(rot_centers_new)[:, :2]
        else:
            rot_centers_new = rot_centers[:, :2]

        if len(rot_vertices) != 0:
            rot_vertices_new = t['rot_vertices']
            rot_vertices_new = torch.tensor(rot_vertices_new).reshape(
                num_objects, -1, 4)[:, :, :2]
        else:
            rot_vertices_new = rot_vertices[:, :, :2].reshape(0, 0, 2)

        rot_vertices_new, rot_centers_new, rot_orders_new, num_vertices_new, isEllipse_new \
            = remove_out_of_range(rot_vertices_new, rot_centers_new, rot_orders, num_vertices, isEllipse)

        data['ellipse_center'] = ellipse_center_new
        data['ellipse_line'] = ellipse_line_new
        data['gt_ellipses'] = (t['gt_ellipses'] == 100).float()
        data['gt_axis'] = t['gt_axis']
        data['original_shape'] = original_shape
        data['filename'] = filename
        data['rot_orders'] = rot_orders_new
        data['rot_centers'] = rot_centers_new
        data['rot_vertices'] = rot_vertices_new
        data['num_vertices'] = num_vertices_new
        data['isEllipse'] = isEllipse_new
        return data

    def collate_fn(self, batch):
        # random.seed(42)
        img = list()
        seg_mask = list()
        seg_mask_rot = list()
        gt_lines = list()
        filename = list()
        gt_ellipses = list()
        ellipse_center = list()
        ellipse_line = list()
        rot_centers = list()
        rot_vertices = list()
        isEllipse = list()

        for idx, b in enumerate(batch):
            img.append(b['img'])
            seg_mask.append(b['seg_mask'])
            seg_mask_rot.append(b['seg_mask_rot'])
            gt_lines.append(b['gt_lines'])  # (1, h, w)
            filename.append(b['filename'])
            gt_ellipses.append(b['gt_ellipses'].unsqueeze(0))
            ellipse_center.append(b['ellipse_center'].unsqueeze(0))
            ellipse_line.append(b['ellipse_line'].unsqueeze(0))
            rot_centers.append(b['rot_centers'].unsqueeze(0))
            rot_vertices.append(b['rot_vertices'].unsqueeze(0))
            isEllipse.append(b['isEllipse'].unsqueeze(0))
            

        img = torch.stack(img, dim=0)
        gt_ellipses = torch.stack(gt_ellipses, dim=0)
        ellipse_center = torch.stack(ellipse_center, dim=0)
        ellipse_line = torch.stack(ellipse_line, dim=0)
        rot_centers = torch.stack(rot_centers, dim=0)
        rot_vertices = torch.stack(rot_vertices, dim=0)
        isEllipse = torch.stack(isEllipse, dim=0)

        data = {}
        data['img'] = img
        data['seg_mask'] = seg_mask     
        data['seg_mask_rot'] = seg_mask_rot
        data['gt_lines'] = gt_lines
        data['filename'] = filename
        data['gt_ellipses'] = gt_ellipses
        data['ellipse_center'] = ellipse_center
        data['ellipse_line'] = ellipse_line
        data['rot_centers'] = rot_centers
        data['rot_vertices'] = rot_vertices
        data['isEllipse'] = isEllipse

        return data

class AxisSymDataset(DendiDataset):

    def __init__(self, *args, **kwargs):
        super(AxisSymDataset, self).__init__(*args, **kwargs)
        if self.fix_seed:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
        self.max_point = 700

    def reorder_lines(self, lines):
        for i in range(len(lines)):
            if (lines[i][2] < lines[i][0]):
                lines[i] = lines[i][[2, 3, 0, 1]]
            elif (lines[i][0] == lines[i][2]) and (lines[i][1] < lines[i][3]):
                lines[i] = lines[i][[2, 3, 0, 1]]

        return lines
    
    def adjust_line(self, lines):
        if len(lines) == 0:
            return lines
        else:
            x1 = lines[:, 0].unsqueeze(1)
            y1 = lines[:, 1].unsqueeze(1)
            x2 = lines[:, 2].unsqueeze(1)
            y2 = lines[:, 3].unsqueeze(1)

            x_max = (x2 - x1) / (y2 - y1 + 1e-6) * (1 - y1) + x1
            x_min = (x2 - x1) / (y2 - y1 + 1e-6) * (0 - y1) + x1
            y_max = (y2 - y1) / (x2 - x1 + 1e-6) * (1 - x1) + y1
            y_min = (y2 - y1) / (x2 - x1 + 1e-6) * (0 - x1) # x 0일때 y

            x1_max = x1 > 1
            x1_min = x1 < 0
            y1_max = y1 > 1
            y1_min = y1 < 0
            x2_max = x2 > 1
            x2_min = x2 < 0
            y2_max = y2 > 1
            y2_min = y2 < 0

            x1[x1_max] = 1.
            y1[x1_max] = y_max[x1_max]
            x1[x1_min] = 0.
            y1[x1_min] = y_min[x1_min]

            y1[y1_max] = 1.
            x1[y1_max] = x_max[y1_max]
            y1[y1_min] = 0.
            x1[y1_min] = x_min[y1_min]

            x2[x2_max] = 1.
            y2[x2_max] = y_max[x2_max]
            x2[x2_min] = 0.
            y2[x2_min] = y_min[x2_min]

            y2[y2_max] = 1.
            x2[y2_max] = x_max[y2_max]
            y2[y2_min] = 0.
            x2[y2_min] = x_min[y2_min]

            line = torch.cat([x1, y1, x2, y2], dim=1)
            # 같은 방향으로 두개가 빠지면 안됨
            filter = (((x1_max * x2_max) != 1) * ((x1_min * x2_min) != 1) * ((y1_min * y2_min) != 1) * ((y1_max * y2_max) != 1)).squeeze()
            # 인접한 방향으로 빠지는거 생각

            line = line[filter != 0]
            if len(line.shape) == 3:
                line = line.squeeze(0)
            return line

    def get_confidence_map(self, midpoints, height, width, sigma, rho_mask, theta_mask):
        confidence_map = torch.zeros(self.num_anchor, height, width)
        center_pos_idx = torch.round(midpoints).to(torch.int32)
        center_pos_idx = torch.clamp(center_pos_idx, 0 ,height-1)

        filter_range = torch.linspace(0, self.kernel_size - 1, self.kernel_size) - (self.kernel_size // 2)
        grid_x, grid_y = torch.meshgrid(filter_range, filter_range)
        mask = torch.cat([grid_x.flatten().unsqueeze(0), grid_y.flatten().unsqueeze(0)], dim=0).T.to(torch.int32)

        if len(center_pos_idx) != 0:
            for anchor in range(self.num_anchor):
                center_pos_idx_anchor = center_pos_idx[theta_mask[anchor]]
                for mid in center_pos_idx_anchor:
                    idxs = mid + mask

                    # idxs = torch.clamp(idxs, min=0, max=height-1)
                    idxs[:, 0] = torch.clamp(idxs[:, 0], min=0, max=width-1)
                    idxs[:, 1] = torch.clamp(idxs[:, 1], min=0, max=height-1)
                    confidence_map[anchor][idxs[:,1], idxs[:,0]] = torch.maximum(torch.exp(-(mask[:,0] ** 2 + mask[:,1] ** 2) / (2 * sigma ** 2)), 
                            confidence_map[anchor][idxs[:,1], idxs[:,0]])
                                                                                                                
        confidence_points = torch.zeros_like(center_pos_idx)
        center_pos_idx = torch.clamp(center_pos_idx, 0, height-1)

        return confidence_map, center_pos_idx, confidence_points

    def get_rot_center_map(self, centers, height, width, sigma):
        centers = torch.tensor(centers) * torch.tensor([width, height])
        rot_center_map = torch.zeros(1, height, width)
        rot_center_pos_idx = torch.round(centers).to(torch.int32) # (n, 2)
        rot_center_pos_idx = torch.clamp(rot_center_pos_idx, 0, height-1) # (n, 2)

        filter_range = torch.linspace(0, self.kernel_size - 1, self.kernel_size) - (self.kernel_size // 2)
        grid_x, grid_y = torch.meshgrid(filter_range, filter_range)
        mask = torch.cat([grid_x.flatten().unsqueeze(0), grid_y.flatten().unsqueeze(0)], dim=0).T.to(torch.int32)   

        if len(rot_center_pos_idx) != 0:
            for rot_center in rot_center_pos_idx:
                idxs = rot_center + mask
                idxs = torch.clamp(idxs, min=0, max=height-1)
                values = torch.exp(-(mask[:,0] ** 2 + mask[:,1] ** 2) / (2 * sigma ** 2))
                indices = (idxs[:, 1], idxs[:, 0])
                rot_center_map[0][indices] = torch.maximum(rot_center_map[0][indices], values)

        return rot_center_map, rot_center_pos_idx

    def get_rot_scale_and_fold_map(self, centers, vertices, orders, height, width):
        """Generate rotation scale and fold maps for symmetric objects."""
        device = centers.device if torch.is_tensor(centers) else 'cpu'
        scale_map = torch.zeros(height, width, device=device)
        fold_map = - torch.ones(height, width, device=device)
        DEFAULT_SCALE = 5

        if len(centers) == 0 or len(orders) == 0:
            return scale_map, fold_map

        centers = torch.as_tensor(centers, device=device).float(
        ).view(-1, 2) * torch.tensor([width, height])
        vertices = torch.as_tensor(vertices, device=device).float().view(
            len(centers), -1, 2) * torch.tensor([width, height])
        orders = torch.as_tensor(orders, device=device).float().view(-1)

        for idx in range(len(centers)):
            center = centers[idx]
            verts = vertices[idx]
            order = orders[idx]

            # Handle all objects (including order=0) using vertex data
            valid_mask = torch.norm(verts, dim=1) > 1e-4
            valid_verts = verts[valid_mask]

            if len(valid_verts) > 0:
                distances = torch.norm(valid_verts - center, dim=1)
                max_distance = torch.max(distances)
                radius = max_distance.item()
                kernel_size = 2 * int(radius) + 1
                k = (kernel_size - 1) // 2
            else:
                k = DEFAULT_SCALE  # Fallback for empty vertices

            x, y = center[0].item(), center[1].item()
            x_idx = min(max(round(x), 0), width-1)
            y_idx = min(max(round(y), 0), height-1)

            if 0 <= x_idx < width and 0 <= y_idx < height:
                scale_map[y_idx, x_idx] = k
                fold_map[y_idx, x_idx] = order

        return scale_map, fold_map

    def get_one_hot_fold_map(self, fold_map, height, width):
        # ----------------------------
        # Step 1. One-hot + Gaussian Smoothing
        # ----------------------------
        fold_to_channel = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6}
        num_channels = len(fold_to_channel)  # 7 channels

        # Build one-hot map: shape (7, height, width)
        fold_map_onehot = torch.zeros(num_channels, height, width,
                                      device=fold_map.device, dtype=torch.float32)
        for fold_val, channel in fold_to_channel.items():
            mask = (fold_map == fold_val)
            fold_map_onehot[channel][mask] = 1.0
        # Positions with fold_map == -1 remain zeros.

        # Create Gaussian kernel.
        kernel_size = self.kernel_size
        sigma = self.sigma
        ax = torch.arange(kernel_size, dtype=torch.float32,
                          device=fold_map.device) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        # Prepare kernel for group convolution: shape (7, 1, k, k)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(num_channels, 1, 1, 1)

        onehot_unsqueezed = fold_map_onehot.unsqueeze(
            0)  # add batch dim: (1, 7, H, W)
        fold_map_onehot_gs = F.conv2d(onehot_unsqueezed, kernel,
                                      padding=kernel_size // 2, groups=num_channels)
        fold_map_onehot_gs = fold_map_onehot_gs.squeeze(0)  # (7, H, W)


        fold_map_sum = fold_map_onehot.sum(dim=0).unsqueeze(0)
        padding_map = 1 - fold_map_sum
        output_map = torch.cat([fold_map_onehot, padding_map], dim=0)

        return output_map


        # # ----------------------------
        # # Step 2. Normalization per pixel
        # # ----------------------------
        # # We'll produce an output of shape (8, H, W) where the extra channel (index 7)
        # # holds the "no fold" value for pixels where the 7 channels sum < 1.
        # normalized = torch.zeros(num_channels + 1, height,
        #                          width, device=fold_map.device, dtype=torch.float32)

        # # Compute per-pixel sum of the 7 fold channels: shape (H, W)
        # fiber_sum = fold_map_onehot_gs.sum(dim=0)

        # # Create a mask for pixels where any channel is exactly 1 (locked pixels)
        # # Expand fold_map_onehot_gs to check per pixel.
        # locked_mask = (fold_map_onehot_gs == 1.0).any(dim=0)  # shape (H, W)

        # # --- Group A: Locked pixels ---
        # # For locked pixels, we need to keep the channel that is 1 and zero-out all others.
        # # We determine the first occurrence along the channel dimension.
        # # (If multiple channels are exactly 1, we take the first one.)
        # if locked_mask.any():
        #     # Get indices for locked pixels
        #     # We'll process all pixels at once.
        #     # Create a full index grid.
        #     H, W = height, width
        #     grid_h = torch.arange(
        #         H, device=fold_map.device).view(H, 1).expand(H, W)
        #     grid_w = torch.arange(
        #         W, device=fold_map.device).view(1, W).expand(H, W)
        #     # For the 7 channels, create a mask of shape (7, H, W) for exact ones.
        #     locked_channel_mask = (fold_map_onehot_gs == 1.0)
        #     # For each pixel, find the first channel index where locked_channel_mask is True.
        #     # We'll set these pixels accordingly.
        #     # To avoid loops, we use masked_select and scatter.
        #     # First, create a tensor for the locked channel index for each pixel, default -1.
        #     locked_channel = torch.full(
        #         (H, W), -1, device=fold_map.device, dtype=torch.long)
        #     for ch in range(num_channels):
        #         # Only update those pixels that are not yet locked.
        #         not_locked = (locked_channel == -1)
        #         update = locked_channel_mask[ch] & not_locked
        #         if update.any():
        #             locked_channel[update] = ch
        #     # Now, for pixels that are locked, set that channel to 1.
        #     # We can scatter these values into the output.
        #     # First, set all 7 channels at locked pixels to 0.
        #     normalized[:num_channels] *= 0.0  # already 0, just for clarity.
        #     # Then, using advanced indexing, assign 1 to the locked channel.
        #     normalized[locked_channel, grid_h, grid_w] = torch.where(
        #         locked_mask, 1.0, 0.0)
        #     # The extra "no fold" channel remains 0.

        # # --- Group B: Pixels with fiber_sum < 1 ---
        # under_mask = (fiber_sum < 1.0) & (~locked_mask)
        # if under_mask.any():
        #     # For these pixels, keep the 7-channel fiber unchanged and set extra channel to (1 - sum).
        #     normalized[:num_channels,
        #                under_mask] = fold_map_onehot_gs[:, under_mask]
        #     normalized[num_channels, under_mask] = 1.0 - fiber_sum[under_mask]

        # # --- Group C: Pixels with fiber_sum > 1 (and not locked) ---
        # over_mask = (fiber_sum > 1.0) & (~locked_mask)
        # if over_mask.any():
        #     # We'll process only the over-filled pixels in a vectorized manner.
        #     # Let X be the 7-channel fiber for these pixels.
        #     # X has shape (7, N) where N is the number of over-filled pixels.
        #     X = fold_map_onehot_gs[:, over_mask]  # shape: (7, N)
        #     S = X.sum(dim=0)  # shape: (N,)
        #     # amount to subtract from each fiber, shape: (N,)
        #     diff = S - 1.0

        #     # Sort each fiber in ascending order.
        #     # sorted_X: shape (7, N); indices: shape (7, N)
        #     sorted_X, sort_indices = torch.sort(X, dim=0, descending=False)
        #     # Compute cumulative sum along channel dimension.
        #     cumsum_X = sorted_X.cumsum(dim=0)  # shape (7, N)

        #     # For each pixel, determine the number r (0-indexed) such that
        #     # cumsum_X[r] is the first value that is NOT strictly less than diff.
        #     # That is, r = (# of entries with cumsum < diff).
        #     # We can compute a mask and then sum along the channel dimension.
        #     # Make diff broadcastable: shape (1, N)
        #     diff_exp = diff.unsqueeze(0)
        #     less_mask = cumsum_X < diff_exp  # shape (7, N); boolean
        #     r = less_mask.sum(dim=0)  # shape (N,); r is an integer in [0,7]
        #     # r tells us that indices 0..r-1 will be zeroed out.
        #     # For the r-th entry, we subtract (diff - (cumsum at r-1, or 0 if r==0))
        #     # and for entries r+1 and beyond we keep the original values.

        #     # Gather the cumulative sum at index r-1 for each pixel.
        #     # For pixels where r==0, use 0.
        #     HN = X.shape[1]  # number of over-filled pixels
        #     device = X.device
        #     # Create a tensor for r-1, but clip at 0.
        #     r_minus = torch.clamp(r - 1, min=0)
        #     # Prepare an index tensor for gather.
        #     # r_minus has shape (N,). We need it to have shape (1, N) to gather along dim=0.
        #     r_minus = r_minus.unsqueeze(0)  # shape (1, N)
        #     cumsum_prev = torch.gather(
        #         cumsum_X, 0, r_minus).squeeze(0)  # shape (N,)
        #     # For pixels with r == 0, we want cumsum_prev to be 0.
        #     cumsum_prev = torch.where(
        #         r == 0, torch.zeros_like(cumsum_prev), cumsum_prev)

        #     # Now compute the new value for the channel at position r for each pixel.
        #     # For each pixel n, new_val = sorted_X[r, n] - (diff[n] - cumsum_prev[n])
        #     # We need to gather sorted_X at index r for each pixel.
        #     # For this, build an index tensor for each pixel.
        #     r_unsqueezed = r.unsqueeze(0)  # shape (1, N)
        #     new_val = torch.gather(sorted_X, 0, r_unsqueezed).squeeze(
        #         0) - (diff - cumsum_prev)
        #     # Ensure new_val is not negative (it should not be by construction).
        #     new_val = torch.clamp(new_val, min=0.0)

        #     # Build new_sorted_X:
        #     # For each over-filled pixel (each column):
        #     # - For indices < r, set to 0.
        #     # - For index == r, set to new_val.
        #     # - For indices > r, leave unchanged.
        #     idx = torch.arange(num_channels, device=device).unsqueeze(
        #         1)  # shape (7,1)
        #     r_expanded = r.unsqueeze(0).expand(
        #         num_channels, HN)  # shape (7, N)
        #     new_sorted_X = torch.where(
        #         idx < r_expanded, torch.zeros_like(sorted_X), sorted_X)
        #     new_sorted_X = torch.where(
        #         idx == r_expanded, new_val.unsqueeze(0), new_sorted_X)
        #     # At this point, new_sorted_X has the adjusted values and its columns sum to 1.

        #     # Now, we need to revert from sorted order back to the original order.
        #     # We'll build an empty tensor for the adjusted fiber.
        #     X_adjusted = torch.empty_like(new_sorted_X)
        #     # We know sort_indices tells us where each sorted element came from.
        #     # For each column, scatter new_sorted_X back.
        #     X_adjusted = X_adjusted.scatter(
        #         dim=0, index=sort_indices, src=new_sorted_X)

        #     # Now assign these adjusted fibers to the normalized tensor for the over-filled pixels.
        #     normalized[:num_channels, over_mask] = X_adjusted
        #     # extra channel remains 0.
        #     normalized[num_channels, over_mask] = 0.0

        # return normalized

    def get_geometric_map(self, gt_lines, midpoints, height, width):

        rho_map = torch.zeros(self.num_anchor, height, width)
        theta_map = torch.zeros(self.num_anchor, height, width)

        if len(gt_lines) != 0:
            rho = (torch.sum((gt_lines[:,:2] - gt_lines[:,2:])**2, dim=1) ** 0.5) # [n, ]
            interval = torch.pi / self.num_anchor
            mask = []
            one_side_interval = interval/2

            # 계산 편하게 하기 위해 0 ~ 180 으로 map
            theta = torch.asin((gt_lines[:, 3] - gt_lines[:, 1]) / (rho + 1e-4)) + torch.pi /2 # [n, ]
            for i in range(self.num_anchor):
                if i == 0:
                    mask.append((theta >= (torch.pi - one_side_interval)) * (theta < torch.pi) + \
                                (theta >= 0) * (theta < one_side_interval))
                else:
                    mask.append((theta >= (interval*i - one_side_interval)) * (theta < (interval*i + one_side_interval)))
                # mask.append((theta >= (interval*i)) * (theta <= (interval*(i+1))))
            mask = torch.stack(mask) # [num_anchor, n]

            for i in range(self.num_anchor):
                rho_map[i][midpoints[mask[i]][:, 1], midpoints[mask[i]][:, 0]] = rho[mask[i]]

                if i == 0:
                    theta[mask[i]] = (theta[mask[i]] - torch.pi) * (theta[mask[i]] >= interval * (self.num_anchor) - one_side_interval) + \
                        theta[mask[i]] * (theta[mask[i]] < one_side_interval) 

                theta_map[i][midpoints[mask[i]][:, 1], midpoints[mask[i]][:, 0]] = theta[mask[i]] - (i*interval)
        
        else:
            mask = torch.zeros([self.num_anchor, 0])
            rho_map = torch.zeros(self.num_anchor, height, width)
            theta_map = torch.zeros(self.num_anchor, height, width)
        
        geometric_map = torch.cat([rho_map, theta_map]) 

        return geometric_map, mask

    def additional_pipeline(self, data):
        img_h = data['img'].shape[1]
        img_w = data['img'].shape[2]
        
        # Spatial resolution of the output map
        if self.split != 'test':
            map_h = self.map_size[0]
            map_w = self.map_size[1]
        else:
            map_h = img_h
            map_w = img_w

        # Center, Line position
        
        gt_lines = data['gt_lines'] # (n, 4)
        gt_lines = self.adjust_line(gt_lines)
        gt_lines = self.reorder_lines(gt_lines)
        data['gt_lines'] = gt_lines
        if len(data['gt_lines']) == 0:
            center_pos = torch.zeros(0, 2)
            line_pos = torch.zeros(0, 4)
            center_pos_idx = torch.zeros(0, 2)
        else:
            center_pos = ((gt_lines[:,:2] + gt_lines[:,2:]) / 2) * torch.tensor([map_w, map_h]) # (#center, 2), float
            line_pos = torch.clamp(gt_lines * torch.tensor([map_w, map_h, map_w, map_h]), min = 0, max = max(map_h, map_w)-1)
            center_pos_idx = torch.round(center_pos).to(torch.int32) # (#center, 2), float
            center_pos_idx = torch.clamp(center_pos_idx, 0, map_h-1) # (#center, 2), float

        # mask : [num_anchor, n]
        geometric_map, theta_mask = self.get_geometric_map(line_pos, center_pos_idx, map_h, map_w) # (2*num_anchor, h, w)

        # Midpoint map
        midpoint_confidence_map, midpoints, confidence_points = \
              self.get_confidence_map(center_pos, map_h, map_w, sigma=self.sigma, 
                                      rho_mask=(geometric_map[:self.num_anchor]>0), theta_mask=theta_mask)

        # For visualization and metric
        midpoints_map = torch.zeros(1, map_h, map_w)
        midpoints_map[:, midpoints[:,1], midpoints[:,0]] = 1.

        midpoints_map_stack = torch.zeros(self.num_anchor, map_h, map_w)
        if len(center_pos_idx) != 0:
            for i in range(self.num_anchor):
                if len(center_pos_idx[theta_mask[i]]) != 0:
                    midpoints_map_stack[i][center_pos_idx[theta_mask[i]][:, 1], center_pos_idx[theta_mask[i]][:, 0]] = 1    

        rot_center_map, rot_center_pos_idx = self.get_rot_center_map(
            data['rot_centers'], map_h, map_w, sigma=self.sigma)
        
        scale_map, fold_map = self.get_rot_scale_and_fold_map(data['rot_centers'], data['rot_vertices'],
                                                              data['rot_orders'], map_h, map_w)
        fold_map_onehot = self.get_one_hot_fold_map(fold_map, map_h, map_w)

        gt_ellipses = data['gt_ellipses'][None, None, :, :]  # [1, 1, H, W]
        gt_ellipses_resized = F.interpolate(gt_ellipses, size=(map_h, map_w), mode='nearest')
        data['gt_ellipses'] = gt_ellipses_resized.squeeze(1)  # [map_h, map_w]

        if not self.orientational_anchor:
            midpoint_confidence_map = torch.max(midpoint_confidence_map, dim=0, keepdim=True)[0]  # [1, H, W]
            # Split geometric map into two halves and pool each half separately
            geometric_map_split = geometric_map.reshape(2, geometric_map.shape[0]//2, *geometric_map.shape[-2:])
            geometric_map = torch.max(geometric_map_split, dim=1)[0]  # [2, H, W]
            midpoints_map_stack = torch.max(midpoints_map_stack, dim=0, keepdim=True)[0]  # [1, H, W]
        
        data['n_lines'] = len(data['gt_lines'])
        data['gt_lines'] = torch.vstack((data['gt_lines'], torch.zeros(self.max_point - len(data['gt_lines']), 4))) \
            if len(data['gt_lines']) != 0 else torch.zeros(self.max_point, 4)
        data['midpoint_confidence_map'] = midpoint_confidence_map 
        data['rot_center_map'] = rot_center_map
        data['rot_scale_map'] = scale_map
        data['rot_fold_map'] = fold_map
        data['rot_fold_map_onehot'] = fold_map_onehot
        data['geometric_map'] = geometric_map
        data['midpoints'] = torch.vstack((midpoints, torch.zeros(self.max_point - len(midpoints), 2))) # (n, 2), for gt 
        data['confidence_points'] = torch.vstack((confidence_points, torch.zeros(self.max_point * 4 - len(confidence_points), 2))) 
        data['midpoint_map'] = midpoints_map # for visualize and test metric
        data['midpoint_map_stack'] = midpoints_map_stack

        mask = data['rot_orders'] == 0
        data['rot_vertices'][mask] = 0.0


        return data
    
    def collate_fn(self, batch):
        # random.seed(42)
        img = list()
        gt_lines = list()
        gt_ellipses = list()
        midpoint_confidence_map = list()
        geometric_map = list()
        midpoints = list()
        confidence_points = list()
        midpoint_map = list()
        midpoint_map_stack = list()
        n_lines = list()
        original_shape = list()
        filename = list()

        rot_center_map = list()
        rot_scale_map = list()
        rot_fold_map = list()
        rot_fold_map_onehot = list()
        ellipse_center = list()
        ellipse_line = list()
        rot_centers = list()
        rot_vertices = list()

        for idx, b in enumerate(batch):
            img.append(b['img'])
            gt_lines.append(b['gt_lines']) # (1, h, w)
            gt_ellipses.append(b['gt_ellipses'])
            midpoint_confidence_map.append(b['midpoint_confidence_map'])
            geometric_map.append(b['geometric_map'])
            midpoints.append(b['midpoints'])
            confidence_points.append(b['confidence_points'])
            midpoint_map.append(b['midpoint_map'])
            midpoint_map_stack.append(b['midpoint_map_stack'])
            rot_center_map.append(b['rot_center_map'])
            rot_scale_map.append(b['rot_scale_map'])
            rot_fold_map.append(b['rot_fold_map'])
            rot_fold_map_onehot.append(b['rot_fold_map_onehot'])
            n_lines.append(b['n_lines'])
            original_shape.append(b['original_shape'])
            filename.append(b['filename'])

        img = torch.stack(img, dim=0)
        gt_ellipses = torch.stack(gt_ellipses, dim=0)
        gt_lines = torch.stack(gt_lines, dim=0)
        midpoint_confidence_map = torch.stack(midpoint_confidence_map, dim=0)
        geometric_map = torch.stack(geometric_map, dim=0)
        midpoints = torch.stack(midpoints, dim=0)
        confidence_points = torch.stack(confidence_points, dim=0)
        midpoint_map = torch.stack(midpoint_map, dim=0)
        midpoint_map_stack = torch.stack(midpoint_map_stack, dim=0)
        rot_center_map = torch.stack(rot_center_map, dim=0)
        rot_scale_map = torch.stack(rot_scale_map, dim=0)
        rot_fold_map = torch.stack(rot_fold_map, dim=0)
        rot_fold_map_onehot = torch.stack(rot_fold_map_onehot, dim=0)
        n_lines = torch.tensor(n_lines)
        original_shape = torch.tensor(original_shape)

        ellipse_center = [b['ellipse_center'] for b in batch]
        ellipse_line = [b['ellipse_line'] for b in batch]

        # Pad sequences
        ellipse_center = pad_sequence(ellipse_center, batch_first=True, padding_value=0)
        data = {}
        data['img'] = img
        data['gt_lines'] = gt_lines
        data['filename'] = filename
        data['gt_ellipses'] = gt_ellipses
        data['ellipse_center'] = ellipse_center
        data['ellipse_line'] = ellipse_line
        data['midpoint_confidence_map'] = midpoint_confidence_map
        data['geometric_map'] = geometric_map
        data['midpoints'] = midpoints
        data['confidence_points'] = confidence_points
        data['midpoint_map'] = midpoint_map
        data['midpoint_map_stack'] = midpoint_map_stack
        data['n_lines'] = n_lines
        data['rot_center_map'] = rot_center_map
        data['original_shape'] = original_shape
        data['rot_fold_map_onehot'] = rot_fold_map_onehot
        rot_centers = []
        rot_orders = []
        rot_vertices = []
        num_vertices = []
        
        
        for b in batch:
            # Convert to tensor if not already
            centers = torch.tensor(b['rot_centers']) if len(b['rot_centers']) > 0 else torch.zeros(0, 2)
            orders = torch.tensor(b['rot_orders']) if len(b['rot_orders']) > 0 else torch.zeros(0)
            vertices = torch.tensor(b['rot_vertices']) if len(b['rot_vertices']) > 0 else torch.zeros(0, 1, 2)


            rot_centers.append(centers)
            rot_orders.append(orders)
            rot_vertices.append(vertices)
            if sum(torch.tensor(vertices.shape) == 0) == 0:
                num_verts = (vertices != 0).sum(dim=2).sum(dim=1) // 2
            else:
                num_verts = torch.zeros(0, dtype=torch.long)
            num_vertices.append(num_verts)


        # Store as tuples instead of padded tensors
        data['rot_centers'] = tuple(rot_centers)
        data['rot_orders'] = tuple(rot_orders)
        data['rot_vertices'] = tuple(rot_vertices)
        data['num_vertices'] = tuple(num_vertices)
        return data
    
    def __len__(self):
        """Total number of samples of data."""
        if self.num_data:
            return self.num_data
        else:
            return len(self.data_infos)
    
    def __getitem__(self, idx):
        if self.fix_seed:
            random.seed(42 + idx)
            np.random.seed(42 + idx)
            torch.manual_seed(42 + idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42 + idx)
        
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)

        data = self.do_pipeline(results)
        data = self.additional_pipeline(data)
        return data