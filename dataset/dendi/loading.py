import os.path as osp
import mmcv
import mmengine
import numpy as np

### loading
class LoadSymmetryAnnotations:
    def __init__(self,
                 with_lines=True,
                 with_ellipses=True,
                 with_seg=True,
                 with_seg_rot=True,
                 with_axis=True,
                 with_rotation=True,
                 file_client_args=dict(backend='disk')):
        self.with_lines = with_lines
        self.with_ellipses = with_ellipses
        self.with_seg = with_seg
        self.with_seg_rot = with_seg_rot
        self.with_axis = with_axis
        self.with_rotation = with_rotation
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_axis(self, results):
        filename_axis = results['img_info'].get('filename_axis', '')
        if filename_axis:
            filename = osp.join(results['img_prefix'], filename_axis)
            if osp.exists(filename):
                if self.file_client is None:
                    self.file_client = mmengine.fileio.file_client.FileClient(**self.file_client_args)
                img_bytes = self.file_client.get(filename)
                results['gt_axis'] = mmcv.imfrombytes(img_bytes, flag='unchanged').squeeze()
                results['seg_fields'].append('gt_axis')
            else:
                # Assign default empty mask
                img_shape = results['img_shape'][:2]
                results['gt_axis'] = np.zeros(img_shape, dtype=np.uint8)
        else:
            # Assign default empty mask
            img_shape = results['img_shape'][:2]
            results['gt_axis'] = np.zeros(img_shape, dtype=np.uint8)
        return results

    def _load_lines(self, results):
        ann_info = results['ann_info']
        gt_lines_list = ann_info.get('line', [])
        if len(gt_lines_list) > 0:
            gt_lines = np.asarray(gt_lines_list.copy())
            if len(gt_lines.shape) == 2:
                results['gt_lines1'] = gt_lines[:, [0, 1, 0, 1]]
                results['gt_lines2'] = gt_lines[:, [2, 3, 2, 3]]
            else:
                results['gt_lines1'] = gt_lines
                results['gt_lines2'] = gt_lines
        else:
            # Assign empty arrays
            results['gt_lines1'] = np.zeros((0, 4), dtype=np.float32)
            results['gt_lines2'] = np.zeros((0, 4), dtype=np.float32)
        return results

    def _load_ellipses(self, results):
        filename_ellipse = results['img_info'].get('filename_ellipse', '')
        if filename_ellipse:
            filename = osp.join(results['img_prefix'], filename_ellipse)
            if osp.exists(filename):
                if self.file_client is None:
                    self.file_client = mmcv.FileClient(**self.file_client_args)
                img_bytes = self.file_client.get(filename)
                results['gt_ellipses'] = mmcv.imfrombytes(img_bytes, flag='unchanged').squeeze()
                results['seg_fields'].append('gt_ellipses')
            else:
                # Assign default empty mask
                img_shape = results['img_shape'][:2]
                results['gt_ellipses'] = np.zeros(img_shape, dtype=np.uint8)
        else:
            # Assign default empty mask
            img_shape = results['img_shape'][:2]
            results['gt_ellipses'] = np.zeros(img_shape, dtype=np.uint8)

        # Handle ellipse annotations
        ellipses_line = results['ann_info'].get('ellipse', [])
        # print(ellipses_line)
        if len(ellipses_line) > 0:
            ellipse_line = np.asarray(ellipses_line.copy())
            results['ellipse_center'] = ellipse_line[:, 0, [0, 1, 0, 1]]  # [num_ellipse, 4]
            results['ellipse_line'] = ellipse_line[:,
                                                   :, [0, 1, 0, 1]].reshape(-1, 4)
        else:
            # Assign empty array
            results['ellipse_center'] = np.zeros((0, 4), dtype=np.float32)
        return results

    def _load_reflective_seg(self, results):
        filename_reflection_mask = results['img_info'].get('filename_reflection_mask', '')
        if filename_reflection_mask:
            filename = osp.join(results['img_prefix'], filename_reflection_mask)
            if osp.exists(filename):
                if self.file_client is None:
                    self.file_client = mmcv.FileClient(**self.file_client_args)
                img_bytes = self.file_client.get(filename)
                results['gt_reflective_seg'] = mmcv.imfrombytes(img_bytes, flag='unchanged').squeeze()
                results['seg_fields'].append('gt_reflective_seg')
            else:
                # Assign default empty mask
                img_shape = results['img_shape'][:2]
                results['gt_reflective_seg'] = np.zeros(img_shape, dtype=np.uint8)
        else:
            # Assign default empty mask
            img_shape = results['img_shape'][:2]
            results['gt_reflective_seg'] = np.zeros(img_shape, dtype=np.uint8)
        return results

    def _load_rotational_seg(self, results):
        filename_rotation_mask = results['img_info'].get(
            'filename_rotation_mask', '')
        if filename_rotation_mask:
            filename = osp.join(
                results['img_prefix'], filename_rotation_mask)
            if osp.exists(filename):
                if self.file_client is None:
                    self.file_client = mmcv.FileClient(**self.file_client_args)
                img_bytes = self.file_client.get(filename)
                results['gt_rotational_seg'] = mmcv.imfrombytes(
                    img_bytes, flag='unchanged').squeeze()
                results['seg_fields'].append('gt_rotational_seg')
            else:
                # Assign default empty mask
                img_shape = results['img_shape'][:2]
                results['gt_rotational_seg'] = np.zeros(
                    img_shape, dtype=np.uint8)
        else:
            # Assign default empty mask
            img_shape = results['img_shape'][:2]
            results['gt_rotational_seg'] = np.zeros(img_shape, dtype=np.uint8)
        return results

    def _load_rotation(self, results):
        """Load rotation annotations with expanded dimensions to match ellipse format."""
        ann_info = results['ann_info']
        rot_annotations = ann_info.get('rot', [])

        centers = []
        orders = []
        max_vertices = 0
        vertices_list = []
        num_vertices = []
        isEllipse = []
        
        # Add is_ellipse
        for rot in rot_annotations:
            if 'isEllipse' in rot:
                if rot['isEllipse'] == True:
                    isEllipse.append(True)
                else:
                    isEllipse.append(False)
            else:
                isEllipse.append(False)
            if 'center' in rot:
                # Expand center to 4 dimensions [x, y, x, y]
                centers.append(rot['center'] * 2)
                orders.append(rot.get('order', 0))

                if 'vertices' in rot:
                    vertices = rot['vertices']
                    max_vertices = max(max_vertices, len(vertices))
                    # Expand each vertex to 4 dimensions [x, y, x, y]
                    vertices_list.append([v * 2 for v in vertices])
                else:
                    vertices_list.append([])

        if centers:
            results['rot_centers'] = np.array(centers, dtype=np.float32)
            results['rot_orders'] = np.array(orders, dtype=np.int32)

            padded_vertices = []
            for vertices in vertices_list:
                if vertices:
                    padded = vertices + [[0, 0, 0, 0]] * \
                        (max_vertices - len(vertices))
                    num_vertices.append(len(vertices))
                else:
                    padded = [[0, 0, 0, 0]] * max_vertices
                    num_vertices.append(0)
                padded_vertices.append(padded)
            
            results['rot_vertices'] = np.array(padded_vertices, dtype=np.float32)
            results['num_vertices'] = np.array(num_vertices, dtype=np.int32)
            results['isEllipse'] = np.array(isEllipse, dtype=bool)
        else:
            # Return empty arrays with expanded dimensions
            results['rot_centers'] = np.zeros((0, 4), dtype=np.float32)
            results['rot_orders'] = np.zeros(0, dtype=np.int32)
            results['rot_vertices'] = np.zeros((0, 0, 4), dtype=np.float32)
            results['num_vertices'] = np.zeros(0, dtype=np.int32)
            results['isEllipse'] = np.zeros(0, dtype=bool)
        return results

    def __call__(self, results):
        # Load image shape for default masks
        results['img_shape'] = results['img'].shape

        if self.with_axis:
            results = self._load_axis(results)
        else:
            # Assign default empty mask
            img_shape = results['img_shape'][:2]
            results['gt_axis'] = np.zeros(img_shape, dtype=np.uint8)

        if self.with_lines:
            results = self._load_lines(results)
        else:
            # Assign empty arrays
            results['gt_lines1'] = np.zeros((0, 4), dtype=np.float32)
            results['gt_lines2'] = np.zeros((0, 4), dtype=np.float32)

        if self.with_ellipses:
            results = self._load_ellipses(results)
        else:
            # Assign default empty mask and empty ellipse_center
            img_shape = results['img_shape'][:2]
            results['gt_ellipses'] = np.zeros(img_shape, dtype=np.uint8)
            results['ellipse_center'] = np.zeros((0, 4), dtype=np.float32)

        if self.with_seg:
            results = self._load_reflective_seg(results)
        else:
            # Assign default empty mask
            img_shape = results['img_shape'][:2]
            results['gt_reflective_seg'] = np.zeros(img_shape, dtype=np.uint8)

        if self.with_seg_rot:
            results = self._load_rotational_seg(results)
        else:
            # Assign default empty mask
            img_shape = results['img_shape'][:2]
            results['gt_rotational_seg'] = np.zeros(img_shape, dtype=np.uint8)

        if self.with_rotation:
            results = self._load_rotation(results)
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_lines={self.with_lines}, '
        repr_str += f'with_ellipses={self.with_ellipses}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'file_client_args={self.file_client_args})'
        return repr_str

class LoadImageFromFile:
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='bgr',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmengine.fileio.file_client.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"channel_order='{self.channel_order}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str
