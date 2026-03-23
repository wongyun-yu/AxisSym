"""
Implementation of ReResNet V2.
@author: Jiaming Han
"""
import math
import os
from collections import OrderedDict

import e2cnn.nn as enn
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from e2cnn import gspaces
from mmengine.model import constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm

from .enn_layers import FIELD_TYPE, build_norm_layer, conv1x1, conv3x3
from .base_backbone import BaseBackbone



class BasicBlock(enn.EquivariantModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 gspace=None,
                 downsample_block=False,
                 fixparams=False):
        super(BasicBlock, self).__init__()
        self.in_type = FIELD_TYPE['regular']( # channel 넘어오면 fixparam setting 따라서 자동으로 지정
            gspace, in_channels, fixparams=fixparams)
        self.out_type = FIELD_TYPE['regular'](
            gspace, out_channels, fixparams=fixparams)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert self.expansion == 1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.downsample_block = downsample_block

        norm1_channels = self.mid_channels if not fixparams \
            else int(self.mid_channels * math.sqrt(gspace.fibergroup.order()))
        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, gspace, norm1_channels, postfix=1)
        
        norm2_channels = out_channels if not fixparams \
            else int(out_channels * math.sqrt(gspace.fibergroup.order()))        
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, gspace, norm2_channels, postfix=2)

        self.conv1 = conv3x3(
            gspace,
            in_channels, 
            self.mid_channels,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            fixparams=fixparams)
        self.add_module(self.norm1_name, norm1)
        self.relu1 = enn.ReLU(self.conv1.out_type, inplace=True)
        self.conv2 = conv3x3(
            gspace,
            self.mid_channels,
            out_channels,
            padding=1,
            bias=False,
            fixparams=fixparams)
        self.add_module(self.norm2_name, norm2)
        self.relu2 = enn.ReLU(self.conv1.out_type, inplace=True)

        self.downsample = downsample

        if self.out_channels != 64 and self.downsample_block:
            self.maxpoo1_1 = enn.PointwiseMaxPool(
                self.conv1.out_type, kernel_size=2, stride=2, padding=0)
            self.maxpool_2 = enn.PointwiseMaxPool(
                self.conv2.out_type, kernel_size=2, stride=2, padding=0)
            self.conv1x1 = conv1x1(gspace, in_channels, out_channels, stride=1, bias=False)
            self.norm =  build_norm_layer(dict(type='BN'), gspace, out_channels)[1]

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu1(out)
            if self.out_channels != 64 and self.downsample_block:
                out = self.maxpoo1_1(out)

            out = self.conv2(out)
            out = self.norm2(out)

            # 첫 번째는 maxpool로 이미 줄이니까,, downsample 하지말기
            # if self.downsample_block and self.out_channels != 64:
            #     identity = self.downsample(x) # 1x1 conv stride 2

            if self.downsample_block and self.out_channels != 64:
                identity = self.conv1x1(identity)
                identity = self.norm(identity)
                identity = self.maxpool_2(identity)

            out += identity

            # if self.downsample_block and self.out_channels != 64:
            #     out = self.maxpool(out)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu2(out)

        return out

    def evaluate_output_shape(self, input_shape):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.downsample is not None:
            return self.downsample.evaluate_output_shape(input_shape)
        else:
            return input_shape

    def export(self):
        self.eval()
        submodules = []
        # convert all the submodules if necessary
        for name, module in self._modules.items():
            if hasattr(module, 'export'):
                module = module.export()
            submodules.append((name, module))
        return torch.nn.ModuleDict(OrderedDict(submodules))

class Bottleneck(enn.EquivariantModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=4,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 gspace=None,
                 downsample_block=False,
                 fixparams=False):
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        self.in_type = FIELD_TYPE['regular'](
            gspace, in_channels, fixparams=fixparams)
        self.out_type = FIELD_TYPE['regular'](
            gspace, out_channels, fixparams=fixparams)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.downsample_block = downsample_block

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, gspace, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, gspace, self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, gspace, out_channels, postfix=3)

        self.conv1 = conv1x1(
            gspace,
            in_channels,
            self.mid_channels,
            stride=self.conv1_stride,
            bias=False,
            fixparams=fixparams)
        self.add_module(self.norm1_name, norm1)
        self.relu1 = enn.ReLU(self.conv1.out_type, inplace=True)
        self.conv2 = conv3x3(
            gspace,
            self.mid_channels,
            self.mid_channels,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            fixparams=fixparams)

        self.add_module(self.norm2_name, norm2)
        self.relu2 = enn.ReLU(self.conv2.out_type, inplace=True)
        self.conv3 = conv1x1(
            gspace,
            self.mid_channels,
            out_channels,
            bias=False,
            fixparams=fixparams)
        self.add_module(self.norm3_name, norm3)
        self.relu3 = enn.ReLU(self.conv3.out_type, inplace=True)

        self.downsample = downsample

        if self.downsample_block:
            self.conv1x1 = conv1x1(gspace, in_channels, out_channels, stride=1, bias=False)
            self.norm =  build_norm_layer(dict(type='BN'), gspace, out_channels)[1] 

            if self.out_channels != 256:
                self.maxpoo1_1 = enn.PointwiseMaxPool(
                    self.conv1.out_type, kernel_size=2, stride=2, padding=0)
                self.maxpool_2 = enn.PointwiseMaxPool(
                    self.conv3.out_type, kernel_size=2, stride=2, padding=0)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    # conv1 마치고 downsample 하는걸로 통일
    def forward(self, x):
        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu1(out)

            if self.out_channels != 256 and self.downsample_block:
                out = self.maxpoo1_1(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu2(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample_block:
                identity = self.conv1x1(identity)
                identity = self.norm(identity)
                if self.out_channels != 256: # 첫 번째는 maxpool 있으니까 downsample 안함
                    identity = self.maxpool_2(identity)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu3(out)

        return out

    def evaluate_output_shape(self, input_shape):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.downsample is not None:
            return self.downsample.evaluate_output_shape(input_shape)
        else:
            return input_shape

    def export(self):
        self.eval()
        submodules = []
        # convert all the submodules if necessary
        for name, module in self._modules.items():
            if hasattr(module, 'export'):
                module = module.export()
            submodules.append((name, module))
        return torch.nn.ModuleDict(OrderedDict(submodules))


def get_expansion(block, expansion=None):
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, BasicBlock):
            expansion = 1
        elif issubclass(block, Bottleneck):
            expansion = 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion

class ResLayer(nn.Sequential):
    def __init__(self,
                 block, # basicblock
                 num_blocks, # 3 -> 4 -> 6 -> 3
                 in_channels, # 64 -> 64 -> 128 -> 256
                 out_channels, # 64 -> 128 -> 256 -> 512
                 expansion=None, 
                 stride=1, # 1 -> 2 -> 2 -> 2
                 avg_down=False, # False
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 gspace=None, # gspaces.FlipRot2dOnR2(8)
                 fixparams=False,
                 channel_multiplier=1,
                 **kwargs):
        self.block = block
        self.expansion = get_expansion(block, expansion)

        in_channels *= channel_multiplier
        out_channels *= channel_multiplier

        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride 
            if avg_down and stride != 1:
                conv_stride = 1
                in_mul = int((in_channels / gspace.fibergroup.order()))
                in_type = enn.FieldType(gspace, [gspace.regular_repr] * in_mul)
                downsample.append(
                    enn.PointwiseAvgPool(
                        in_type,
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True))
                
            norm_out_channels = out_channels

            downsample.extend([
                conv1x1(gspace, in_channels, out_channels,
                        stride=conv_stride, bias=False, fixparams=fixparams),
                build_norm_layer(norm_cfg, gspace, norm_out_channels)[1]
            ])
            downsample = enn.SequentialModule(*downsample)

        layers = []
        layers.append(
            block( # block이 일반적으로 conv 2개짜리
                in_channels=in_channels,
                out_channels=out_channels,
                expansion=self.expansion,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                gspace=gspace,
                fixparams=fixparams,
                downsample_block=True,
                **kwargs))
        in_channels = out_channels
        for i in range(1, num_blocks): # 3, 4, 6, 3 각각 함수 따로 호출 (각각에서 위에꺼 하나 뺀거)
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=self.expansion,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    gspace=gspace,
                    fixparams=fixparams,
                    downsample_block=False,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)

    def export(self):
        self.eval()
        submodules = []
        # convert all the submodules if necessary
        for name, module in self._modules.items():
            if hasattr(module, 'export'):
                module = module.export()
            submodules.append((name, module))
        return torch.nn.ModuleDict(OrderedDict(submodules))

class ReResNet(BaseBackbone):

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                 num_stages=4,
                 strides = (1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3,),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 orientation=8,
                 fixparams=False,
                 channel_multiplier=1):
        super(ReResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.expansion = get_expansion(self.block, expansion)

        self.orientation = orientation
        self.fixparams = fixparams
        self.channel_multiplier = channel_multiplier
        self.gspace = gspaces.FlipRot2dOnR2(orientation)
        self.in_type = enn.FieldType(
            self.gspace, [self.gspace.trivial_repr] * 3)

        self._make_stem_layer(self.gspace, in_channels, stem_channels)

        self.res_layers = []
        _in_channels = stem_channels
        _out_channels = base_channels * self.expansion
        for i, num_blocks in enumerate(self.stage_blocks): # [3, 4, 6, 3]
            stride = strides[i]
            dilation = dilations[i]
            res_layer = self.make_res_layer(
                block=self.block, # basicblock
                num_blocks=num_blocks, # 3 -> 4 -> 6 -> 3
                in_channels=_in_channels, # 64 -> 64 -> 128 -> 256
                out_channels=_out_channels, # 64 -> 128 -> 256 -> 512
                expansion=self.expansion, # 1
                stride=stride, # 1 -> 2 -> 2 ->2
                dilation=dilation, # 1 -> 1 -> 1 -> 1
                style=self.style, # 'pytorch'
                avg_down=self.avg_down, # False
                with_cp=with_cp, # False
                conv_cfg=conv_cfg, # None
                norm_cfg=norm_cfg, # dict(type='BN', requires_grad=True)
                gspace=self.gspace, # gspaces.FlipRot2dOnR2(8)
                fixparams=self.fixparams, # False
                channel_multiplier=self.channel_multiplier) # 1
            _in_channels = _out_channels
            _out_channels *= 2
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = res_layer[-1].out_channels

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, gspace, in_channels, stem_channels):
        if not self.deep_stem:
            in_type = enn.FieldType(
                gspace, in_channels * [gspace.trivial_repr])
            # out_type = FIELD_TYPE['regular'](gspace, stem_channels, fixparams=self.fixparams) 
            stem_out = int((stem_channels / gspace.fibergroup.order()) * self.channel_multiplier)
            out_type = enn.FieldType(gspace, [gspace.regular_repr] * stem_out)

            self.conv1 = enn.R2Conv(in_type, out_type, 7, #723
                                    stride=2,
                                    padding=3,
                                    bias=False,
                                    sigma=None,                
                                    frequencies_cutoff=lambda r: 3 * r)
            
            # stem_channels = stem_channels if not self.fixparams \
            #     else int(stem_channels * math.sqrt(gspace.fibergroup.order()))
            stem_channels = int(stem_channels * self.channel_multiplier)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, gspace, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = enn.ReLU(self.conv1.out_type, inplace=True)
        self.maxpool = enn.PointwiseMaxPool( # 321
            self.conv1.out_type, kernel_size=2, stride=2, padding=0)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if not self.deep_stem:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        super(ReResNet, self).init_weights(pretrained)
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

    def forward(self, x):
        outs = []
        if not self.deep_stem:
            x = enn.GeometricTensor(x, self.in_type) # image input to trivial representation
            x = self.conv1(x) # / 2 -> 256
            x = self.norm1(x)
            x = self.relu(x)
            outs.append(x)
        x = self.maxpool(x) # -> 128
        # print(x.shape)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i != (len(self.res_layers) - 1):
                outs.append(x)
            else:
                final_output = x
        
        return final_output, outs

    def train(self, mode=True):
        super(ReResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def export(self):
        self.eval()
        submodules = []
        # convert all the submodules if necessary
        for name, module in self._modules.items():
            if hasattr(module, 'export'):
                module = module.export()
            submodules.append((name, module))
        return torch.nn.ModuleDict(OrderedDict(submodules))
