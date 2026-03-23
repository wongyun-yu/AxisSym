import torch
import torch.nn as nn
import torchvision


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int
        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Ensure that in_channels and out_channels are divisible by groups
        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        assert out_channels % groups == 0, "out_channels must be divisible by groups"
        
        # Offset and modulator convolutions
        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1] * groups,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     groups=groups,
                                     bias=True)
        
        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1] * groups,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        groups=groups,
                                        bias=True)
        
        # Regular convolution
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      groups=groups,
                                      bias=bias)
        
        # Initialize weights
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

    def forward(self, x):
        batch_size, _, height, width = x.shape
        
        # Compute offsets and modulators
        offset = self.offset_conv(x)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        # Reshape offsets and modulators for grouped operation
        offset = offset.view(batch_size, self.groups, -1, height, width)
        modulator = modulator.view(batch_size, self.groups, -1, height, width)
        
        # Perform grouped deformable convolution
        x = torch.cat([
            torchvision.ops.deform_conv2d(
                input=x[:, i*x.size(1)//self.groups:(i+1)*x.size(1)//self.groups],
                offset=offset[:, i],
                weight=self.regular_conv.weight[i*self.regular_conv.weight.size(0)//self.groups:
                                                (i+1)*self.regular_conv.weight.size(0)//self.groups],
                bias=self.regular_conv.bias[i*self.regular_conv.out_channels//self.groups:
                                            (i+1)*self.regular_conv.out_channels//self.groups] if self.regular_conv.bias is not None else None,
                padding=self.padding,
                mask=modulator[:, i],
                stride=self.stride,
                dilation=self.dilation
            ) for i in range(self.groups)
        ], dim=1)
        
        return x