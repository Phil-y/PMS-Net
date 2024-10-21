from torch import nn
import torch.nn.functional as F
import math
from functools import partial



class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))

# class Gate_Transport_Block(nn.Module):
#     def __init__(self, in_c, out_c, kernel_size):
#         super().__init__()
#         self.w1 = nn.Sequential(
#             DepthWiseConv2d(in_c, in_c, kernel_size, padding=kernel_size // 2),
#             nn.Sigmoid()
#         )
#
#         self.w2 = nn.Sequential(
#             DepthWiseConv2d(in_c, in_c, kernel_size + 2, padding=(kernel_size + 2) // 2),
#             # nn.Sigmoid()
#             nn.GELU()
#         )
#         self.wo = nn.Sequential(
#             DepthWiseConv2d(in_c, out_c, kernel_size + 2),
#             nn.GELU()
#         )
#
#         self.cw = nn.Sequential(
#             DepthWiseConv2d(in_c, out_c, kernel_size + 2),
#             nn.GELU()
#         )
#         # self.cw = nn.Conv2d(in_c, out_c, 1)
#     def forward(self, x):
#         x1, x2 = self.w1(x), self.w2(x)
#         out = self.wo(x1 + x2) + self.cw(x)
#         return out

# class Split_Combination_Gate_Bridge(nn.Module):
#     def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[7, 5, 2, 1]):
#         super().__init__()
#         self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
#         group_size = dim_xl // 2
#         self.g0 = nn.Sequential(
#             LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
#             nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
#                       padding=(k_size + (k_size - 1) * (d_list[0] - 1)) // 2,
#                       dilation=d_list[0], groups=group_size + 1)
#         )
#         self.g1 = nn.Sequential(
#             LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
#             nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
#                       padding=(k_size + (k_size - 1) * (d_list[1] - 1)) // 2,
#                       dilation=d_list[1], groups=group_size + 1)
#         )
#         self.g2 = nn.Sequential(
#             LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
#             nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
#                       padding=(k_size + (k_size - 1) * (d_list[2] - 1)) // 2,
#                       dilation=d_list[2], groups=group_size + 1)
#         )
#         self.g3 = nn.Sequential(
#             LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
#             nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
#                       padding=(k_size + (k_size - 1) * (d_list[3] - 1)) // 2,
#                       dilation=d_list[3], groups=group_size + 1)
#         )
#         self.gtb = Gate_Transport_Block(dim_xl * 2 + 4, dim_xl, 1)
#     def forward(self, xh, xl, mask):
#         xh = self.pre_project(xh)
#         xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode='bilinear', align_corners=True)
#         xh = torch.chunk(xh, 4, dim=1)
#         xl = torch.chunk(xl, 4, dim=1)
#         x0 = self.g0(torch.cat((xh[0], xl[0], mask), dim=1))
#         x1 = self.g1(torch.cat((xh[1], xl[1], mask), dim=1))
#         x2 = self.g2(torch.cat((xh[2], xl[2], mask), dim=1))
#         x3 = self.g3(torch.cat((xh[3], xl[3], mask), dim=1))
#         x = torch.cat((x0, x1, x2, x3), dim=1)
#         x = self.gtb(x)
#         return x






class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = self.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale



class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = self.sigmoid(x_out)
        return x * scale.expand_as(x)


class DoubleGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(DoubleGate, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out




nonlinearity = partial(F.relu, inplace=True)
class MDDG(nn.Module):
    def __init__(self, channel):
        super(MDDG, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
        self.dg = DoubleGate(channel)
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        out = self.dg(out)
        return out


class SCEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.CE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.SE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())
    def forward(self, x):
        x = x * self.CE(x) + x * self.SE(x)
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)





from functools import partial
import torch
from timm.models.layers import DropPath, trunc_normal_
from torch import nn
class PreConv(nn.Module):
    def __init__(self,in_channels, out_channels, stride=1):
        super(PreConv, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        # if stride == 2:
        #     self.avgpool = nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False)
        #     self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        #     self.norm = norm_layer(out_channels)
        if in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        # else:
        #     self.avgpool = nn.Identity()
        #     self.conv = nn.Identity()
        #     self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.avgpool(x)))

NORM_EPS = 1e-5
class DHConv(nn.Module):
    def __init__(self, out_channels,):
        super(DHConv, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        self.conv33 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,  bias=False)
        self.norm = norm_layer(out_channels)
        self.activte = nn.ReLU(inplace=True)
        self.conv11 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
    def forward(self, x):
        out = self.conv33(x)
        out = self.norm(out)
        out = self.activte(out)
        out = self.conv11(out)
        return out

class PDCA(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, path_dropout=0,
                 drop=0, mlp_ratio=3):
        super(PDCA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)

        self.pc = PreConv(in_channels, out_channels, stride)
        self.dhc = DHConv(out_channels,)
        self.norm = norm_layer(out_channels)
        self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop, bias=True)
        self.dropout = DropPath(path_dropout)
        self.is_bn_merged = False
    def merge_bn(self):
        if not self.is_bn_merged:
            self.mlp.merge_bn(self.norm)
            self.is_bn_merged = True
    def forward(self, x):
        x = self.pc(x)
        x = x + self.dropout(self.dhc(x))
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm(x)
        else:
            out = x
        x = x + self.dropout(self.mlp(out))
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class Mlp(nn.Module):
    def __init__(self, in_features, out_features=None, mlp_ratio=None, drop=0., bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_dim = _make_divisible(in_features * mlp_ratio, 32)
        self.conv1 = nn.Conv2d(in_features, hidden_dim, kernel_size=1, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, out_features, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)
    def merge_bn(self, pre_norm):
        merge_pre_bn(self.conv1, pre_norm)
    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def merge_pre_bn(module, pre_bn_1, pre_bn_2=None):
    """ Merge pre BN to reduce inference runtime.
    """
    weight = module.weight.data
    if module.bias is None:
        zeros = torch.zeros(module.out_channels, device=weight.device).type(weight.type())
        module.bias = nn.Parameter(zeros)
    bias = module.bias.data
    if pre_bn_2 is None:
        assert pre_bn_1.track_running_stats is True, "Unsupport bn_module.track_running_stats is False"
        assert pre_bn_1.affine is True, "Unsupport bn_module.affine is False"

        scale_invstd = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        extra_weight = scale_invstd * pre_bn_1.weight
        extra_bias = pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd
    else:
        assert pre_bn_1.track_running_stats is True, "Unsupport bn_module.track_running_stats is False"
        assert pre_bn_1.affine is True, "Unsupport bn_module.affine is False"

        assert pre_bn_2.track_running_stats is True, "Unsupport bn_module.track_running_stats is False"
        assert pre_bn_2.affine is True, "Unsupport bn_module.affine is False"

        scale_invstd_1 = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        scale_invstd_2 = pre_bn_2.running_var.add(pre_bn_2.eps).pow(-0.5)

        extra_weight = scale_invstd_1 * pre_bn_1.weight * scale_invstd_2 * pre_bn_2.weight
        extra_bias = scale_invstd_2 * pre_bn_2.weight *(pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd_1 - pre_bn_2.running_mean) + pre_bn_2.bias

    if isinstance(module, nn.Linear):
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
    elif isinstance(module, nn.Conv2d):
        assert weight.shape[2] == 1 and weight.shape[3] == 1
        weight = weight.reshape(weight.shape[0], weight.shape[1])
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
        weight = weight.reshape(weight.shape[0], weight.shape[1], 1, 1)
    bias.add_(extra_bias)
    module.weight.data = weight
    module.bias.data = bias


# 8,16,24,32,48,64
# GFLOPs :0.18, Params : 0.62
# 8,16,32,48,64,96
# GFLOPs :0.35, Params : 1.28
# 16,24,32,48,64,128
# GFLOPs :0.42, Params : 1.81
# 8,16,32,64,128,160
# GFLOPs :0.86, Params : 3.75
# 16,32,48,64,128,256
# GFLOPs :1.18, Params : 6.70
# 16,32,64,128,160,256
# GFLOPs :2.34, Params : 8.68
# 16,32,64,128,256,512
# GFLOPs :4.58, Params : 26.64

'''img_size = 512 BatchSize = 4(2,8) learning_rate = 1e-3 optimizer = Adam(AdamW)'''

class PMS_Net(nn.Module):
    def __init__(self, n_classes=1, n_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True):
        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.bridge = bridge
        self.gt_ds = gt_ds

        self.encoder1 = nn.Sequential(
            nn.Conv2d(n_channels, c_list[0], 3, stride=1, padding=1),
        )

        self.encoder2 = nn.Sequential(
            DepthWiseConv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )

        self.encoder3 = nn.Sequential(
            DepthWiseConv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )

        self.encoder4 = nn.Sequential(
            PDCA(c_list[2], c_list[3], 3),
            MDDG(c_list[3]),
        )


        self.encoder5 = nn.Sequential(
            PDCA(c_list[3], c_list[4], 3),
            MDDG(c_list[4]),
        )


        self.encoder6 = nn.Sequential(
            PDCA(c_list[4], c_list[5],3),
            MDDG(c_list[5]),
        )

        # build Bottleneck layers
        self.sce = SCEBlock(c_list[5])

        # if bridge:
        #     self.SCGB1 = Split_Combination_Gate_Bridge(c_list[1], c_list[0])
        #     self.SCGB2 = Split_Combination_Gate_Bridge(c_list[2], c_list[1])
        #     self.SCGB3 = Split_Combination_Gate_Bridge(c_list[3], c_list[2])
        #     self.SCGB4 = Split_Combination_Gate_Bridge(c_list[4], c_list[3])
        #     self.SCGB5 = Split_Combination_Gate_Bridge(c_list[5], c_list[4])
        # if gt_ds:
        #     self.gt_conv1 = nn.Sequential(nn.Conv2d(c_list[4], 1, 1))
        #     self.gt_conv2 = nn.Sequential(nn.Conv2d(c_list[3], 1, 1))
        #     self.gt_conv3 = nn.Sequential(nn.Conv2d(c_list[2], 1, 1))
        #     self.gt_conv4 = nn.Sequential(nn.Conv2d(c_list[1], 1, 1))
        #     self.gt_conv5 = nn.Sequential(nn.Conv2d(c_list[0], 1, 1))


        self.decoder1 = nn.Sequential(
            MDDG(c_list[5]),
            PDCA(c_list[5], c_list[4], 3)
        )


        self.decoder2 = nn.Sequential(
            MDDG(c_list[4]),
            PDCA(c_list[4], c_list[3], 3)
        )

        self.decoder3 = nn.Sequential(
            MDDG(c_list[3]),
            PDCA(c_list[3], c_list[2], 3),
        )


        self.decoder4 = nn.Sequential(
            DepthWiseConv2d(c_list[2], c_list[1], 3),
        )

        self.decoder5 = nn.Sequential(
            DepthWiseConv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])

        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], n_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels

            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        # out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3_1(self.encoder3(out))),2,2))
        t3 = out  # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        # out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4_1(self.encoder4(out))),2,2))
        t4 = out  # b, c3, H/16, W/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        # out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5_1(self.encoder5(out))),2,2))
        t5 = out  # b, c4, H/32, W/32

        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32
        # out = F.gelu(self.encoder6_1(self.encoder6(out))) # b, c5, H/32, W/32
        t6 = out

        out_bottleneck = self.sce(out)

        out5 = F.gelu(self.dbn1(self.decoder1(out_bottleneck)))  # b, c4, H/32, W/32
        # out5 = F.gelu(self.dbn1(self.decoder1_1(self.decoder1(out)))) # b, c4, H/32, W/32
        # if self.gt_ds:
        #     gt_pre5 = self.gt_conv1(out5)
        #     t5 = self.SCGB5(t6, t5, gt_pre5)
        #     # gt_pre5 = F.interpolate(gt_pre5, scale_factor=32, mode ='bilinear', align_corners=True)
        # else:
        #     t5 = self.SCGB5(t6, t5)
        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16
        # out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2_1(self.decoder2(out5))),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c3, H/16, W/16
        # if self.gt_ds:
        #     gt_pre4 = self.gt_conv2(out4)
        #     t4 = self.SCGB4(t5, t4, gt_pre4)
        #     # gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode ='bilinear', align_corners=True)
        # else:
        #     t4 = self.SCGB4(t5, t4)
        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8
        # out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3_1(self.decoder3(out4))),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c2, H/8, W/8
        # if self.gt_ds:
        #     gt_pre3 = self.gt_conv3(out3)
        #     t3 = self.SCGB3(t4, t3, gt_pre3)
        #     # gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode ='bilinear', align_corners=True)
        # else:
        #     t3 = self.SCGB3(t4, t3)
        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        # out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4_1(self.decoder4(out3))),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c1, H/4, W/4
        # if self.gt_ds:
        #     gt_pre2 = self.gt_conv4(out2)
        #     t2 = self.SCGB2(t3, t2, gt_pre2)
        #     # gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode ='bilinear', align_corners=True)
        # else:
        #     t2 = self.SCGB2(t3, t2)
        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2
        # if self.gt_ds:
        #     gt_pre1 = self.gt_conv5(out1)
        #     t1 = self.SCGB1(t2, t1, gt_pre1)
        #     # gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode ='bilinear', align_corners=True)
        # else:
        #     t1 = self.SCGB1(t2, t1)
        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W

        return torch.sigmoid(out0)


# from thop import profile
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = PMS_Net().to(device)
# input = torch.randn(1, 3, 224, 224).to(device)
# flops, params = profile(model, inputs=(input, ))
# print("GFLOPs :{:.2f}, Params : {:.2f}".format(flops/1e9,params/1e6)) #flops单位G，para单位M
# # GFLOPs :0.18, Params : 0.62