import torch
import torch.nn as nn
import math
from torch.nn.modules.utils import _ntuple
from collections import OrderedDict
import torch.nn.functional as F

PRIMITIVES = {
    "graph": lambda C_in, C_out, expansion, stride, length, **kwargs: Graphormer(
        C_in, C_out, stride, length
    ),
    "trans": lambda C_in, C_out, expansion, stride, length, **kwargs: Trans(
        C_in, C_out, stride, length
    ),
    "GSAU": lambda C_in, C_out, expansion, stride, length, **kwargs: GSAU(
        C_in, C_out, stride, length
    ),
    # "MultiKernalConv": lambda C_in, C_out, expansion, stride, length, **kwargs: GSAU(
    #     C_in, C_out, stride, length
    # ),
    "unet": lambda C_in, C_out, expansion, stride, length, **kwargs: UNet(
        C_in, C_out, stride, length
    ),
    "skip": lambda C_in, C_out, expansion, stride, length, **kwargs: Identity(
        C_in, C_out, stride
    ),
    "ir_k3_e1": lambda C_in, C_out, expansion, stride, length, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=3, **kwargs
    ),
    "ir_k3_e3": lambda C_in, C_out, expansion, stride, length, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=3, **kwargs
    ),
    "ir_k3_e6": lambda C_in, C_out, expansion, stride, length, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=3, **kwargs
    ),
    "ir_k5_e1": lambda C_in, C_out, expansion, stride, length, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=5, **kwargs
    ),
    "ir_k5_e3": lambda C_in, C_out, expansion, stride, length, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=5, **kwargs
    ),
    "ir_k5_e6": lambda C_in, C_out, expansion, stride, length, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=5, **kwargs
    ),
    "ir_k7_e1": lambda C_in, C_out, expansion, stride, length, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=7, **kwargs
    ),
    "ir_k7_e3": lambda C_in, C_out, expansion, stride, length, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=7, **kwargs
    ),
    "ir_k7_e6": lambda C_in, C_out, expansion, stride, length, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=7, **kwargs
    ),
    "ir_k3_e1_se": lambda C_in, C_out, expansion, stride, length, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=3, se=True, **kwargs
    ),
    "ir_k3_e3_se": lambda C_in, C_out, expansion, stride, length, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=3, se=True, **kwargs
    ),
    "ir_k3_e6_se": lambda C_in, C_out, expansion, stride, length, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=3, se=True, **kwargs
    ),
    "ir_k5_e1_se": lambda C_in, C_out, expansion, stride, length, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=5, se=True, **kwargs
    ),
    "ir_k5_e3_se": lambda C_in, C_out, expansion, stride, length, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=5, se=True, **kwargs
    ),
    "ir_k5_e6_se": lambda C_in, C_out, expansion, stride, length, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=5, se=True, **kwargs
    ),
    "ir_k7_e1_se": lambda C_in, C_out, expansion, stride, length, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=7, se=True, **kwargs
    ),
    "ir_k7_e3_se": lambda C_in, C_out, expansion, stride, length, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=7, se=True, **kwargs
    ),
    "ir_k7_e6_se": lambda C_in, C_out, expansion, stride, length, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=7, se=True, **kwargs
    )
}

class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class Conv2d(torch.nn.Conv2d):
    def forward(self, x):
        if x.numel() > 0:
            return super(Conv2d, self).forward(x)
        # get output shape

        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
            )
        ]
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)
    
class Conv1d(torch.nn.Conv1d):
    def forward(self, x):
        if x.numel() > 0:
            return super(Conv1d, self).forward(x)
        # get output shape

        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
            )
        ]
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)

class BatchNorm2d(torch.nn.BatchNorm2d):
    def forward(self, x):
        if x.numel() > 0:
            return super(BatchNorm2d, self).forward(x)
        # get output shape
        output_shape = x.shape
        return _NewEmptyTensorOp.apply(x, output_shape)
class BatchNorm1d(torch.nn.BatchNorm1d):
    def forward(self, x):
        if x.numel() > 0:
            return super(BatchNorm1d, self).forward(x)
        # get output shape
        output_shape = x.shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm1d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1)
        bias = bias.reshape(1, -1, 1)
        return x * scale + bias


class FrozenBatchNorm1d(nn.Module):
    """
    BatchNorm1d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm1d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

class ConvBNRelu2d(nn.Sequential):
    def __init__(
        self,
        input_depth,
        output_depth,
        kernel,
        stride,
        pad,
        no_bias,
        use_relu,
        bn_type,
        group=1,
        *args,
        **kwargs
    ):
        super(ConvBNRelu2d, self).__init__()

        assert use_relu in ["relu", None]
        if isinstance(bn_type, (list, tuple)):
            assert len(bn_type) == 2
            assert bn_type[0] == "gn"
            gn_group = bn_type[1]
            bn_type = bn_type[0]
        assert bn_type in ["bn", "af", "gn", None]
        assert stride in [1, 2, 4, 5]

        op = Conv2d(
            input_depth,
            output_depth,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            bias=not no_bias,
            groups=group,
            *args,
            **kwargs
        )
        nn.init.kaiming_normal_(op.weight, mode="fan_out", nonlinearity="relu")
        if op.bias is not None:
            nn.init.constant_(op.bias, 0.0)
        self.add_module("conv", op)

        if bn_type == "bn":
            bn_op = BatchNorm2d(output_depth)
        elif bn_type == "gn":
            bn_op = nn.GroupNorm(num_groups=gn_group, num_channels=output_depth)
        elif bn_type == "af":
            bn_op = FrozenBatchNorm2d(output_depth)
        if bn_type is not None:
            self.add_module("bn", bn_op)

        if use_relu == "relu":
            self.add_module("relu", nn.ReLU(inplace=True))

class ConvBNRelu(nn.Sequential):
    def __init__(
        self,
        input_depth,
        output_depth,
        kernel,
        stride,
        pad,
        no_bias,
        use_relu,
        bn_type,
        group=1,
        *args,
        **kwargs
    ):
        super(ConvBNRelu, self).__init__()

        assert use_relu in ["relu", None]
        if isinstance(bn_type, (list, tuple)):
            assert len(bn_type) == 2
            assert bn_type[0] == "gn"
            gn_group = bn_type[1]
            bn_type = bn_type[0]
        assert bn_type in ["bn", "af", "gn", None]
        assert stride in [1, 2, 4, 5]

        op = Conv1d(
            input_depth,
            output_depth,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            bias=not no_bias,
            groups=group,
            *args,
            **kwargs
        )
        nn.init.kaiming_normal_(op.weight, mode="fan_out", nonlinearity="relu")
        if op.bias is not None:
            nn.init.constant_(op.bias, 0.0)
        self.add_module("conv", op)

        if bn_type == "bn":
            bn_op = BatchNorm1d(output_depth)
        elif bn_type == "gn":
            bn_op = nn.GroupNorm(num_groups=gn_group, num_channels=output_depth)
        elif bn_type == "af":
            bn_op = FrozenBatchNorm1d(output_depth)
        if bn_type is not None:
            self.add_module("bn", bn_op)

        if use_relu == "relu":
            self.add_module("relu", nn.ReLU(inplace=True))


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class Identity(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Identity, self).__init__()
        self.output_depth = C_out # ANNA's code here
        self.conv = (
            ConvBNRelu(
                C_in,
                C_out,
                kernel=1,
                stride=stride,
                pad=0,
                no_bias=1,
                use_relu="relu",
                bn_type="bn",
            )
            if C_in != C_out or stride != 1
            else None
        )

    def forward(self, x):
        if self.conv:
            out = self.conv(x)
        else:
            out = x
        return out


class DoubleConv1d(nn.Module):
    """(convolution => [BN] => ReLU) * 2, for 1D data"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv1d, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class Up(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=False) -> None:
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(in_channel, out_channel, kernel_size=2, stride=2)
        self.conv = DoubleConv1d(in_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, C_in, C_out, stride, length, bilinear=False):
        super(UNet, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.length = length

        self.inc = DoubleConv1d(C_in, 128)
        self.down1 = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1d(128, 256)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1d(256, 512)
        )
        factor = 2 if bilinear else 1
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.outlayer = ConvBNRelu(
            32,
            self.C_out,
            kernel=3,
            stride=self.stride,
            pad=1,
            no_bias=1,
            use_relu="relu",
            bn_type="bn",
            group=1,
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outlayer(x)
        return x
class Trans(nn.Module):
    def __init__(self, C_in, C_out, stride, length):
        super(Trans, self).__init__()
        self.d_model = length
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.trans = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=5, batch_first=True)
        self.outlayer = ConvBNRelu(
            self.C_in,
            self.C_out,
            kernel=3,
            stride=self.stride,
            pad=1,
            no_bias=1,
            use_relu="relu",
            bn_type="bn",
            group=1,
        )

    def forward(self, x):

        out_mid = self.trans(x)


        return self.outlayer(out_mid)



class Graphormer(nn.Module):
    def __init__(self, C_in, C_out, stride, length):
        super().__init__()
        self.length = length
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.rank_encoder = nn.Linear(self.length,self.length)
        self.mhsa = EncoderLayer(hidden_size=self.length, ffn_size=self.length, 
                    dropout_rate=0.5, attention_dropout_rate=0.5, num_heads=5)
        self.outlayer = ConvBNRelu(
            self.C_in,
            self.C_out,
            kernel=3,
            stride=self.stride,
            pad=1,
            no_bias=1,
            use_relu="relu",
            bn_type="bn",
            group=1,
        )
    
    def forward(self, x):
        
        if x.dim() == 3:
            x = x.unsqueeze(1)
        rank = self._getrank_(x)
        node_feature = x + self.rank_encoder(rank)
        out = self.mhsa(node_feature)
    
        out = out.squeeze(1)
        return self.outlayer(out)
        
    def _getrank_(self, x):
        _, idx = torch.sort(x, descending=True)
        _, rank = idx.sort()
        rank = (self.length-rank)/self.length
        return rank


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x



class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)
        x = x.unsqueeze(1)

        assert x.size() == orig_q_size
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x
class GSAU(nn.Module):
    r"""Gated Spatial Attention Unit.

    Args:
        n_feats: Number of input channels

    """

    def __init__(self, C_in, C_out, stride, length) -> None:
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.length = length    
        n_feats = C_in
        i_feats = n_feats * 2

        self.Conv1 = nn.Conv1d(n_feats, i_feats,kernel_size=1)
        self.DWConv1 = nn.Conv1d(
            n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        self.Conv2 = nn.Conv1d(n_feats, n_feats,kernel_size=1)

        self.norm = nn.LayerNorm( [n_feats,self.length])
        self.scale = nn.Parameter(torch.zeros(
            (1, n_feats, 1)), requires_grad=True)
        self.outlayer = ConvBNRelu(
            self.C_in,
            self.C_out,
            kernel=3,
            stride=self.stride,
            pad=1,
            no_bias=1,
            use_relu="relu",
            bn_type="bn",
            group=1,
        )
    

    def forward(self, x) -> torch.Tensor:
        shortcut = x.clone()

        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)
        x = x * self.scale + shortcut
        return self.outlayer(x)





class IRFBlock(nn.Module):
    def __init__(
        self,
        input_depth,
        output_depth,
        expansion,
        stride,
        bn_type="bn",
        kernel=3,
        width_divisor=1,
        shuffle_type=None,
        pw_group=1,
        se=False,
        cdw=False,
        dw_skip_bn=False,
        dw_skip_relu=False,
    ):
        super(IRFBlock, self).__init__()

        assert kernel in [1, 3, 5, 7], kernel

        self.use_res_connect = stride == 1 and input_depth == output_depth
        self.output_depth = output_depth

        mid_depth = int(input_depth * expansion)
        mid_depth = _get_divisible_by(mid_depth, width_divisor, width_divisor)

        # pw
        self.pw = ConvBNRelu(
            input_depth,
            mid_depth,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=1,
            use_relu="relu",
            bn_type=bn_type,
            group=pw_group,
        )

        # negative stride to do upsampling
        self.upscale, stride = _get_upsample_op(stride)

        # dw
        if kernel == 1:
            self.dw = nn.Sequential()
        elif cdw:
            dw1 = ConvBNRelu(
                mid_depth,
                mid_depth,
                kernel=kernel,
                stride=stride,
                pad=(kernel // 2),
                group=mid_depth,
                no_bias=1,
                use_relu="relu",
                bn_type=bn_type,
            )
            dw2 = ConvBNRelu(
                mid_depth,
                mid_depth,
                kernel=kernel,
                stride=1,
                pad=(kernel // 2),
                group=mid_depth,
                no_bias=1,
                use_relu="relu" if not dw_skip_relu else None,
                bn_type=bn_type if not dw_skip_bn else None,
            )
            self.dw = nn.Sequential(OrderedDict([("dw1", dw1), ("dw2", dw2)]))
        else:
            self.dw = ConvBNRelu(
                mid_depth,
                mid_depth,
                kernel=kernel,
                stride=stride,
                pad=(kernel // 2),
                group=mid_depth,
                no_bias=1,
                use_relu="relu" if not dw_skip_relu else None,
                bn_type=bn_type if not dw_skip_bn else None,
            )

        # pw-linear
        self.pwl = ConvBNRelu(
            mid_depth,
            output_depth,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=1,
            use_relu=None,
            bn_type=bn_type,
            group=pw_group,
        )

        self.shuffle_type = shuffle_type
        if shuffle_type is not None:
            self.shuffle = ChannelShuffle(pw_group)

        self.se4 = SEModule(output_depth) if se else nn.Sequential()

        self.output_depth = output_depth

    def forward(self, x):
        y = self.pw(x)
        if self.shuffle_type == "mid":
            y = self.shuffle(y)
        if self.upscale is not None:
            y = self.upscale(y)
        y = self.dw(y)
        y = self.pwl(y)
        if self.use_res_connect:
            y += x
        y = self.se4(y)
        return y


class SEModule(nn.Module):
    reduction = 4

    def __init__(self, C):
        super(SEModule, self).__init__()
        mid = max(C // self.reduction, 8)
        conv1 = Conv1d(C, mid, 1, 1, 0)
        conv2 = Conv1d(mid, C, 1, 1, 0)

        self.op = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), conv1, nn.ReLU(inplace=True), conv2, nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.op(x)

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N, C, L] -> [N, g, C/g, L] -> [N, C/g, g, L] -> [N, C, L]"""
        N, C, L = x.size()
        g = self.groups
        assert C % g == 0, f"Incompatible group size {g} for input channel {C}"
        return (
            x.view(N, g, C // g, L)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(N, C, L)
        )

class Upsample(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=None):
        super(Upsample, self).__init__()
        self.scale = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return interpolate(
            x, scale_factor=self.scale, mode=self.mode, align_corners=self.align_corners
        )



def _get_divisible_by(num, divisible_by, min_val):
    ret = int(num)
    if divisible_by > 0 and num % divisible_by != 0:
        ret = int((_py2_round(num / divisible_by) or min_val) * divisible_by)
    return ret


def _get_upsample_op(stride):
    assert (
        stride in [1, 2, 4, 5]
        or stride in [-1, -2, -4, -5]
        or (isinstance(stride, tuple) and all(x in [-1, -2, -4, -5] for x in stride))
    )

    scales = stride
    ret = None
    if isinstance(stride, tuple) or stride < 0:
        scales = [-x for x in stride] if isinstance(stride, tuple) else -stride
        stride = 1
        ret = Upsample(scale_factor=scales, mode="nearest", align_corners=None)

    return ret, stride


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    if input.numel() > 0:
        return torch.nn.functional.interpolate(
            input, size, scale_factor, mode, align_corners
        )

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        if size is not None and scale_factor is not None:
            raise ValueError("only one of size or scale_factor should be defined")
        if (
            scale_factor is not None
            and isinstance(scale_factor, tuple)
            and len(scale_factor) != dim
        ):
            raise ValueError(
                f"scale_factor shape must match input shape. "
                f"Input is {dim}D, scale_factor size is {len(scale_factor)}"
            )

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        return [
            int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)
        ]

    output_shape = tuple(_output_size(1))
    output_shape = input.shape[:-1] + output_shape
    return _NewEmptyTensorOp.apply(input, output_shape)


def _py2_round(x):
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


# from thop import profile
# model = Graphormer(1,128,1,200)
# dummy_input = torch.randn(1, 128, 1,200)
# flops, params = profile(model, (dummy_input))
# print('FLOPs: ', flops, 'params: ', params)
# print('FLOPs: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
# FLOPs:  36044800.0 params:  282200.0
# FLOPs: 36.04 M, params: 0.28 M

# from thop import profile
# model = Trans(64,128,1,200)
# dummy_input = torch.randn(1, 1, 64,200)
# flops, params = profile(model, (dummy_input))
# print('FLOPs: ', flops, 'params: ', params)
# print('FLOPs: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
# FLOPs:  52531200.0 params:  822248.0
# FLOPs: 52.53 M, params: 0.82 M

# from thop import profile
# model = Trans(128,128,1,200)
# dummy_input = torch.randn(1, 1, 128,200)
# flops, params = profile(model, (dummy_input))
# print('FLOPs: ', flops, 'params: ', params)
# print('FLOPs: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
# FLOPs:  105062400.0 params:  822248.0
# FLOPs: 105.06 M, params: 0.82 M


# from thop import profile
# model = GSAU(128,128,1,200)
# dummy_input = torch.randn(1, 1, 128,200)
# flops, params = profile(model, (dummy_input))
# print('FLOPs: ', flops, 'params: ', params)
# print('FLOPs: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
# FLOPs:  10112000.0 params:  101760.0
# FLOPs: 10.11 M, params: 0.10 M


# from thop import profile
# model = IRFBlock(128,128,1,1)
# dummy_input = torch.randn(1,1,128, 200)
# flops, params = profile(model, (dummy_input))
# print('FLOPs: ', flops, 'params: ', params)
# print('FLOPs: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))


# n_feats = 3
# block = GSAU(n_feats,3,1,64)
# input = torch.rand(1, 3, 64)
# output = block(input)
# print(input.size())
# print(output.size())