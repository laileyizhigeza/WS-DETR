import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from timm.models.layers import DropPath, to_2tuple

def get_conv2d(in_channels,
               out_channels,
               kernel_size,
               stride,
               padding,
               dilation,
               groups,
               bias,
               attempt_use_lk_impl=True):

    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (kernel_size[0] // 2, kernel_size[1] // 2)

    if attempt_use_lk_impl and need_large_impl:
        print('---------------- trying to import iGEMM implementation for large-kernel conv')
        try:
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
            print('---------------- found iGEMM implementation ')
        except:
            DepthWiseConv2dImplicitGEMM = None
            print('---------------- found no iGEMM. use original conv. follow https://github.com/AILab-CVC/UniRepLKNet to install it.')
        if DepthWiseConv2dImplicitGEMM is not None and need_large_impl and in_channels == out_channels \
                and out_channels == groups and stride == 1 and dilation == 1:
            print(f'===== iGEMM Efficient Conv Impl, channels {in_channels}, kernel size {kernel_size} =====')
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)

    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     groups=groups,
                     bias=bias)


def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)


def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (conv_bias - bn.running_mean) * bn.weight / std

def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:,i:i+1,:,:], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)

def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel

class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 19:
            self.kernel_sizes = [5, 7, 9, 9, 3, 3, 3]
            self.dilates = [1, 1, 1, 2, 4, 5, 7]
        elif kernel_size == 17:
            self.kernel_sizes = [5, 7, 9, 3, 3, 3]
            self.dilates = [1, 1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 7, 5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 7, 5, 3, 3]
            self.dilates = [1, 1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))


def im2cswin(x, h_sp, w_sp, num_heads):
    b, c, h, w = x.shape
    num_heads = num_heads
    head_dim = c // num_heads
    # b, h * w, c ==> b, h // h_sp, w // w_sp, h_sp, w_sp, c
    x = x.view(b, c, h // h_sp, h_sp, w // w_sp, w_sp).permute(0, 2, 4, 3, 5, 1).contiguous()
    # b, h // h_sp, w // w_sp, h_sp, w_sp, c ==> -1, num_heads, h_sp * w_sp, head_dim
    x = x.view(-1, h_sp * w_sp, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    return x

def cswin2im(x, h_sp, w_sp, b, h, w):
    _, num_heads, window_nums, head_dim = x.shape
    c = num_heads * head_dim
    # -1, num_heads, h_sp * w_sp, head_dim ==> b, h // h_sp, w // w_sp, h_sp, w_sp, c
    x = x.transpose(1, 2).contiguous().view(b, h // h_sp, w // w_sp, h_sp, w_sp, c)
    # b, h // h_sp, w // w_sp, h_sp, w_sp, c ==> b, c, h, w
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(b, c, h, w)
    return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value,
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):

        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])

        return x

class ECA(nn.Module):
    def __init__(self, k=5):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous().unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class LayerNorm2d(nn.Module):
    def __init__(self, c_channel):
        super(LayerNorm2d, self).__init__()
        self.ln = nn.LayerNorm(c_channel)

    def forward(self, x):
        x = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x

class EAA(nn.Module):
    def __init__(self, dim, idx, split_size=8, num_heads=8):
        super().__init__()
        assert idx == 0 or 1, f"idx should be 0 or 1."
        self.idx = idx
        self.num_heads = num_heads
        self.split_size = split_size
        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.softmax = nn.Softmax(dim=-1)

    def im2cswin(self, x, h_sp, w_sp):
        b, c, h, w = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads
        # b, h * w, c ==> b, h // h_sp, w // w_sp, h_sp, w_sp, c
        x = x.view(b, c, h // h_sp, h_sp, w // w_sp, w_sp).permute(0, 2, 4, 3, 5, 1).contiguous()
        # b, h // h_sp, w // w_sp, h_sp, w_sp, c ==> -1, num_heads, h_sp * w_sp, head_dim
        x = x.view(-1, h_sp * w_sp, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, q, k, v):
        b, c, h, w = q.shape
        h_sp, w_sp = h, w
        if self.idx == 0:
            w_sp = self.split_size
        elif self.idx == 1:
            h_sp = self.split_size
        cpe = self.cpe(v)
        # b, c, h, w ==> -1, num_heads, h_sp * w_sp, head_dim
        q = self.im2cswin(q, h_sp, w_sp)
        k = self.im2cswin(k, h_sp, w_sp)
        v = self.im2cswin(v, h_sp, w_sp)
        # (h_sp * w_sp, head_dim) @ (head_dim, h_sp * w_sp) ==> (h_sp * w_sp, h_sp * w_sp)
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)
        # (h_sp * w_sp, h_sp * w_sp) @ (h_sp * w_sp, head_dim) ==> (h_sp * w_sp, head_dim)
        v = attn @ v
        # -1, num_heads, h_sp * w_sp, head_dim ==> -1, h_sp, w_sp, c
        v = v.transpose(1, 2).contiguous().view(b, h // h_sp, w // w_sp, h_sp, w_sp, c)
        # b, h // h_sp, w // w_sp, h_sp, w_sp, c ==> b, c, h, w
        v = v.permute(0, 5, 1, 3, 2, 4).contiguous().view(b, c, h, w)
        v = v + cpe
        return v

class ResDWConv(nn.Conv2d):
    '''
    Depthwise conv with residual connection
    '''
    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)

    def forward(self, x):
        x = x + super().forward(x)
        return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, C, H, W)
    """
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))


    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x

class HWCross(nn.Module):
    def __init__(self, c_in, c_out, split_size, num_heads=8, mlp_ratio=4, deploy=False, use_gemm=False, ls_init_value=None):
        super(HWCross, self).__init__()
        self.split_size = split_size
        self.num_heads = num_heads
        self.c_in = c_in
        c_middle = c_in // 2
        
        # 空间信息指导
        self.cpe = nn.Conv2d(2, 1, kernel_size=7, padding=3, padding_mode='reflect', bias=False)

        # 位置编码矩阵
        self.pos_embed = nn.Conv2d(c_in, c_in, kernel_size=7, stride=1, padding=3, bias=False, groups=c_in)
        # 第一层LayerNorm
        self.ln1 = LayerNorm2d(c_in)
        
        # 大核卷积局部信息提取
        self.lepe = nn.Sequential(
            DilatedReparamBlock(c_in, kernel_size=7, deploy=deploy, use_sync_bn=False, attempt_use_lk_impl=use_gemm),
            nn.BatchNorm2d(c_in),
        )

        # 得到Q矩阵
        self.q_proj = nn.Conv2d(c_middle, c_middle, kernel_size=1, bias=False)
        # 得到K矩阵
        self.k_proj = nn.Conv2d(c_middle, 2 * c_middle, kernel_size=1, bias=False)
        # 得到V矩阵
        self.v_proj = nn.Conv2d(c_middle, 2 * c_middle, kernel_size=1, bias=False)
        
        # EAA模块
        self.eaa_x = EAA(c_middle, idx=0, split_size=self.split_size, num_heads=num_heads)
        self.eaa_y = EAA(c_middle, idx=1, split_size=self.split_size, num_heads=num_heads)
        
        
        self.ls1 = LayerScale(c_in, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()

        # 第二层LayerNorm
        self.ln2 = LayerNorm2d(c_in)
        self.mlp = nn.Sequential(
            nn.Conv2d(c_in, c_in * mlp_ratio, kernel_size=1),
            nn.GELU(),
            ResDWConv(c_in * mlp_ratio, kernel_size=3),
            GRN(c_in * mlp_ratio, use_bias=True),
            nn.Conv2d(c_in * mlp_ratio, c_out, kernel_size=1),
        )
    
    def reparm(self):
        for m in self.modules():
            if isinstance(m, DilatedReparamBlock):
                m.merge_dilated_branches()
    def forward(self, x):
        
        # 添加条件位置编码
        x = x + self.pos_embed(x)
        identity = x
        
        # 通过LayerNorm归一化
        x = self.ln1(x)
        
        # 通过大核卷积进行局部特征的提取
        lepe = self.lepe(x)
        
        # 将特征图划分为两部分 分别是 具有多语义信息的x_1 和 具有更多全局信息的 x_2
        x_1, x_2 = x.chunk(2, dim=1)
        
        # 特征图x_2 通过空间指导x_1
        x_avg = torch.mean(x_1, dim=1, keepdim=True)
        x_max, _ = torch.max(x_1, dim=1, keepdim=True)
        x_spatial = torch.concat([x_avg, x_max], dim=1)
        sattn = self.cpe(x_spatial)
        
        x_2 = x_2 + sattn
        
        # 得到Q矩阵
        q = self.q_proj(x_1)
        # 得到K矩阵
        k = self.k_proj(x_2)
        # 得到V矩阵
        v = self.v_proj(x_2)
        
        qkv = [q]
        for i in range(2):
            qkv.append(k[:, i::2, :, :])
            qkv.append(v[:, i::2, :, :])
        x = torch.concat([self.eaa_x(qkv[0], qkv[1], qkv[3]), self.eaa_y(qkv[0], qkv[2], qkv[4])], dim=1)
        # x = self.proj_eaa(x)
        
        # 将EAA融合特征图与大核卷积特征图结果相加
        x = x + lepe
        
        # 与残差连接相加
        x = self.ls1(x) + identity
        
        x = self.mlp(self.ln2(x))
        return x

class HWCrossv2(nn.Module):
    def __init__(self, split_direction='H', num_panes=4):
        """
        Args:
            split_direction (str): 'H' 表示沿高度切分，'W' 表示沿宽度切分
            num_panes (int): 切分成多少个窗格
        """
        super(HWCrossv2, self).__init__()
        self.split_direction = split_direction
        self.num_panes = num_panes
        
        # 示例：每个 pane 使用一个简单的卷积块作为处理模块
        self.pane_conv = nn.Conv2d(
            in_channels=1,  # 每个 pane 的通道数（假设C被平均分配）
            out_channels=1,
            kernel_size=3,
            padding=1
        )

    def _split_feature_map(self, x):
        B, C, H, W = x.shape
        if self.split_direction == 'H':
            # Split along height (H)
            assert H % self.num_panes == 0, f"Height {H} 不可被 {self.num_panes} 整除"
            h_per_pane = H // self.num_panes
            panes = [x[:, :, i*h_per_pane:(i+1)*h_per_pane, :] for i in range(self.num_panes)]
        elif self.split_direction == 'W':
            # Split along width (W)
            assert W % self.num_panes == 0, f"Width {W} 不可被 {self.num_panes} 整除"
            w_per_pane = W // self.num_panes
            panes = [x[:, :, :, i*w_per_pane:(i+1)*w_per_pane] for i in range(self.num_panes)]
        else:
            raise ValueError("split_direction 必须为 'H' 或 'W'")
        
        return panes

    def _process_each_pane(self, panes):
        """
        对每个 pane 做处理，例如卷积、注意力等。
        这里仅作为示例，使用相同的卷积层。
        """
        processed = []
        for pane in panes:
            # 示例：每个 pane 独立处理
            # 注意：pane shape: (B, C, h_p, w_p)
            processed_pane = self.pane_conv(pane)
            processed.append(processed_pane)
        return processed

    def forward(self, x):
        panes = self._split_feature_map(x)
        processed_panes = self._process_each_pane(panes)
        # 再拼接回去
        if self.split_direction == 'H':
            output = torch.cat(processed_panes, dim=2)  # 沿高度拼接
        elif self.split_direction == 'W':
            output = torch.cat(processed_panes, dim=3)  # 沿宽度拼接
        return output

if __name__ == '__main__':
    x = torch.randn(2, 512, 64, 64).cuda()
    model = HWCross(c_in=512, c_out=156, split_size=4, num_heads=8, mlp_ratio=2).cuda()
    from torchinfo import summary
    # summary(model, input_size=(2, 512, 64, 64), col_names=["input_size", "output_size", "num_params", "trainable"], depth=4)
    
    # print(model)
    model.cuda()
    model.eval()
    y = model(x)
    print(y.shape)

    # Reparametrize model, more details can be found at:
    # https://github.com/AILab-CVC/UniRepLKNet/tree/main
    model.reparm()
    z = model(x)

    # Showing difference between original and reparametrized model
    print((z - y).abs().sum() / y.abs().sum())
    
    
    output = model(x)
    print(output.shape)