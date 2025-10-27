import sys
from pathlib import Path
# 获取当前文件的绝对路径，并向上找 3 层到 "ultralytics" 的父目录
sys.path.append(str(Path(__file__).resolve().parents[3]))

# 然后使用绝对导入
from ultralytics.nn.extra_modules.wtconv2d import create_wavelet_filter, wavelet_transform_init, inverse_wavelet_transform_init
from ultralytics.nn.modules.block import BasicBlock

from SS2Dv2 import SS2D
from CSDA import ECA
import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, dims, eps=1e-5):
        super(LayerNorm2d, self).__init__()
        self.layernorm = nn.LayerNorm(dims, eps=eps)

    def forward(self, x):
        x = self.layernorm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)

class WTMambav2(nn.Module):
    def __init__(self, in_channels, wt_type='db1'):
        super(WTMambav2, self).__init__()

        self.in_channels = in_channels 

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        
        self.wt_function = wavelet_transform_init(self.wt_filter)
        self.iwt_function = inverse_wavelet_transform_init(self.iwt_filter)

        self.wt_low_scale = _ScaleModule([1, in_channels, 1, 1], init_scale=0.1)
        self.wt_high_scale = _ScaleModule([1, in_channels * 3, 1, 1], init_scale=0.1)
        
        # 高频信息通过SS2D进行信息的摘取
        self.ss2d_high = nn.Sequential(
            SS2D((in_channels * 3) // 2, d_state=16, d_conv=3, expand=2),
            LayerNorm2d(3*in_channels)
            )
        
        # 低频信息通过PKIConv
        self.ss2d_low = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.SiLU()
        )
        

        self.channel_attention = ECA(k=5)
    def forward(self, x):
        b, c, h, w = x.shape
        wt_x = x

        # 进行小波变换
        wt_x = self.wt_function(wt_x)

        # 获得低频信息和高频信息 并进行通道方向的变换
        wt_x_low = self.wt_low_scale(wt_x[:, :, 0, :, :])
        wt_x_high = self.wt_high_scale(wt_x[:, :, 1:, :, :].reshape(wt_x.shape[0], wt_x.shape[1] * 3, wt_x.shape[3], wt_x.shape[4]).contiguous())

        # 低频信息通过 1x1Conv + SiLU 进行处理
        wt_x_low = self.ss2d_low(wt_x_low)
        # 高频信息通过SS2D进行处理
        wt_x_high = self.ss2d_high(wt_x_high)
        
        # 低频信息与高频信息进行相乘
        wt_mamba = wt_x_low.repeat(1, 3, 1, 1) * wt_x_high
        
        wt_x = torch.cat([wt_x_low, wt_mamba], dim=1).unsqueeze(2).reshape(wt_x.shape[0], wt_x.shape[1], 4, wt_x.shape[3], wt_x.shape[4])
        wt_x = self.iwt_function(wt_x)
        x = self.channel_attention(wt_x) + x

        return x

class WTMambaBlock(BasicBlock):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        self.wtmamba = WTMambav2(ch_out)

    def forward(self, x):
        x = super().forward(x)
        x = self.wtmamba(x)
        return x


if __name__ == '__main__':
    from torchsummary import summary
    model = WTMambav2(128).cuda()
    summary(model, input_size=(128, 224, 224))
    x = torch.randn(1, 128, 224, 224).cuda()
    y = model(x)
    print(y.shape)