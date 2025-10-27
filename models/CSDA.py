import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ESA(nn.Module):
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x_se = x - self.gap(x).expand_as(x)
        x_max, _ = torch.max(x_se, dim=1, keepdim=True)
        x_min, _ = torch.min(x_se, dim=1, keepdim=True)
        x = x * self.sigmoid(x_max) * self.alpha + x * self.sigmoid(x_min) * (1 - self.alpha)
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

# split_size: 图像分割窗口大小(轴向)
class CSDA(nn.Module):
    def __init__(self, dim, num_heads=8, split_size=4, mlp_ratio = 4):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."
        self.eca = ECA()
        self.esa = ESA()
        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.GroupNorm(1, dim)
        self.act_proj = nn.Conv2d(dim, dim, 1)
        self.act = nn.ReLU(inplace=True)
        self.in_proj = nn.Conv2d(dim, dim, 1)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.qkv = nn.Conv2d(dim, 5 * dim, 1)
        self.eaa_x = EAA(dim, idx=0, split_size=split_size, num_heads=num_heads)
        self.eaa_y = EAA(dim, idx=1, split_size=split_size, num_heads=num_heads)
        self.out_proj = nn.Conv2d(dim, dim, 1)
        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = nn.GroupNorm(1, dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(dim * mlp_ratio, dim, 1)
        )

    def forward(self, x):
        # ECA + ESA
        x = self.eca(x) + self.esa(x)
        x = x + self.cpe1(x)
        shortcut = x
        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.act(self.dwc(self.in_proj(x)))
        # EDA
        x = self.qkv(x)
        qkv = []
        for i in range(5):
            qkv.append(x[:, i::5, :, :])
        x = self.eaa_x(qkv[0], qkv[1], qkv[2]) + self.eaa_y(qkv[3], qkv[4], qkv[2])
        x = self.out_proj(x * act_res)
        x = x + shortcut
        x = x + self.cpe2(x)
        # FFN
        x = self.mlp(self.norm2(x)) + x
        return x
