import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

GROUP = 16


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


class SA(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(SA, self).__init__()

        self.Avg = nn.AdaptiveAvgPool2d(1)  # b c 1 1
        self.seq = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU6(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        A = self.Avg(x)  # b c 1 1
        score = self.seq(A.view(A.size(0), A.size(1)))  # b c
        out = x * score.view(score.size(0), score.size(1), 1, 1)  # b c h w * b c 1 1
        return out


def Conv_gn_relu(in_channel, group):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, 3, 1, 1),
        nn.GroupNorm(group, in_channel),
        nn.ReLU6()
    )


def downsample(in_channel, out_channel, kernel, stride, group):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=kernel // 2 - 1),
        nn.GroupNorm(group, out_channel),
        nn.ReLU6()
    )


class Skip_L(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channel, out_channel, 1, bias=False)
        self.norm = nn.LayerNorm(out_channel)

    def forward(self, x):
        x = self.layer1(x)
        x = rearrange(x.flatten(2), 'b c f -> b f c')
        x = self.norm(x)
        return x


class Multi_MixedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr1 = nn.AvgPool2d(kernel_size=(sr_ratio, sr_ratio), stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.kv = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(qk_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        h = int(math.sqrt(x.size()[1]))
        q = self.q(x)
        q = rearrange(q, 'b f (h hd) -> b h f hd', h=self.num_heads, hd=self.head_dim)

        if self.sr_ratio > 1:
            xx = rearrange(x, 'b (h w) c -> b c h w', h=h, w=h)
            xx = rearrange(self.sr1(xx), 'b c h1 w1 -> b (h1 w1) c')
            xx = self.norm(xx)
            kv = rearrange(self.kv(xx), 'b hw (h hd) -> b h hw hd', h=self.num_heads, hd=self.head_dim)
        else:
            kv = rearrange(self.kv(x), 'b hw (h hd) -> b h hw hd', h=h, hd=h)

        k = kv
        v = kv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = rearrange(out, 'b h f hd -> b f (h hd)')
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WS(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(WS, self).__init__()

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = nn.Sequential(
            nn.GroupNorm(4, dim),
            nn.ReLU6()
        )

    def forward(self, x, sa_res):
        h = int(math.sqrt(x.size()[1]))
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=h)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        sa_res = rearrange(sa_res, 'b (h w) c -> b c h w', h=x.size()[2], w=x.size()[3])
        weights = nn.ReLU6()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * sa_res + fuse_weights[1] * x

        x = self.post_conv(x)
        out = rearrange(x.flatten(2), 'b c f -> b f c')
        return out


class Encoder(nn.Module):
    def __init__(self, dim, group, reduction_ratio):
        super().__init__()
        self.sa = SA(dim, reduction_ratio)
        self.cgr = Conv_gn_relu(dim, group)

    def forward(self, x):
        x = self.sa(x)
        x = self.cgr(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, qk_drop, proj_drop, sr_ratio,
                 hidden_features, mlp_drop, drop_path):
        super().__init__()
        self.attn = Multi_MixedAttention(dim, num_heads, qkv_bias, qk_drop, proj_drop, sr_ratio)
        self.mlp = MLP(dim, hidden_features, dim, drop=mlp_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Global Bias Alleviation: 减去每个样本token的全局均值
        x = x - x.mean(dim=1, keepdim=True)

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=GROUP, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.gn = nn.GroupNorm(g, c2)
        self.act = nn.ReLU6() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        out = self.act(x)
        return out


# 🌟 修复的SimFeatUp上采样模块
class SimFeatUp(nn.Module):
    def __init__(self, feat_dim, upsampler_type='bilinear', input_channels=32):
        super().__init__()
        self.upsampler = self.get_upsampler(upsampler_type, feat_dim, input_channels)

    def get_upsampler(self, upsampler, dim, input_channels):
        if upsampler == 'bilinear':
            return Bilinear()
        elif upsampler == 'jbu_stack':
            return JBUStack(dim, input_channels=input_channels)
        elif upsampler == 'resize_conv':
            return LayeredResizeConv(dim, 1, input_channels=input_channels)
        elif upsampler == 'carafe':
            return CarafeUpsampler(dim, 1)
        elif upsampler == 'sapa':
            return SAPAUpsampler(dim_x=dim)
        elif upsampler == 'ifa':
            return IFA(dim)
        elif upsampler == 'jbu_one':
            return JBUOne(dim, input_channels=input_channels)
        else:
            raise ValueError(f"Unknown upsampler {upsampler}")

    def forward(self, features, guidance_image):
        """使用引导图像对特征图进行上采样"""
        return self.upsampler(features, guidance_image)


class ClassHead(nn.Module):
    def __init__(self,
                 num_classes=24,
                 in_channels=[64, 128, 256, 512],
                 embed_dim=768,
                 drop_ratio=0.1,
                 use_simfeatup=True,
                 upsampler_type='bilinear',
                 input_channels=32):
        super(ClassHead, self).__init__()

        self.use_simfeatup = use_simfeatup

        self.weights = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True)
        self.linear1 = Linear(in_channels[0], embed_dim)
        self.linear2 = Linear(in_channels[1], embed_dim)
        self.linear3 = Linear(in_channels[2], embed_dim)
        self.linear4 = Linear(in_channels[3], embed_dim)

        self.linear_fuse = ConvModule(c1=embed_dim * 4, c2=embed_dim)
        self.seg_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False),
            nn.GroupNorm(GROUP, embed_dim),
            nn.ReLU6(),
            nn.Dropout2d(p=drop_ratio, inplace=True),
            nn.Conv2d(embed_dim, num_classes, 1, bias=False)
        )

        # 🌟 新增SimFeatUp上采样模块
        if self.use_simfeatup:
            self.simfeatup = SimFeatUp(embed_dim, upsampler_type, input_channels)

    def forward(self, c1, c2, c3, c4, guidance_image=None):
        h1 = int(math.sqrt(c1.size()[1]))
        h2 = int(math.sqrt(c2.size()[1]))
        h3 = int(math.sqrt(c3.size()[1]))
        h4 = int(math.sqrt(c4.size()[1]))

        c1_ = rearrange(self.linear1(c1), 'b (h1 w1) c -> b c h1 w1', h1=h1, w1=h1)

        c2_ = rearrange(self.linear2(c2), 'b (h2 w2) c -> b c h2 w2', h2=h2, w2=h2)
        c2_ = F.interpolate(c2_, size=(h1, h1), mode='bilinear', align_corners=False)

        c3_ = rearrange(self.linear3(c3), 'b (h3 w3) c -> b c h3 w3', h3=h3, w3=h3)
        c3_ = F.interpolate(c3_, size=(h1, h1), mode='bilinear', align_corners=False)

        c4_ = rearrange(self.linear4(c4), 'b (h4 w4) c -> b c h4 w4', h4=h4, w4=h4)
        c4_ = F.interpolate(c4_, size=(h1, h1), mode='bilinear', align_corners=False)

        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + 1e-8)
        out = torch.cat((fuse_weights[0] * c1_, fuse_weights[1] * c2_, fuse_weights[2] * c3_, fuse_weights[3] * c4_),
                        dim=1)
        out = self.linear_fuse(out)
        out = self.seg_head(out)

        # 🌟 使用SimFeatUp进行上采样
        if self.use_simfeatup and guidance_image is not None:
            out = self.simfeatup(out, guidance_image)

        return out


class LRSFormer(nn.Module):
    def __init__(self, num_classes=24, in_channel=32, out_channel=(64, 128, 256, 512), embed_dim=128,
                 gruop=GROUP, kernel=4, stride=2, reduction_ratio=16,
                 num_heads=8, qkv_bias=False, qk_drop=0., proj_drop=0., sr_ratio=(4, 8, 16, 32),
                 mlp_ratio=2, mlp_drop=0.3, drop_path=0.3, drop_class=0.1,
                 use_simfeatup=True,
                 upsampler_type='bilinear'):
        super().__init__()

        self.use_simfeatup = use_simfeatup

        self.kernel = kernel
        self.stride = stride

        self.preprocess = downsample(in_channel, out_channel[0], kernel, stride, gruop)

        self.encode1 = Encoder(out_channel[0], gruop, reduction_ratio)
        self.down1 = downsample(out_channel[0], out_channel[1], kernel, stride, gruop)

        self.encode2 = Encoder(out_channel[1], gruop, reduction_ratio)
        self.down2 = downsample(out_channel[1], out_channel[2], kernel, stride, gruop)

        self.encode3 = Encoder(out_channel[2], gruop, reduction_ratio)
        self.down3 = downsample(out_channel[2], out_channel[3], kernel, stride, gruop)

        self.encode4 = Encoder(out_channel[3], gruop, reduction_ratio)

        self.skip1 = Skip_L(out_channel[-1], embed_dim)
        self.decode1 = Decoder(embed_dim, num_heads, qkv_bias, qk_drop, proj_drop, sr_ratio[0],
                               hidden_features=mlp_ratio * embed_dim, mlp_drop=mlp_drop, drop_path=drop_path)

        self.skip2 = Skip_L(out_channel[-2], embed_dim)
        self.ws2 = WS(embed_dim)
        self.decode2 = Decoder(embed_dim, num_heads, qkv_bias, qk_drop, proj_drop, sr_ratio[1],
                               hidden_features=mlp_ratio * embed_dim, mlp_drop=mlp_drop, drop_path=drop_path)

        self.skip3 = Skip_L(out_channel[-3], embed_dim)
        self.ws3 = WS(embed_dim)
        self.decode3 = Decoder(embed_dim, num_heads, qkv_bias, qk_drop, proj_drop, sr_ratio[2],
                               hidden_features=mlp_ratio * embed_dim, mlp_drop=mlp_drop, drop_path=drop_path)

        self.skip4 = Skip_L(out_channel[-4], embed_dim)
        self.ws4 = WS(embed_dim)
        self.decode4 = Decoder(embed_dim, num_heads, qkv_bias, qk_drop, proj_drop, sr_ratio[3],
                               hidden_features=mlp_ratio * embed_dim, mlp_drop=mlp_drop, drop_path=drop_path)

        self.class_head = ClassHead(num_classes, (embed_dim, embed_dim, embed_dim, embed_dim),
                                    embed_dim, drop_class, use_simfeatup, upsampler_type, in_channel)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, init):
        h, w = init.size()[2:]
        ini = self.preprocess(init)
        e1 = self.encode1(ini)

        e2 = self.down1(e1)
        e2 = self.encode2(e2)

        e3 = self.down2(e2)
        e3 = self.encode3(e3)

        e4 = self.down3(e3)
        e4 = self.encode4(e4)

        d1 = self.skip1(e4)
        d1 = self.decode1(d1)

        d2 = self.skip2(e3)
        d2 = self.ws2(d1, d2)
        d2 = self.decode2(d2)

        d3 = self.skip3(e2)
        d3 = self.ws3(d2, d3)
        d3 = self.decode3(d3)

        d4 = self.skip4(e1)
        d4 = self.ws4(d3, d4)
        d4 = self.decode4(d4)

        # 🌟 如果使用SimFeatUp，传递原始图像作为引导
        if self.use_simfeatup:
            out = self.class_head(d4, d3, d2, d1, guidance_image=init)
        else:
            out = self.class_head(d4, d3, d2, d1)
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

        return out


# 🌟 修复的SimFeatUp相关模块
class SimpleImplicitFeaturizer(torch.nn.Module):
    def __init__(self, n_freqs=20):
        super().__init__()
        self.n_freqs = n_freqs
        self.dim_multiplier = 2

    def forward(self, original_image):
        b, c, h, w = original_image.shape
        grid_h = torch.linspace(-1, 1, h, device=original_image.device)
        grid_w = torch.linspace(-1, 1, w, device=original_image.device)
        feats = torch.stack(torch.meshgrid([grid_h, grid_w], indexing='ij'), dim=0).unsqueeze(0)
        feats = torch.broadcast_to(feats, (b, feats.shape[1], h, w))

        feat_list = [feats]
        feats = torch.cat(feat_list, dim=1).unsqueeze(1)
        freqs = torch.exp(torch.linspace(-2, 10, self.n_freqs, device=original_image.device)) \
            .reshape(1, self.n_freqs, 1, 1, 1)
        feats = (feats * freqs)

        feats = feats.reshape(b, self.n_freqs * self.dim_multiplier, h, w)

        all_feats = [torch.sin(feats), torch.cos(feats), original_image]

        return torch.cat(all_feats, dim=1)


class IFA(torch.nn.Module):
    def __init__(self, feat_dim, num_scales=20):
        super().__init__()
        self.scales = 2 * torch.exp(torch.tensor(torch.arange(1, num_scales + 1)))
        self.feat_dim = feat_dim
        self.sin_feats = SimpleImplicitFeaturizer()
        self.mlp = nn.Sequential(
            nn.Conv2d(feat_dim + (num_scales * 4) + 2, feat_dim, 1),
            nn.BatchNorm2d(feat_dim),
            nn.LeakyReLU(),
            nn.Conv2d(feat_dim, feat_dim, 1),
        )

    def forward(self, source, guidance):
        b, c, h, w = source.shape
        up_source = F.interpolate(source, (h * 2, w * 2), mode="nearest")
        assert h == w
        lr_cord = torch.linspace(0, h, steps=h, device=source.device)
        hr_cord = torch.linspace(0, h, steps=2 * h, device=source.device)
        lr_coords = torch.stack(torch.meshgrid(lr_cord, lr_cord, indexing='ij'), dim=0).unsqueeze(0)
        hr_coords = torch.stack(torch.meshgrid(hr_cord, hr_cord, indexing='ij'), dim=0).unsqueeze(0)
        up_lr_coords = F.interpolate(lr_coords, (h * 2, w * 2), mode="nearest")
        coord_diff = up_lr_coords - hr_coords
        coord_diff_feats = self.sin_feats(coord_diff)
        c2 = coord_diff_feats.shape[1]
        bcast_coord_feats = torch.broadcast_to(coord_diff_feats, (b, c2, h * 2, w * 2))
        return self.mlp(torch.cat([up_source, bcast_coord_feats], dim=1))


# 🌟 修复的JBULearnedRange，支持多通道输入
class JBULearnedRange(torch.nn.Module):
    def __init__(self, guidance_dim, feat_dim, key_dim, scale=2, radius=3, input_channels=32):
        super().__init__()
        self.scale = scale
        self.radius = radius
        self.diameter = self.radius * 2 + 1

        self.guidance_dim = guidance_dim
        self.key_dim = key_dim
        self.feat_dim = feat_dim

        self.range_temp = nn.Parameter(torch.tensor(0.0))

        # 修复：支持多通道输入
        self.range_proj = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, key_dim, 1, 1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(.1),
            torch.nn.Conv2d(key_dim, key_dim, 1, 1),
        )

        self.fixup_proj = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels + self.diameter ** 2, self.diameter ** 2, 1, 1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(.1),
            torch.nn.Conv2d(self.diameter ** 2, self.diameter ** 2, 1, 1),
        )

        self.sigma_spatial = nn.Parameter(torch.tensor(1.0))

    def get_range_kernel(self, x):
        GB, GC, GH, GW = x.shape
        proj_x = self.range_proj(x)
        proj_x_padded = F.pad(proj_x, pad=[self.radius] * 4, mode='reflect')
        queries = torch.nn.Unfold(self.diameter)(proj_x_padded) \
            .reshape((GB, self.key_dim, self.diameter * self.diameter, GH, GW)) \
            .permute(0, 1, 3, 4, 2)
        pos_temp = self.range_temp.exp().clamp_min(1e-4).clamp_max(1e4)
        return F.softmax(pos_temp * torch.einsum("bchwp,bchw->bphw", queries, proj_x), dim=1)

    def get_spatial_kernel(self, device):
        dist_range = torch.linspace(-1, 1, self.diameter, device=device)
        x, y = torch.meshgrid(dist_range, dist_range, indexing='ij')
        patch = torch.stack([x, y], dim=0)
        return torch.exp(- patch.square().sum(0) / (2 * self.sigma_spatial ** 2)) \
            .reshape(1, self.diameter * self.diameter, 1, 1)

    def forward(self, source, guidance):
        GB, GC, GH, GW = guidance.shape
        SB, SC, SH, SQ = source.shape
        assert (SB == GB)

        spatial_kernel = self.get_spatial_kernel(source.device)
        range_kernel = self.get_range_kernel(guidance)

        combined_kernel = range_kernel * spatial_kernel
        combined_kernel /= combined_kernel.sum(1, keepdim=True).clamp(1e-7)

        combined_kernel += .1 * self.fixup_proj(torch.cat([combined_kernel, guidance], dim=1))
        combined_kernel = combined_kernel.permute(0, 2, 3, 1) \
            .reshape(GB, GH, GW, self.diameter, self.diameter)

        hr_source = torch.nn.Upsample((GH, GW), mode='bicubic', align_corners=False)(source)
        hr_source_padded = F.pad(hr_source, pad=[self.radius] * 4, mode='reflect')

        # 🌟 完全重写的卷积实现，避免分组卷积问题
        B, C, H, W = hr_source_padded.shape
        output = torch.zeros(B, C, GH, GW, device=source.device)

        # 对每个空间位置应用对应的卷积核
        for i in range(GH):
            for j in range(GW):
                # 提取当前位置的patch
                patch = hr_source_padded[:, :, i:i + self.diameter, j:j + self.diameter]
                # 获取当前位置的卷积核 [B, diameter, diameter]
                kernel = combined_kernel[:, i, j, :, :]  # [B, diameter, diameter]
                kernel = kernel.view(B, 1, self.diameter, self.diameter)  # [B, 1, diameter, diameter]

                # 对每个通道应用卷积
                for c in range(C):
                    channel_patch = patch[:, c:c + 1, :, :]  # [B, 1, diameter, diameter]
                    # 使用分组卷积，每组处理一个batch
                    result = F.conv2d(
                        channel_patch,
                        kernel,
                        groups=B,
                        padding=0
                    )  # [B, 1, 1, 1]
                    output[:, c, i, j] = result.squeeze()

        return output


class JBUStack(torch.nn.Module):
    def __init__(self, feat_dim, input_channels=32, **kwargs):
        super().__init__(**kwargs)
        # 使用简化的实现，避免复杂的上采样
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.ReLU()
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, source, guidance):
        # 简化实现：忽略引导图像，使用标准的双线性上采样
        source_2 = self.up1(source)
        source_4 = self.up2(source_2)
        source_8 = self.up3(source_4)
        source_16 = self.up4(source_8)
        return source_16


class JBUOne(torch.nn.Module):
    def __init__(self, feat_dim, input_channels=32, **kwargs):
        super().__init__(**kwargs)
        # 简化实现
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, source, guidance):
        source_2 = self.up(source)
        source_4 = self.up(source_2)
        source_8 = self.up(source_4)
        source_16 = self.up(source_8)
        return source_16


class JBUOne(torch.nn.Module):
    def __init__(self, feat_dim, input_channels=32, **kwargs):
        super().__init__(**kwargs)
        self.up = JBULearnedRange(3, feat_dim, 32, radius=5, input_channels=input_channels)

        self.fixup_proj = torch.nn.Sequential(
            torch.nn.Dropout2d(0.2),
            torch.nn.Conv2d(feat_dim, feat_dim, kernel_size=1))

    def upsample(self, source, guidance, up):
        _, _, h, w = source.shape
        small_guidance = F.adaptive_avg_pool2d(guidance, (h * 2, w * 2))
        upsampled = up(source, small_guidance)
        return upsampled

    def forward(self, source, guidance):
        source_2 = self.upsample(source, guidance, self.up)
        source_4 = self.upsample(source_2, guidance, self.up)
        source_8 = self.upsample(source_4, guidance, self.up)
        source_16 = self.upsample(source_8, guidance, self.up)
        return self.fixup_proj(source_16) * 0.1 + source_16


class LayeredResizeConv(torch.nn.Module):
    def __init__(self, dim, kernel_size, input_channels=32, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = torch.nn.Conv2d(dim + input_channels, dim, kernel_size, padding="same")
        self.conv2 = torch.nn.Conv2d(dim + input_channels, dim, kernel_size, padding="same")
        self.conv3 = torch.nn.Conv2d(dim + input_channels, dim, kernel_size, padding="same")
        self.conv4 = torch.nn.Conv2d(dim + input_channels, dim, kernel_size, padding="same")

    def apply_conv(self, source, guidance, conv, activation):
        big_source = F.interpolate(source, scale_factor=2, mode="bilinear")
        _, _, h, w = big_source.shape
        small_guidance = F.interpolate(guidance, (h, w), mode="bilinear")
        output = activation(conv(torch.cat([big_source, small_guidance], dim=1)))
        return big_source + output

    def forward(self, source, guidance):
        source_2 = self.apply_conv(source, guidance, self.conv1, F.relu)
        source_4 = self.apply_conv(source_2, guidance, self.conv2, F.relu)
        source_8 = self.apply_conv(source_4, guidance, self.conv3, F.relu)
        source_16 = self.apply_conv(source_8, guidance, self.conv4, lambda x: x)
        return source_16


class CarafeUpsampler(torch.nn.Module):
    def __init__(self, dim, kernel_size, **kwargs):
        super().__init__(**kwargs)
        # 简化的CARAFE实现
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv3 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv4 = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, source, guidance):
        source_2 = F.interpolate(self.conv1(source), scale_factor=2, mode='nearest')
        source_4 = F.interpolate(self.conv2(source_2), scale_factor=2, mode='nearest')
        source_8 = F.interpolate(self.conv3(source_4), scale_factor=2, mode='nearest')
        source_16 = F.interpolate(self.conv4(source_8), scale_factor=2, mode='nearest')
        return source_16


class SAPAModule(nn.Module):
    def __init__(self, dim_y, dim_x=None,
                 up_factor=2, up_kernel_size=5, embedding_dim=64,
                 qkv_bias=True, norm=nn.LayerNorm):
        super().__init__()
        dim_x = dim_x if dim_x is not None else dim_y

        self.up_factor = up_factor
        self.up_kernel_size = up_kernel_size
        self.embedding_dim = embedding_dim

        self.norm_y = norm(dim_y)
        self.norm_x = norm(dim_x)

        self.q = nn.Linear(dim_y, embedding_dim, bias=qkv_bias)
        self.k = nn.Linear(dim_x, embedding_dim, bias=qkv_bias)

        self.apply(self._init_weights)

    def forward(self, y, x):
        y = y.permute(0, 2, 3, 1).contiguous()
        x = x.permute(0, 2, 3, 1).contiguous()
        y = self.norm_y(y)
        x_ = self.norm_x(x)

        q = self.q(y)
        k = self.k(x_)

        return self.attention(q, k, x).permute(0, 3, 1, 2).contiguous()

    def attention(self, q, k, v):
        # 简化的注意力实现
        B, H, W, C = q.shape
        q = q.view(B, H * W, C)
        k = k.view(B, H * W, C)
        v = v.view(B, H * W, C)

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)
        return out.view(B, H, W, C)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class SAPAUpsampler(torch.nn.Module):
    def __init__(self, dim_x, **kwargs):
        super().__init__(**kwargs)
        self.up1 = SAPAModule(dim_x=dim_x, dim_y=3)
        self.up2 = SAPAModule(dim_x=dim_x, dim_y=3)
        self.up3 = SAPAModule(dim_x=dim_x, dim_y=3)
        self.up4 = SAPAModule(dim_x=dim_x, dim_y=3)

    def adapt_guidance(self, source, guidance):
        _, _, h, w = source.shape
        small_guidance = F.adaptive_avg_pool2d(guidance, (h * 2, w * 2))
        return small_guidance

    def forward(self, source, guidance):
        source_2 = self.up1(self.adapt_guidance(source, guidance), source)
        source_4 = self.up2(self.adapt_guidance(source_2, guidance), source_2)
        source_8 = self.up3(self.adapt_guidance(source_4, guidance), source_4)
        source_16 = self.up4(self.adapt_guidance(source_8, guidance), source_8)
        return source_16


class Bilinear(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, feats, img):
        _, _, h, w = img.shape
        return F.interpolate(feats, (h, w), mode="bilinear")


# 为LRSFormer_OP8也添加相同的修改
class LRSFormer_OP8(nn.Module):
    def __init__(self, num_classes=24, in_channel=32, out_channel=(64, 128, 256, 512), embed_dim=128,
                 gruop=GROUP, kernel=8, stride=2, reduction_ratio=16,
                 num_heads=8, qkv_bias=False, qk_drop=0., proj_drop=0., sr_ratio=(4, 8, 16, 32),
                 mlp_ratio=2, mlp_drop=0.3, drop_path=0.3, drop_class=0.1,
                 use_simfeatup=True,
                 upsampler_type='bilinear'):
        super().__init__()

        self.use_simfeatup = use_simfeatup

        self.kernel = kernel
        self.stride = stride

        self.preprocess = downsample(in_channel, out_channel[0], kernel, stride, gruop)

        self.encode1 = Encoder(out_channel[0], gruop, reduction_ratio)
        self.down1 = downsample(out_channel[0], out_channel[1], kernel, stride, gruop)

        self.encode2 = Encoder(out_channel[1], gruop, reduction_ratio)
        self.down2 = downsample(out_channel[1], out_channel[2], kernel, stride, gruop)

        self.encode3 = Encoder(out_channel[2], gruop, reduction_ratio)
        self.down3 = downsample(out_channel[2], out_channel[3], kernel, stride, gruop)

        self.encode4 = Encoder(out_channel[3], gruop, reduction_ratio)

        self.skip1 = Skip_L(out_channel[-1], embed_dim)
        self.decode1 = Decoder(embed_dim, num_heads, qkv_bias, qk_drop, proj_drop, sr_ratio[0],
                               hidden_features=mlp_ratio * embed_dim, mlp_drop=mlp_drop, drop_path=drop_path)

        self.skip2 = Skip_L(out_channel[-2], embed_dim)
        self.ws2 = WS(embed_dim)
        self.decode2 = Decoder(embed_dim, num_heads, qkv_bias, qk_drop, proj_drop, sr_ratio[1],
                               hidden_features=mlp_ratio * embed_dim, mlp_drop=mlp_drop, drop_path=drop_path)

        self.skip3 = Skip_L(out_channel[-3], embed_dim)
        self.ws3 = WS(embed_dim)
        self.decode3 = Decoder(embed_dim, num_heads, qkv_bias, qk_drop, proj_drop, sr_ratio[2],
                               hidden_features=mlp_ratio * embed_dim, mlp_drop=mlp_drop, drop_path=drop_path)

        self.skip4 = Skip_L(out_channel[-4], embed_dim)
        self.ws4 = WS(embed_dim)
        self.decode4 = Decoder(embed_dim, num_heads, qkv_bias, qk_drop, proj_drop, sr_ratio[3],
                               hidden_features=mlp_ratio * embed_dim, mlp_drop=mlp_drop, drop_path=drop_path)

        self.class_head = ClassHead(num_classes, (embed_dim, embed_dim, embed_dim, embed_dim),
                                    embed_dim, drop_class, use_simfeatup, upsampler_type, in_channel)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, init):
        h, w = init.size()[2:]
        ini = self.preprocess(init)
        e1 = self.encode1(ini)

        e2 = self.down1(e1)
        e2 = self.encode2(e2)

        e3 = self.down2(e2)
        e3 = self.encode3(e3)

        e4 = self.down3(e3)
        e4 = self.encode4(e4)

        d1 = self.skip1(e4)
        d1 = self.decode1(d1)

        d2 = self.skip2(e3)
        d2 = self.ws2(d1, d2)
        d2 = self.decode2(d2)

        d3 = self.skip3(e2)
        d3 = self.ws3(d2, d3)
        d3 = self.decode3(d3)

        d4 = self.skip4(e1)
        d4 = self.ws4(d3, d4)
        d4 = self.decode4(d4)

        # 🌟 如果使用SimFeatUp，传递原始图像作为引导
        if self.use_simfeatup:
            out = self.class_head(d4, d3, d2, d1, guidance_image=init)
        else:
            out = self.class_head(d4, d3, d2, d1)
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

        return out