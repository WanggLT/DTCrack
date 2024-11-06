import torch
import math
import copy
from torch import nn
from torch.nn import functional as F
from mmcv.cnn.bricks import ConvModule, build_activation_layer, build_norm_layer
from ops_dcnv3.modules.dcnv3 import DCNv3 as dcn
from timm.models.layers import DropPath, to_2tuple
from torch.utils import checkpoint
from mmcv.runner.checkpoint import load_checkpoint
from mmseg.models.builder import BACKBONES as seg_BACKBONES
from mmseg.utils import get_root_logger
from mmcv.runner import _load_checkpoint
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
has_mmseg = True


class PatchEmbed(nn.Module):
    def __init__(self,
                 patch_size=16,
                 stride=16,
                 padding=0,
                 in_channels=3,
                 embed_dim=768,
                 norm_layer=dict(type='BN2d'),
                 act_cfg=None, ):
        super().__init__()
        self.proj = ConvModule(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
            norm_cfg=norm_layer,
            act_cfg=act_cfg,
        )

    def forward(self, x):
        return self.proj(x)


class Attention(nn.Module):
    def __init__(self, dim,
                 num_heads=1,
                 qk_scale=None,
                 attn_drop=0,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # head_dim开平方
        self.sr_ratio = sr_ratio
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        if sr_ratio > 1:  # sr_ratio 8， 4， 2， 1
            if sr_ratio == 8:
                self.sr1 = nn.Sequential(
                    ConvModule(dim, dim,
                               kernel_size=sr_ratio + 3,
                               stride=sr_ratio,
                               padding=(sr_ratio + 3) // 2,
                               groups=dim,
                               bias=False,
                               norm_cfg=dict(type='BN2d'),
                               act_cfg=dict(type='ReLU')),
                    ConvModule(dim, dim,
                               kernel_size=1,
                               groups=dim,
                               bias=False,
                               norm_cfg=dict(type='BN2d'),
                               act_cfg=None, ), )
                self.sr2 = nn.Sequential(
                    ConvModule(dim, dim,
                               kernel_size=sr_ratio // 2 + 3,
                               stride=sr_ratio // 2,
                               padding=(sr_ratio // 2 + 3) // 2,
                               groups=dim,
                               bias=False,
                               norm_cfg=dict(type='BN2d'),
                               act_cfg=dict(type='ReLU')),
                    ConvModule(dim, dim,
                               kernel_size=1,
                               groups=dim,
                               bias=False,
                               norm_cfg=dict(type='BN2d'),
                               act_cfg=None, ), )
                self.kv2 = nn.Linear(dim, dim, bias=True)
            if sr_ratio == 4:
                self.sr1 = nn.Sequential(
                    ConvModule(dim, dim,
                                kernel_size=sr_ratio + 3,
                                stride=sr_ratio,
                                padding=(sr_ratio + 3) // 2,
                                groups=dim,
                                bias=False,
                                norm_cfg=dict(type='BN2d'),
                                act_cfg=dict(type='ReLU')),
                    ConvModule(dim, dim,
                                kernel_size=1,
                                groups=dim,
                                bias=False,
                                norm_cfg=dict(type='BN2d'),
                                act_cfg=None, ), )
                self.sr2 = nn.Sequential(
                    ConvModule(dim, dim,
                                kernel_size=sr_ratio // 2 + 3,
                                stride=sr_ratio // 2,
                                padding=(sr_ratio // 2 + 3) // 2,
                                groups=dim,
                                bias=False,
                                norm_cfg=dict(type='BN2d'),
                                act_cfg=dict(type='ReLU')),
                    ConvModule(dim, dim,
                                kernel_size=1,
                                groups=dim,
                                bias=False,
                                norm_cfg=dict(type='BN2d'),
                                act_cfg=None, ), )
            if sr_ratio == 2:
                self.sr1 = nn.Sequential(
                    ConvModule(dim, dim,
                                kernel_size=sr_ratio + 3,
                                stride=sr_ratio,
                                padding=(sr_ratio + 3) // 2,
                                groups=dim,
                                bias=False,
                                norm_cfg=dict(type='BN2d'),
                                act_cfg=dict(type='ReLU')),
                    ConvModule(dim, dim,
                                kernel_size=1,
                                groups=dim,
                                bias=False,
                                norm_cfg=dict(type='BN2d'),
                                act_cfg=None, ), )
                self.sr2 = nn.Sequential(
                    ConvModule(dim, dim,
                               kernel_size=sr_ratio // 2 + 3,
                               stride=sr_ratio // 2,
                               padding=(sr_ratio // 2 + 3) // 2,
                               groups=dim,
                               bias=False,
                               norm_cfg=dict(type='BN2d'),
                               act_cfg=dict(type='ReLU')),
                    ConvModule(dim, dim,
                               kernel_size=1,
                               groups=dim,
                               bias=False,
                               norm_cfg=dict(type='BN2d'),
                               act_cfg=None, ), )

        else:
            self.sr = nn.Identity()  # Identity()输入是什么，输出就是什么
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, relative_pos_enc=None):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)   # transpose函数就是将矩阵转置，-1是将倒数第一维度放在第一维度
        if self.sr_ratio > 1:
            kv1 = self.sr1(x)
            kv2 = self.sr2(x)
            kv1 = self.local_conv(kv1) + kv1
            kv2 = self.local_conv(kv2) + kv2
            k1, v1 = torch.chunk(self.kv(kv1), chunks=2, dim=1)
            k2, v2 = torch.chunk(self.kv(kv2), chunks=2, dim=1)
            k1 = k1.reshape(B, self.num_heads // 2, C // self.num_heads, -1)
            v1 = v1.reshape(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
            k2 = k2.reshape(B, self.num_heads // 2, C // self.num_heads, -1)
            v2 = v2.reshape(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
            attn1 = (q[:, :self.num_heads // 2] @ k1) * self.scale
            attn2 = (q[:, :self.num_heads // 2] @ k2) * self.scale
            if relative_pos_enc is not None:
                relative_pos_enc = relative_pos_enc.permute(1, 0, 2, 3)
                n, b, h, w = relative_pos_enc.shape
                relative_pos_enc1, relative_pos_enc2 = relative_pos_enc[0].reshape(-1, b, h, w).permute(1,0,2,3), relative_pos_enc[1].reshape(-1, b, h, w).permute(1,0,2,3)
                if attn1.shape[2:] != relative_pos_enc1.shape[2:]:
                    relative_pos_enc1 = F.interpolate(relative_pos_enc1, size=attn1.shape[2:],
                                                    mode='bicubic', align_corners=False)
                attn1 = attn1 + relative_pos_enc1
                if attn2.shape[2:] != relative_pos_enc2.shape[2:]:
                    relative_pos_enc2 = F.interpolate(relative_pos_enc2, size=attn2.shape[2:],
                                                    mode='bicubic', align_corners=False)
                attn2 = attn2 + relative_pos_enc2
            attn1 = torch.softmax(attn1, dim=-1)
            attn1 = self.attn_drop(attn1)
            attn2 = torch.softmax(attn2, dim=-1)
            attn2 = self.attn_drop(attn2)
            x1 = (attn1 @ v1)
            x2 = (attn2 @ v2)
            x = torch.cat([x1, x2], dim=-1)
            return x.reshape(B, C, H, W)
        else:
            kv = self.sr(x)
            kv = self.local_conv(kv) + kv
            k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)  # 沿通道维度进行分割，分成两份
            k = k.reshape(B, self.num_heads, C // self.num_heads, -1)
            v = v.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
            attn = (q @ k) * self.scale
            if relative_pos_enc is not None:
                if attn.shape[2:] != relative_pos_enc.shape[2:]:
                    relative_pos_enc = F.interpolate(relative_pos_enc, size=attn.shape[2:],
                                                    mode='bicubic', align_corners=False)
                attn = attn + relative_pos_enc
            attn = torch.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(-1, -2)
            return x.reshape(B, C, H, W)


class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i] // 2,
                             groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x


class Mlp(nn.Module):
    """
    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            build_activation_layer(act_cfg),
            nn.BatchNorm2d(hidden_features),
        )
        self.dwconv = MultiScaleDWConv(hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_features),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x) + x
        x = self.norm(self.act(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class DCNv3_YOLO(nn.Module):
    def __init__(self, inc, reduction_ratio=2, k=1, s=1, p=None, g=1, d=1, act_cfg=dict(type='GELU')):
        super().__init__()

        self.Conv = nn.Conv2d(inc, inc, kernel_size=1)
        self.dcnv3 = dcn(inc, kernel_size=k, stride=s, group=g, dilation=d)
        self.Conv1 = nn.Conv2d(inc,inc,kernel_size=1,stride=s,dilation=d, groups=g)
        self.Conv2 = nn.Conv2d(inc, inc, kernel_size=1)
        self.bn = nn.BatchNorm2d(inc)
        self.act = build_activation_layer(act_cfg)

    def forward(self, x):
        x = self.Conv(x)
        x1 = x
        # x = x.permute(0, 2, 3, 1)
        # x = self.dcnv3(x)
        x = self.Conv1(x)
        x = self.Conv2(x)
        # x = x.permute(0, 3, 1, 2)
        x = self.bn(x)
        x = x + x1
        # x = self.act(self.bn(x))
        return x


class HybridTokenMixer(nn.Module):
    def __init__(self, dim,
                 kernel_size=3,
                 num_groups=2,
                 num_heads=1,
                 sr_ratio=1,
                 reduction_ratio=8):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."

        self.local_unit = DCNv3_YOLO(inc=dim // 2, k=kernel_size, g=num_groups)
        self.global_unit = Attention(dim=dim // 2, num_heads=num_heads, sr_ratio=sr_ratio)
        inner_dim = max(16, dim // reduction_ratio)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x, relative_pos_enc=None):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        B, _, H, W = x1.shape
        x1 = self.local_unit(x1)
        x1 = F.interpolate(x1, size=(H,W), mode='bilinear', align_corners=False)
        x2 = self.global_unit(x2, relative_pos_enc)
        x = torch.cat([x1, x2], dim=1)
        x = self.proj(x) + x
        return x


class LayerScale(nn.Module):
    def __init__(self,dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value,requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim),requires_grad=True)

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x


class Block(nn.Module):
    def __init__(self,
                 dim=64,
                 kernel_size=3,
                 sr_ration=1,
                 num_groups=2,
                 num_heads=1,
                 mlp_ratio=4,
                 norm_cfg=dict(type="GN", num_groups=1),
                 act_cfg=dict(type='GELU'),
                 drop=0,
                 drop_path=0,
                 layer_scale_init_value=1e-5,
                 grad_checkpoint=False):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.token_mixer = HybridTokenMixer(dim,
                                            kernel_size=kernel_size,
                                            num_groups=num_groups,
                                            num_heads=num_heads,
                                            sr_ratio=sr_ration)
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_cfg=act_cfg, drop=drop,)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale(dim, layer_scale_init_value)
            self.layer_scale_2 = LayerScale(dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()

    def _forward_impl(self, x, relative_pos_enc=None):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.layer_scale_1(self.token_mixer(self.norm1(x), relative_pos_enc)))
        x = x + self.drop_path(self.layer_scale_2(self.mlp(self.norm2(x))))
        return x

    def forward(self, x, relative_pos_enc=None):
        if self.grad_checkpoint and x.requires_grad:
            x = checkpoint.checkpoint(self._forward_impl(), x, relative_pos_enc)
        else:
            x = self._forward_impl(x, relative_pos_enc)
        return x


def basic_blocks(dim,
                 index,
                 layers,
                 kernel_size=3,
                 num_groups=2,
                 num_heads=1,
                 sr_ratio=1,
                 mlp_ratio=4,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 drop_rate=0,
                 drop_path_rate=0,
                 layer_scale_init_value=1e-5,
                 grad_checkpoint=False):
    blocks = nn.ModuleList()
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(
            Block(
                dim, kernel_size=kernel_size,
                num_groups=num_groups,num_heads=num_heads,
                sr_ration=sr_ratio, mlp_ratio=mlp_ratio,
                norm_cfg=norm_cfg, act_cfg=act_cfg, drop=drop_rate,
                drop_path=block_dpr, layer_scale_init_value=layer_scale_init_value,
                grad_checkpoint=grad_checkpoint,
            )
        )
    return blocks


class Mynet(nn.Module):
    arch_settings = {
        **dict.fromkeys(['t', 'tiny', 'T'],
                        {'layers': [3, 3, 9, 3],
                         'embed_dims': [64, 128, 320, 512],
                         'kernel_size': [3, 3, 3, 3],
                         'num_groups': [2, 2, 2, 2],
                         'sr_ratio': [8, 4, 2, 1],
                         'num_heads': [2, 4, 8, 16],
                         'mlp_ratios': [4, 4, 4, 4],
                         'layer_scale_init_value': 1e-5}),
        # **dict.fromkeys(['t', 'tiny', 'T'],
        #                 {'layers': [3, 3, 9, 3],
        #                  'embed_dims': [48, 96, 224, 448],
        #                  'kernel_size': [7, 7, 7, 7],
        #                  'num_groups': [2, 2, 2, 2],
        #                  'sr_ratio': [8, 4, 2, 1],
        #                  'num_heads': [1, 2, 4, 8],
        #                  'mlp_ratios': [4, 4, 4, 4],
        #                  'layer_scale_init_value': 1e-5}),

        **dict.fromkeys(['s', 'small', 'S'],
                        {'layers': [4, 4, 12, 4],
                         'embed_dims': [64, 128, 320, 512],
                         'kernel_size': [7, 7, 7, 7],
                         'num_groups': [2, 2, 2, 2],
                         'sr_ratio': [8, 4, 2, 1],
                         'num_heads': [1, 2, 4, 8],
                         'mlp_ratios': [6, 6, 4, 4],
                         'layer_scale_init_value': 1e-5, }),
    }

    def __init__(self,
                 image_size=224,
                 arch='t',
                 norm_cfg=dict(type='GN',num_groups=1),
                 act_cfg=dict(type='GELU'),
                 in_chans=3,
                 in_patch_size=7,
                 in_stride=4,
                 in_pad=3,
                 down_patch_size=3,
                 down_stride=2,
                 down_pad=1,
                 drop_rate=0,
                 drop_path_rate=0,
                 grad_checkpoint=False,
                 checkpoint_stage=[0] * 4,
                 num_classes=2,
                 fork_feat=True,
                 start_level=0,
                 init_cfg=None,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.grad_checkpoint = grad_checkpoint
        if isinstance(arch, str):  # 如果 arch 是字符串类型，表示用户输入的是预定义的模型架构的缩写，比如 'tiny'
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]  # 将用户输入的架构缩写映射到相应的配置参数。
        elif isinstance(arch, dict): # 如果 arch 是字典类型，表示用户直接传递了模型的配置参数
            assert 'layers' in arch and 'embed_dims' in arch, \
                f'The arch dict must have "layers" and "embed_dims", ' \
                f'but got {list(arch.keys())}.'

        layers = arch['layers']
        embed_dims = arch['embed_dims']
        kernel_size = arch['kernel_size']
        num_groups = arch['num_groups']
        sr_ratio = arch['sr_ratio']
        num_heads = arch['num_heads']

        if not grad_checkpoint:
            checkpoint_stage = [0] * 4

        mlp_ratios = arch['mlp_ratios'] if 'mlp_ratios' in arch else [4, 4, 4, 4]
        # 获取模型中用于初始化层缩放的参数。如果配置字典中包含 'layer_scale_init_value' 键，则使用相应的值；
        # 否则，使用默认值 1e-5。这个值将在构建模型的主要组件时使用，用于初始化每个阶段的层缩放。
        layer_scale_init_value = arch['layer_scale_init_value'] if 'layer_scale_init_value' in arch else 1e-5
        self.patch_embed = PatchEmbed(patch_size=in_patch_size, stride=in_stride,
                                      padding=in_pad, in_channels=in_chans,
                                      embed_dim=embed_dims[0])
        self.relative_pos_enc = []
        self.pos_enc_record = []
        image_size = to_2tuple(image_size)  # 将image_size变为2元数组，224->[224,224]
        image_size = [math.ceil(image_size[0]/in_stride),
                      math.ceil(image_size[1]/in_stride)]  # math.ceil(x)将x向上舍入到最接近的整数，[224,224]->[56,56]
        for i in range(4):
            num_patches = image_size[0]*image_size[1]
            sr_patches = math.ceil(
                image_size[0]/sr_ratio[i])*math.ceil(image_size[1]/sr_ratio[i])
            self.relative_pos_enc.append(
                nn.Parameter(torch.zeros(1, num_heads[i], num_patches, sr_patches),requires_grad=True))
            self.pos_enc_record.append([image_size[0], image_size[1],
                                       math.ceil(image_size[0]/sr_ratio[i]),
                                       math.ceil(image_size[1]/sr_ratio[i])])
            image_size = [math.ceil(image_size[0] / 2),
                          math.ceil(image_size[1] / 2)]
            self.relative_pos_enc = nn.ParameterList(self.relative_pos_enc)

        network = []

        for i in range(len(layers)):
            stage = basic_blocks(
                embed_dims[i],
                i,
                layers,
                kernel_size=kernel_size[i],
                num_groups=num_groups[i],
                num_heads=num_heads[i],
                sr_ratio=sr_ratio[i],
                mlp_ratio=mlp_ratios[i],
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value,
                grad_checkpoint=checkpoint_stage[i],
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        padding=down_pad,
                        in_channels=embed_dims[i],
                        embed_dim=embed_dims[i + 1]))
        self.network = nn.ModuleList(network)
        if self.fork_feat:
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb < start_level:
                    layer = nn.Identity()
                else:
                    layer = build_norm_layer(norm_cfg, embed_dims[(i_layer + 1) // 2])[1]
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.classifier = nn.Sequential(
                build_norm_layer(norm_cfg, embed_dims[-1])[1],
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(embed_dims[-1], num_classes, kernel_size=1),
            ) if num_classes > 0 else nn.Identity()

        self.apply(self._init_model_weights)
        self.init_cfg = copy.deepcopy(init_cfg)

        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()
            self = nn.SyncBatchNorm.convert_sync_batchnorm(self)
            self.train()

    def _init_model_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    '''
    init for mmdetection or mmsegmentation 
    by loading imagenet pre-trained weights
    '''

    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, False)

            # show for debug
            print('missing_keys: ', missing_keys)
            print('unexpected_keys: ', unexpected_keys)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier[-1].out_channels = num_classes
        else:
            self.classifier = nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        outs1 = []
        pos_idx = 0
        for idx in range(len(self.network)):
            if idx in [0, 2, 4, 6]:
                for blk in self.network[idx]:
                    x = blk(x, self.relative_pos_enc[pos_idx])
                pos_idx += 1
                x1 = getattr(self, f'norm{idx}')(x)
                outs1.append(x1)
            else:
                x = self.network[idx](x)
        # if self.fork_feat and (idx in self.out_indices):
        #     x_out = getattr(self, f'norm{idx}')(x)
        #     outs.append(x_out)
        if self.fork_feat:
            return outs1

        return x

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            # features of four stages for dense prediction
            return x
        else:
            # for image classification
            x = self.classifier(x).flatten(1)
            return x


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'crop_pct': 0.95,
        'interpolation': 'bicubic',   # 'bilinear' or 'bicubic'
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'classifier': 'classifier',
        **kwargs,
    }


def transxnet_t(pretrained=False, pretrained_cfg=None, **kwargs):
    model = Mynet(arch='t', **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    if pretrained:
        # ckpt = 'https://github.com/LMMMEng/TransXNet/releases/download/v1.0/transx-t.pth.tar'
        ckpt = 'pretrain/transx-t.pth.tar'
        load_checkpoint(model, ckpt)
    return model


if __name__ == '__main__':

    image_tensor = torch.rand(2, 3, 256, 256)
    net = transxnet_t(pretrained=False)
    device = torch.device('cuda:0')
    net = net.to(device)
    net = net.cuda()
    image_tensor = image_tensor.cuda()
    image_tensor = net(image_tensor)
    print(image_tensor)








