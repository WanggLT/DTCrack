import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import DeformConv2d
from models.backbones import ResNetD
import math
import warnings


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        # >>> w = torch.empty(3, 5)
        # >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )


class DCNv2(nn.Module):
    def __init__(self, c1, c2, k, s, p, g=1):
        super().__init__()
        self.dcn = DeformConv2d(c1, c2, k, s, p, groups=g)
        self.offset_mask = nn.Conv2d(c2, g * 3 * k * k, k, s, p)
        self._init_offset()

    def _init_offset(self):
        self.offset_mask.weight.data.zero_()
        self.offset_mask.bias.data.zero_()

    def forward(self, x, offset):
        out = self.offset_mask(offset)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        mask = mask.sigmoid()
        return self.dcn(x, offset, mask)


class FSM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv_atten = nn.Conv2d(c1, c1, 1, bias=False)
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        atten = self.conv_atten(F.avg_pool2d(x, x.shape[2:])).sigmoid()
        feat = torch.mul(x, atten)
        x = x + feat
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.lateral_conv = FSM(c1, c2)
        self.offset = nn.Conv2d(c2 * 2, c2, 1, bias=False)
        self.dcpack_l2 = DCNv2(c2, c2, 3, 1, 1, 8)

    def forward(self, feat_l, feat_s):
        feat_up = feat_s
        if feat_l.shape[2:] != feat_s.shape[2:]:
            feat_up = F.interpolate(feat_s, size=feat_l.shape[2:], mode='bilinear', align_corners=False)

        feat_arm = self.lateral_conv(feat_l)
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))

        feat_align = F.relu(self.dcpack_l2(feat_up, offset))
        return feat_align + feat_arm


class FaPNHead(nn.Module):
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        in_channels = in_channels[::-1]
        self.align_modules = nn.ModuleList([ConvModule(in_channels[0], channel, 1)])
        self.output_convs = nn.ModuleList([])

        for ch in in_channels[1:]:
            self.align_modules.append(FAM(ch, channel))
            self.output_convs.append(ConvModule(channel, channel, 3, 1, 1))

        self.conv_seg = nn.Conv2d(channel, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features) -> Tensor:
        features = features[::-1]
        out = self.align_modules[0](features[0])

        for feat, align_module, output_conv in zip(features[1:], self.align_modules[1:], self.output_convs):
            out = align_module(feat, out)
            out = output_conv(out)
        out = self.conv_seg(self.dropout(out))
        return out

class FaPN(nn.Module):
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        self.backbone = ResNetD('50')
        self.decode_head = FaPNHead(in_channels, 256, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y



if __name__ == '__main__':
    # from models.backbones import ResNetD
    #
    # backbone = ResNetD('50')
    # head = FaPNHead([256, 512, 1024, 2048], 128, 19)
    # x = torch.randn(2, 3, 224, 224)
    # features = backbone(x)
    # out = head(features)
    # out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
    # print(out.shape)
    model = FaPN(in_channels=[256, 512, 1024, 2048],num_classes=2)
    # model.load_state_dict(torch.load('checkpoints/pretrained/segformer/segformer.b0.ade.pth', map_location='cpu'))
    x = torch.rand((2, 3, 256, 256))
    y = model(x)
    print(y.shape)