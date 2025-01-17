import torch
from torch import nn, Tensor
from torch.nn import functional as F


class BasicBlock(nn.Module):
    """2 Layer No Expansion Block
    """
    expansion: int = 1
    def __init__(self, c1, c2, s=1, d=1, downsample= None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 3, s, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, 3, 1, d if d != 1 else 1, d, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return F.relu(out)


class Bottleneck(nn.Module):
    """3 Layer 4x Expansion Block
    """
    expansion: int = 4
    def __init__(self, c1, c2, s=1, d=1, downsample=None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, 3, s, d if d != 1 else 1, d, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c2 * self.expansion, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(c2 * self.expansion)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return F.relu(out)


class Stem(nn.Sequential):
    def __init__(self, c1, ch, c2):
        super().__init__(
            nn.Conv2d(c1, ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, c2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1)
        )


resnetd_settings = {
    '18': [BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512]],
    '50': [Bottleneck, [3, 4, 6, 3], [256, 512, 1024, 2048]],
    '101': [Bottleneck, [3, 4, 23, 3], [256, 512, 1024, 2048]]
}


class ResNetD(nn.Module):
    def __init__(self, model_name: str = '50') -> None:
        super().__init__()
        assert model_name in resnetd_settings.keys(), f"ResNetD model name should be in {list(resnetd_settings.keys())}"
        block, depths, channels = resnetd_settings[model_name]

        self.inplanes = 128
        self.channels = channels
        self.stem = Stem(3, 64, self.inplanes)
        self.layer1 = self._make_layer(block, 64, depths[0], s=1)
        self.layer2 = self._make_layer(block, 128, depths[1], s=2)
        self.layer3 = self._make_layer(block, 256, depths[2], s=2, d=2)
        self.layer4 = self._make_layer(block, 512, depths[3], s=2, d=4)


    def _make_layer(self, block, planes, depth, s=1, d=1) -> nn.Sequential:
        downsample = None

        if s != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, s, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = nn.Sequential(
            block(self.inplanes, planes, s, d, downsample=downsample),
            *[block(planes * block.expansion, planes, d=d) for _ in range(1, depth)]
        )
        self.inplanes = planes * block.expansion
        return layers


    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)   # [1, 128, H/4, W/4]
        x1 = self.layer1(x)  # [1, 64/256, H/4, W/4]
        x2 = self.layer2(x1)  # [1, 128/512, H/8, W/8]
        x3 = self.layer3(x2)  # [1, 256/1024, H/16, W/16]
        x4 = self.layer4(x3)  # [1, 512/2048, H/32, W/32]
        return x1, x2, x3, x4


if __name__ == '__main__':
    model = ResNetD('18')
    model.load_state_dict(torch.load('checkpoints/backbones/resnetd/resnetd18.pth', map_location='cpu'), strict=False)
    x = torch.zeros(1, 3, 224, 224)
    outs = model(x)
    for y in outs:
        print(y.shape) 