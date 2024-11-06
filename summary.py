#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from models.net1 import CrackTransUnet
from models.UNet_v2 import UNetV2
from models.segformer import SegFormer
from models.backbones.lite_hrnet import LiteHRNet
from models.lawin import Lawin
from models.stdc import STDC
from models.regseg import RegSeg
from models.fapn import FaPN

if __name__ == "__main__":
    input_shape     = [256, 256]
    num_classes     = 2
    phi             = 'b3'
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = CrackTransUnet(n_classes=num_classes, deep_supervision=False)
    # model = UNetV2(n_classes=2, deep_supervision=True, pretrained_path='pretrain/pvt_v2_b2.pth')
    # model = SegFormer('MiT-B0', num_classes=2)
    # model = Lawin('MiT-B0', num_classes=2)
    # model = FaPN(in_channels=[256, 512, 1024, 2048], num_classes=2)
    # model = LiteHRNet(num_class=2)
    # model = RegSeg(num_class=2)
    model = STDC(num_class=2)
    model = model.cuda()
    summary(model, (3, input_shape[0], input_shape[1]))
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
