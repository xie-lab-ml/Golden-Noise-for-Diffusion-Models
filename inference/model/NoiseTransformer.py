import torch.nn as nn

from torch.nn import functional as F
from timm import create_model


__all__ = ['NoiseTransformer']

class NoiseTransformer(nn.Module):
    def __init__(self, resolution=128):
        super().__init__()
        self.upsample = lambda x: F.interpolate(x, [224,224])
        self.downsample = lambda x: F.interpolate(x, [resolution,resolution])
        self.upconv = nn.Conv2d(7,4,(1,1),(1,1),(0,0))
        self.downconv = nn.Conv2d(4,3,(1,1),(1,1),(0,0))
        # self.upconv = nn.Conv2d(7,4,(1,1),(1,1),(0,0))
        self.swin = create_model("swin_tiny_patch4_window7_224",pretrained=True)


    def forward(self, x, residual=False):
        if residual:
            x = self.upconv(self.downsample(self.swin.forward_features(self.downconv(self.upsample(x))))) + x
        else:
            x = self.upconv(self.downsample(self.swin.forward_features(self.downconv(self.upsample(x)))))

        return x