import torch.nn as nn

__all__ = ['NoiseUnet']

class NoiseUnet(nn.Module):
    def __init__(self, conv_in, in_channels, out_channels):
        super(NoiseUnet, self).__init__()

        self.conv_in = conv_in
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(out_channels, in_channels, kernel_size=(7,7), padding=(3,3), stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(7,7), padding=(3,3), stride=(1,1))
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=(7,7), padding=(3,3), stride=(1,1))
        


    def forward(self, x, residual=False):

        if residual:
            x = self.conv3(self.relu(self.conv2(self.bn(self.conv1(self.relu(self.conv_in(x))))))) + x
        else:
            x = self.conv3(self.relu(self.conv2(self.bn(self.conv1(self.relu(self.conv_in(x))))))) 
        return x