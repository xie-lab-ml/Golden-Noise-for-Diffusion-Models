import torch
import torch.nn as nn

__all__ = ['EncoderNoiseUnet']

class EncoderNoiseUnet(nn.Module):
      def __init__(self, channels, out_channel=512):
            super(EncoderNoiseUnet, self).__init__()
            
            output_size = (1, 4, 128, 128)
            self.outputs = nn.ModuleList([])

            for channel in channels:
                  self.outputs.append(nn.Sequential(nn.Conv2d(channel, channel, (3,3), (1,1), (1,1)),
                                                    nn.ReLU(),
                                                    nn.Conv2d(channel, out_channel, (3,3), (1,1), (1,1))
                                                    ))
                  
            self.inter = lambda x: nn.functional.interpolate(x, size=[output_size[2],output_size[3]])

            self.bn = nn.GroupNorm(1, out_channel)
            self.relu = nn.ReLU()

            self.linear = nn.Conv2d(out_channel, 4, (3,3), (1,1), (1,1))

      def forward(self, xs):
            results = []

            for x, o in zip(xs, self.outputs):
                  results.append(self.inter(o(x)))

            result = torch.stack(results, 0).mean(0)

            return self.linear(self.relu(self.bn(result)))