import torch.nn as nn
import torch


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()

        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))

        self.excitation = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Read the number of batch data images and the number of channels
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)

        return x * y.expand_as(x)
