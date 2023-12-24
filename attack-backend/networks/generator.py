from torch import nn as nn
from torch.nn import functional as func

class Generator(nn.Module):
    def __init__(self, channel=64, leaky_slope = 0.05):
        super(Generator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1         ,channel*1, 4, 2, 1, bias=False),nn.BatchNorm2d(channel*1))
        self.conv2 = nn.Sequential(nn.Conv2d(channel*1, channel*2, 4, 2, 1, bias=False),nn.BatchNorm2d(channel*2))

        self.conv3 = nn.Sequential(nn.Conv2d(channel*2, channel*2, 3, 1, 1, bias=False),nn.BatchNorm2d(channel*2))
        self.conv4 = nn.Sequential(nn.Conv2d(channel*2, channel*2, 3, 1, 1, bias=False),nn.BatchNorm2d(channel*2))

        self.conv5 = nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(channel*2, channel*1, 3, 1, 1, bias=False),nn.BatchNorm2d(channel))
        self.conv6 = nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(channel*1, channel*1, 5, 1, 2, bias=False),nn.BatchNorm2d(channel))
        self.conv7 = nn.Conv2d(channel*1, 1, 5, 1, 2, bias=False)
        
        self.leaky_slope=  leaky_slope
    def forward(self, x):
        y = x.unsqueeze(1)
        y = func.leaky_relu(self.conv1(y), self.leaky_slope)
        y = func.leaky_relu(self.conv2(y), self.leaky_slope)
        y = func.leaky_relu(self.conv3(y), self.leaky_slope)
        y = func.leaky_relu(self.conv4(y), self.leaky_slope)
        y = func.leaky_relu(self.conv5(y), self.leaky_slope)
        y = func.leaky_relu(self.conv6(y), self.leaky_slope)
        y = func.tanh(self.conv7(y))
        y = y.squeeze()
        return y