import torch.nn as nn
import torch.nn.functional as func

class Discriminator(nn.Module):
    def __init__(self, channel=64, classes = 10, leaky_slope = 0.05):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, channel, 4, 2, 1, bias=False)
        self.conv2 = nn.Sequential(nn.Conv2d(channel*1, channel*2, 4, 2, 1, bias=False),nn.BatchNorm2d(channel*2))
        self.conv3 = nn.Sequential(nn.Conv2d(channel*2, channel*4, 4, 2, 1, bias=False),nn.BatchNorm2d(channel*4))
        self.conv4 = nn.Conv2d(channel*4, classes +1, 3, 1, 0, bias=False)
        self.classes = classes
        self.leaky_slope = leaky_slope

    def forward(self, x):
        y = x.unsqueeze(1)
        y = func.leaky_relu(self.conv1(y), self.leaky_slope)
        y = func.leaky_relu(self.conv2(y), self.leaky_slope)
        y = func.leaky_relu(self.conv3(y), self.leaky_slope)
        y = func.softmax(self.conv4(y).squeeze(),dim=-1)
        y = y.squeeze()
        return y
