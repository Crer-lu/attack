import torch
from torch import nn as nn
from torch.nn import functional as func
class GaussianNoiseSimulator(nn.Module):
    def __init__(self):
        super(GaussianNoiseSimulator,self).__init__()

    def forward(self,x,noise_rate):
        noise = torch.normal(torch.zeros_like(x),1)
        y = x + noise_rate * noise
        y = torch.clip(y,0,1)
        return y
    
class AdversarialNoiseSimulator(nn.Module):
    def __init__(self,channels = [512,512,512,512],latent_channel = 16):
        super(AdversarialNoiseSimulator,self).__init__()
        self.dense = nn.ModuleList()
        last_channel = latent_channel
        for channel in channels:
            self.dense.append(nn.Linear(last_channel,channel))
            last_channel = channel
        self.out_dense = nn.Linear(last_channel,latent_channel)
        
    def forward(self,latent):
        y = latent
        for dense in self.dense:
            y = func.relu(dense(y))
        y = self.out_dense(y) + latent
        return y

class ImageNoiseSimulator(nn.Module):
    def __init__(self, channel=64, leaky_slope = 0.05):
        super(ImageNoiseSimulator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1         ,channel*1, 4, 2, 1, bias=False),nn.BatchNorm2d(channel*1))
        self.conv2 = nn.Sequential(nn.Conv2d(channel*1, channel*2, 4, 2, 1, bias=False),nn.BatchNorm2d(channel*2))

        self.conv3 = nn.Sequential(nn.Conv2d(channel*2, channel*2, 3, 1, 1, bias=False),nn.BatchNorm2d(channel*2))
        self.conv4 = nn.Sequential(nn.Conv2d(channel*2, channel*2, 3, 1, 1, bias=False),nn.BatchNorm2d(channel*2))

        self.conv5 = nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(channel*2, channel*1, 3, 1, 1, bias=False),nn.BatchNorm2d(channel))
        self.conv6 = nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(channel*1, channel*1, 5, 1, 2, bias=False),nn.BatchNorm2d(channel))
        self.conv7 = nn.Conv2d(channel*1, 1, 5, 1, 2, bias=False)
        
        self.leaky_slope=  leaky_slope
    def forward(self, x):
        x = x.unsqueeze(1)
        y = x
        y = func.leaky_relu(self.conv1(y), self.leaky_slope)
        y = func.leaky_relu(self.conv2(y), self.leaky_slope)
        y = func.leaky_relu(self.conv3(y), self.leaky_slope)
        y = func.leaky_relu(self.conv4(y), self.leaky_slope)
        y = func.leaky_relu(self.conv5(y), self.leaky_slope)
        y = func.leaky_relu(self.conv6(y), self.leaky_slope)
        y = self.conv7(y)
        y = func.tanh(x + y)
        y = y.squeeze()
        return y
    
class ConditionImageNoiseSimulator(nn.Module):
    def __init__(self, channel=64, leaky_slope = 0.05,classes = 10,size = 28):
        
        super(ConditionImageNoiseSimulator, self).__init__()
        latent_size = size // 4
        self.latent_size = latent_size
        self.channel = channel
        self.embedding = nn.Linear(classes,latent_size*latent_size*channel*2)

        self.conv1 = nn.Sequential(nn.Conv2d(1         ,channel*1, 4, 2, 1, bias=False),nn.BatchNorm2d(channel*1))
        self.conv2 = nn.Sequential(nn.Conv2d(channel*1, channel*2, 4, 2, 1, bias=False),nn.BatchNorm2d(channel*2))

        self.conv3 = nn.Sequential(nn.Conv2d(channel*2, channel*2, 3, 1, 1, bias=False),nn.BatchNorm2d(channel*2))
        self.conv4 = nn.Sequential(nn.Conv2d(channel*2, channel*2, 3, 1, 1, bias=False),nn.BatchNorm2d(channel*2))

        self.conv5 = nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(channel*2, channel*1, 3, 1, 1, bias=False),nn.BatchNorm2d(channel))
        self.conv6 = nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(channel*1, channel*1, 5, 1, 2, bias=False),nn.BatchNorm2d(channel))
        self.conv7 = nn.Conv2d(channel*1, 1, 5, 1, 2, bias=False)
        
        self.leaky_slope = leaky_slope

    def forward(self, x, y):
        batch_size = x.shape[0]
        embedding = self.embedding(y)
        embedding = embedding.reshape(batch_size,self.channel*2,self.latent_size,self.latent_size)
        x = x.unsqueeze(1)
        y = x
        y = func.leaky_relu(self.conv1(y), self.leaky_slope)
        y = func.leaky_relu(self.conv2(y), self.leaky_slope)
        y = func.relu(y + embedding)
        y = func.leaky_relu(self.conv3(y), self.leaky_slope)
        y = func.leaky_relu(self.conv4(y), self.leaky_slope)
        y = func.leaky_relu(self.conv5(y), self.leaky_slope)
        y = func.leaky_relu(self.conv6(y), self.leaky_slope)
        y = self.conv7(y)
        y = func.tanh(x + y)
        y = y.squeeze()
        return y
    
class NumberNoiseSimulation(nn.Module):
    def __init__(self, channel=64, leaky_slope = 0.05):
        super(NumberNoiseSimulation, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1         ,channel*1, 4, 2, 1, bias=False),nn.BatchNorm2d(channel*1))
        self.conv2 = nn.Sequential(nn.Conv2d(channel*1, channel*2, 4, 2, 1, bias=False),nn.BatchNorm2d(channel*2))

        self.conv3 = nn.Sequential(nn.Conv2d(channel*2, channel*2, 3, 1, 1, bias=False),nn.BatchNorm2d(channel*2))
        self.conv4 = nn.Sequential(nn.Conv2d(channel*2, channel*2, 3, 1, 1, bias=False),nn.BatchNorm2d(channel*2))

        self.conv5 = nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(channel*2, channel*1, 3, 1, 1, bias=False),nn.BatchNorm2d(channel))
        self.conv6 = nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(channel*1, channel*1, 5, 1, 2, bias=False),nn.BatchNorm2d(channel))
        self.conv7 = nn.Conv2d(channel*1, 1, 5, 1, 2, bias=False)
        
        self.leaky_slope=  leaky_slope

    def forward(self,mnist):
        y = mnist.unsqueeze(1)
        y = func.leaky_relu(self.conv1(y), self.leaky_slope)
        y = func.leaky_relu(self.conv2(y), self.leaky_slope)
        y = func.leaky_relu(self.conv3(y), self.leaky_slope)
        y = func.leaky_relu(self.conv4(y), self.leaky_slope)
        y = func.leaky_relu(self.conv5(y), self.leaky_slope)
        y = func.leaky_relu(self.conv6(y), self.leaky_slope)
        y = func.tanh(self.conv7(y))
        y = y.squeeze()
        return y
    
class ChineseNoiseSimulation(nn.Module):
    def __init__(self, channel=64, leaky_slope = 0.05):
        super(ChineseNoiseSimulation, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1         ,channel*1, 4, 2, 1, bias=False),nn.BatchNorm2d(channel*1))
        self.conv2 = nn.Sequential(nn.Conv2d(channel*1, channel*2, 4, 2, 1, bias=False),nn.BatchNorm2d(channel*2))

        self.conv3 = nn.Sequential(nn.Conv2d(channel*2, channel*2, 3, 1, 1, bias=False),nn.BatchNorm2d(channel*2))
        self.conv4 = nn.Sequential(nn.Conv2d(channel*2, channel*2, 3, 1, 1, bias=False),nn.BatchNorm2d(channel*2))

        self.conv5 = nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(channel*2, channel*1, 3, 1, 1, bias=False),nn.BatchNorm2d(channel))
        self.conv6 = nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(channel*1, channel*1, 5, 1, 2, bias=False),nn.BatchNorm2d(channel))
        self.conv7 = nn.Conv2d(channel*1, 1, 5, 1, 2, bias=False)
        
        self.leaky_slope=  leaky_slope

    def forward(self,mnist):
        y = mnist.unsqueeze(1)
        y = func.leaky_relu(self.conv1(y), self.leaky_slope)
        y = func.leaky_relu(self.conv2(y), self.leaky_slope)
        y = func.leaky_relu(self.conv3(y), self.leaky_slope)
        y = func.leaky_relu(self.conv4(y), self.leaky_slope)
        y = func.leaky_relu(self.conv5(y), self.leaky_slope)
        y = func.leaky_relu(self.conv6(y), self.leaky_slope)
        y = func.tanh(self.conv7(y))
        y = y.squeeze()
        return y
    