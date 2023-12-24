from torch import nn as nn
from torch.nn import functional as func
    
class Classifier(nn.Module):
    def __init__(self,size = 28,channels = [256,256,256],class_channels = 10):
        super(Classifier,self).__init__()
        self.dense = nn.ModuleList()
        last_channel = size*size
        for channel in channels:
            dense = nn.Sequential(nn.Linear(last_channel,channel),nn.BatchNorm1d(channel))
            self.dense.append(dense)
            last_channel = channel
        self.class_dense = nn.Linear(last_channel,class_channels)
        self.size = size
        self.channels = channels
        self.class_channels = class_channels

    def forward(self,x):
        batch_size = x.shape[0]
        y = x.reshape(batch_size,self.size*self.size)
        for dense in self.dense:
            y = dense(y)
            y = func.relu(y,True)
        y = self.class_dense(y)
        y = func.softmax(y,dim=-1)
        return y
    
class LatentClassifier(nn.Module):
    def __init__(self,latent_channels=16,channels = [256,256,256],class_channels = 10):
        super(LatentClassifier,self).__init__()
        self.dense = nn.ModuleList()
        last_channel = latent_channels
        for channel in channels:
            dense = nn.Sequential(nn.Linear(last_channel,channel),nn.BatchNorm1d(channel))
            self.dense.append(dense)
            last_channel = channel
        self.latent_channels = latent_channels
        self.class_dense = nn.Linear(last_channel,class_channels)
        self.channels = channels
        self.class_channels = class_channels

    def forward(self,x):
        y = x
        for dense in self.dense:
            y = dense(y)
            y = func.relu(y,True)
        y = self.class_dense(y)
        y = func.softmax(y,dim=-1)
        return y