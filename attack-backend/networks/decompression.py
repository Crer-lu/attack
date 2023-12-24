from torch import nn as nn
from torch.nn import functional as func

class DecompressionNetwork(nn.Module):
    def __init__(self,size = 28,channels = [256,256,256],latent_channels = 16):
        super(DecompressionNetwork,self).__init__()
        self.dense = nn.ModuleList()
        last_channel = latent_channels
        for channel in channels:
            dense = nn.Sequential(nn.Linear(last_channel,channel),nn.BatchNorm1d(channel))
            self.dense.append(dense)
            last_channel = channel
        self.out_dense = nn.Linear(last_channel,size*size)
        self.size = size
        self.channels = channels
        self.latent_channels = latent_channels

    def forward(self,x):
        batch_size = x.shape[0]
        y = x
        for dense in self.dense:
            y = dense(y)
            y = func.relu(y,True)
        y = self.out_dense(y)
        y = func.tanh(y)
        y = y.reshape(batch_size,self.size,self.size)
        return y