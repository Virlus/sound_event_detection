import torch
import torch.nn as nn
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = (3, 3), pool_size = (2, 2)):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding = (kernel_size[0]//2, kernel_size[1]//2))
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


def linear_softmax_pooling(x):
    return (x ** 2).sum(1) / x.sum(1)


class Crnn(nn.Module):
    def __init__(self, num_freq, num_class, channels=(1, 16, 32, 64, 128), pool_sizes=(2, 2, 2, 2), gru_layers=3):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     num_class: int, the number of output classes
        ##############################
        super().__init__()
        self.hidden_dim = int(num_freq * channels[-1] // np.prod(pool_sizes))
        self.num_freq = num_freq
        self.num_class = num_class
        self.channels = channels
        self.pool_sizes = pool_sizes
        
        self.bn = nn.BatchNorm2d(channels[0])
        self.cnn = nn.Sequential(
            ConvBlock(channels[0], channels[1], pool_size=(1, pool_sizes[0])),
            ConvBlock(channels[1], channels[2], pool_size=(1, pool_sizes[1])),
            ConvBlock(channels[2], channels[3], pool_size=(1, pool_sizes[2])),
            ConvBlock(channels[3], channels[4], pool_size=(1, pool_sizes[3]))
        )
        
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, gru_layers, bidirectional=True, batch_first=True)
        self.fc_layer = nn.Linear(2 * self.hidden_dim, num_class)
        self.sigmoid = nn.Sigmoid()

    def detection(self, x):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_prob: [batch_size, time_steps, num_class]
        ##############################
        x = self.bn(x.unsqueeze(1))
        x = self.cnn(x)
        x = self.rnn(x.permute(0, 2, 1, 3).flatten(2))[0]
        x = self.fc_layer(x)
        x = self.sigmoid(x)
        return x
        

    def forward(self, x):
        frame_prob = self.detection(x)  # (batch_size, time_steps, class_num)
        clip_prob = linear_softmax_pooling(frame_prob)  # (batch_size, class_num)
        '''(samples_num, feature_maps)'''
        return {
            'clip_prob': clip_prob, 
            'frame_prob': frame_prob
        }
