import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# ce qu'il faut dans keras
import keras
import tensorflow as tf
import keras.layers as layers
import keras.activations as activations

""" class StandardConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(StandardConv1d, self).__init__()
        self.StandardConv1d = nn.Sequential(
            nn.Conv1d(in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = stride //2),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.StandardConv1d(x) """


class DepthwiseSeparableConv1d(tf.keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size, stride):


        super().__init__()

        shape = ([None, in_channels])
        self.kernel = kernel_size
        self.l1= tf.keras.Sequential()
        self.conv1d_1 = tf.keras.layers.Conv1D(out_channels , 1, input_shape = shape)
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = layers.Activation(activations.relu)

        self.conv1d_2 = layers.Conv1D(out_channels,kernel_size, input_shape =(None,out_channels),strides = stride, groups = out_channels)
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.relu2 = layers.Activation(activations.relu)


    def call (self, x):

        x = self.l1(x)
        x = self.conv1d_1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        print(np.shape(x))

        pad = np.zeros((np.shape(x)[0],self.kernel//2,np.shape(x)[2]))
        pad_x = np.concatenate((pad,x.numpy(),pad),axis=1)
        y = self.conv1d_2(pad_x)
        y = self.norm2(y)
        y = self.relu2(y)

        return y









#########

class GRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, bidirectional):
        super(GRUBlock, self).__init__()
        self.GRU = nn.GRU(in_channels, hidden_size, bidirectional=bidirectional)
        
        self.conv = nn.Sequential(nn.Conv1d(hidden_size * (2 if bidirectional==True else 1), out_channels, kernel_size = 1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True))

    def forward(self, x):
        output,h = self.GRU(x)
        output = output.transpose(1,2)
        output = self.conv(output)
        return output

class FirstTrCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(FirstTrCNN, self).__init__()
        self.FirstTrCNN = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(in_channels = out_channels,
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding = stride//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.FirstTrCNN(x)


class TrCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(TrCNN, self).__init__()
        self.TrCNN = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(in_channels = out_channels,
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding = stride//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x1,x2):
        diffY = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2, 0, 0])
        x = torch.cat((x1,x2),1)
        output = self.TrCNN(x)
        return output

class LastTrCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(LastTrCNN, self).__init__()
        self.LastTrCNN = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(in_channels = out_channels,
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding=stride//2))

    def forward(self,x1,x2):
        diffY = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2, 0, 0])
        x = torch.cat((x1,x2),1)
        output = self.LastTrCNN(x)
        return output