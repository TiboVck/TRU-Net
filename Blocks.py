import keras
import tensorflow as tf
import keras.layers as layers
import keras.activations as activations
import numpy as np

class DepthwiseSeparableConv1d(tf.keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size, stride):


        super().__init__()

        shape = ([None, in_channels])
 
        self.conv1d_ptwise = layers.Conv1D(in_channels,1)

        self.norm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = layers.Activation(activations.relu)

        self.conv1d_dpwise = layers.DepthwiseConv1D(kernel_size,strides=stride,padding = 'same',data_format='channels_last' , depth_multiplier=int(out_channels//in_channels),activation=activations.relu)

        self.norm2 = tf.keras.layers.BatchNormalization()
        self.relu2 = layers.Activation(activations.relu)


    def call (self, x):
        #print('dp',np.shape(x))
        x = self.conv1d_ptwise(x)

        x = self.norm1(x)
        x = self.relu1(x)

        #print('dp2',np.shape(x))

        y = self.conv1d_dpwise(x)
       # print('dp3',np.shape(y))

        y = self.norm2(y)
        y = self.relu2(y)

        return y


class FirstBlock_DSConv1d(tf.keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size, stride):


        super().__init__()

        shape = ([None, in_channels])
 
        

        self.conv1d = layers.Conv1D(out_channels,kernel_size,strides=stride,padding = 'same',)

        self.norm = tf.keras.layers.BatchNormalization()
        self.relu = layers.Activation(activations.relu)


    def call (self, x):
        #print('dp',np.shape(x))
        
        y = self.conv1d(x)
        #print('dp3',np.shape(y))

        y = self.norm(y)
        y = self.relu(y)

        return y





class Encoder_TRU_Net(tf.keras.Model):

    def __init__(self, channels_in):


        super().__init__()

        



        self.cnn_1d_block1 = FirstBlock_DSConv1d(channels_in,64,5,2)
        self.cnn_1d_block2 = DepthwiseSeparableConv1d(64,128,3,1)
        self.cnn_1d_block3 = DepthwiseSeparableConv1d(128,128,5,2)
        self.cnn_1d_block4 = DepthwiseSeparableConv1d(128,128,3,1)
        self.cnn_1d_block5 = DepthwiseSeparableConv1d(128,128,5,2)
        self.cnn_1d_block6 = DepthwiseSeparableConv1d(128,128,3,2)
        
        self.faux_layer = layers.Conv1D(257,17)

        self.norm = tf.keras.layers.BatchNormalization()


    def call (self, x):

        print(np.shape(x))
        
        x = self.cnn_1d_block1(x)
        print(np.shape(x))
        x = self.cnn_1d_block2(x)
        print(np.shape(x))
        x = self.cnn_1d_block3(x)
        print(np.shape(x))
        x = self.cnn_1d_block4(x)
        print(np.shape(x))
        x = self.cnn_1d_block5(x)
        print(np.shape(x))
        x = self.cnn_1d_block6(x)
        print(np.shape(x))
        x = self.faux_layer(x)
        print(np.shape(x))
        x = tf.transpose(x,[1,0,2])
        print(np.shape(x))
        out = x[0]


        
        return out