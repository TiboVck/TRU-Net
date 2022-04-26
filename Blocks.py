import keras
import tensorflow as tf
import keras.layers as layers
import keras.activations as activations
import numpy as np


# 1D-CNN Blocks for the encoder
 
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

# FGRU block, along frequency axis

class FGRU_Block(tf.keras.Model):

    def __init__(self,out_channels):


        super().__init__()

        gru = layers.GRU(64,return_sequences=True)

        self.gru_layer = layers.Bidirectional(gru,merge_mode="sum")
        
        self.norm = tf.keras.layers.BatchNormalization()
        self.relu = layers.Activation(activations.relu)

        self.conv1d_ptwise = layers.Conv1D(out_channels,1)
    

  


    def call (self, x):
        
        #print(np.shape(x))


        output  = self.gru_layer(x)
        #print(np.shape(output))
        #out=tf.transpose(output, perm=[0,2,1])
        output = self.conv1d_ptwise(output)
        
        #print(np.shape(output))
        output = self.norm(self.relu(output))

        return output

# TGRU block, along time axis

class TGRU_Block(tf.keras.Model):

    def __init__(self,out_channels):


        super().__init__()

        self.gru = layers.GRU(128,return_sequences=True)

        
        
        self.norm = tf.keras.layers.BatchNormalization()
        self.relu = layers.Activation(activations.relu)

        self.conv1d_ptwise = layers.Conv1D(out_channels,1)
        

        


    def call (self, x):
        
        #print(np.shape(x))


        output = self.gru(x)
        
        # print(np.shape(output))
        
        output = self.conv1d_ptwise(output)
        

        # print(np.shape(output))
        output = self.norm(self.relu(output))

        return output

# Tr-CNN Blocks for the decoder

class TrCNN_Block(tf.keras.Model):

    def __init__(self,out_channels, kernel, stride):


        super().__init__()

        self.conv1d_ptwise = layers.Conv1D(64,1)
        
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = layers.Activation(activations.relu)

        self.TransposeConv = layers.Conv1DTranspose(out_channels,kernel, stride,padding='same')
    
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.relu2 = layers.Activation(activations.relu)
  


    def call (self, x):
        
        print(np.shape(x))


        output  = self.conv1d_ptwise(x)
        
        
        
        output = self.relu1(self.norm1(output))
        print(np.shape(output))
        output  = self.TransposeConv(output)


        print(np.shape(output))
        output = self.norm2(self.relu2(output))

        return output

class Last_TrCNN_Block(tf.keras.Model):

    def __init__(self,out_channels, kernel, stride):


        super().__init__()

        self.conv1d_ptwise = layers.Conv1D(64,1)
        
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = layers.Activation(activations.relu)

        self.TransposeConv = layers.Conv1DTranspose(out_channels,kernel, stride,padding='same')
    
       
  


    def call (self, x):
        
        print(np.shape(x))


        output  = self.conv1d_ptwise(x)
        
        
        
        output = self.relu1(self.norm1(output))
        print(np.shape(output))
        output  = self.TransposeConv(output)


        print(np.shape(output))
       
        return output