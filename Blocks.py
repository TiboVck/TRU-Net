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


class TRUNet_Encoder(tf.keras.Model):

    def __init__(self,channels_in):

        super().__init__()

                
        self.cnn_1d_block1 = FirstBlock_DSConv1d(channels_in,64,5,2)
        self.cnn_1d_block2 = DepthwiseSeparableConv1d(64,128,3,1)
        self.cnn_1d_block3 = DepthwiseSeparableConv1d(128,128,5,2)
        self.cnn_1d_block4 = DepthwiseSeparableConv1d(128,128,3,1)
        self.cnn_1d_block5 = DepthwiseSeparableConv1d(128,128,5,2)
        self.cnn_1d_block6 = DepthwiseSeparableConv1d(128,128,3,2)
    
       
  


    def call (self, input):
        
        print(np.shape(input))

        #Encoder
        out1_enc = self.cnn_1d_block1(input)
        print(1,np.shape(out1_enc))

        out2_enc = self.cnn_1d_block2(out1_enc)
        print(2,np.shape(out2_enc))

        out3_enc = self.cnn_1d_block3(out2_enc)
        print(3,np.shape(out3_enc))

        out4_enc = self.cnn_1d_block4(out3_enc)
        print(4,np.shape(out4_enc))

        out5_enc = self.cnn_1d_block5(out4_enc)
        print(5,np.shape(out5_enc))

        out6_enc = self.cnn_1d_block6(out5_enc)
        print(6,np.shape(out6_enc))


        return out1_enc, out2_enc, out3_enc, out4_enc, out5_enc, out6_enc



class TRUNet_Decoder(tf.keras.Model):

    def __init__(self,channels_in):

        super().__init__()

                
        self.trcnn_1d_block1 = TrCNN_Block(channels_in,3,2)
        self.trcnn_1d_block2 = TrCNN_Block(64,5,2)
        self.trcnn_1d_block3 = TrCNN_Block(64,3,1)
        self.trcnn_1d_block4 = TrCNN_Block(64,5,2)
        self.trcnn_1d_block5 = TrCNN_Block(64,3,1)


    def call (self, out_tgru, out2_enc, out3_enc, out4_enc, out5_enc, out6_enc):
        
        in1_dec = tf.concat([out6_enc,out_tgru],axis=-1)
        out1_dec = self.trcnn_1d_block1(in1_dec)
        print('1D',np.shape(out1_dec))

        in2_dec = tf.concat([out5_enc,out1_dec],axis=-1)
        out2_dec = self.trcnn_1d_block2(in2_dec)
        print(np.shape(out2_dec))

        in3_dec = tf.concat([out4_enc,out2_dec],axis=-1)
        out3_dec = self.trcnn_1d_block3(in3_dec)
        print(np.shape(out3_dec))

        in4_dec = tf.concat([out3_enc,out3_dec],axis=-1)
        out4_dec = self.trcnn_1d_block4(in4_dec)
        print(np.shape(out4_dec))

        in5_dec = tf.concat([out2_enc,out4_dec],axis=-1)
        out5_dec = self.trcnn_1d_block5(in5_dec)
        print(np.shape(out5_dec))

        return out5_dec