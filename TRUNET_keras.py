import tensorflow as tf
import keras
import keras.layers as layers
import keras.activations as activations
import numpy as np

import Blocks as bl


class TRU_Net(tf.keras.Model):

    def __init__(self, channels_in):


        super().__init__()

        

        #Encoder part

        self.cnn_1d_block1 = bl.FirstBlock_DSConv1d(channels_in,64,5,2)
        self.cnn_1d_block2 = bl.DepthwiseSeparableConv1d(64,128,3,1)
        self.cnn_1d_block3 = bl.DepthwiseSeparableConv1d(128,128,5,2)
        self.cnn_1d_block4 = bl.DepthwiseSeparableConv1d(128,128,3,1)
        self.cnn_1d_block5 = bl.DepthwiseSeparableConv1d(128,128,5,2)
        self.cnn_1d_block6 = bl.DepthwiseSeparableConv1d(128,128,3,2)
        

        #FGRU Layer
        self.fgru_layer = bl.FGRU_Block(64)

        #TGRU Layer
        self.tgru_layer = bl.TGRU_Block(64)

        #Decoder part
        self.trcnn_1d_block1 = bl.TrCNN_Block(64,3,2)
        self.trcnn_1d_block2 = bl.TrCNN_Block(64,5,2)
        self.trcnn_1d_block3 = bl.TrCNN_Block(64,3,1)
        self.trcnn_1d_block4 = bl.TrCNN_Block(64,5,2)
        self.trcnn_1d_block5 = bl.TrCNN_Block(64,3,1)
        self.trcnn_1d_block6 = bl.TrCNN_Block(10,5,2)




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


        #FGRU Block

        out_fgru = self.fgru_layer(out6_enc)
        print('fgru',np.shape(out_fgru))

        #TGRU Block

        out_tgru = self.tgru_layer(out_fgru)
        print('tgru',np.shape(out_tgru))

        #Decoder

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

        in6_dec = tf.concat([out1_enc,out5_dec],axis=-1)
        out6_dec = self.trcnn_1d_block6(in6_dec)
        print(np.shape(out6_dec))


        
        return out6_dec



if __name__=='__main__':
    
    TRU = TRU_Net(2)
    
    x = tf.constant(np.random.rand(686,256,2))
    print("input_shape:",tf.shape(x))
    y = TRU(x)

    total_params = TRU.count_params()
    print("total params:",total_params)
    print("output_shape:",tf.shape(y))


