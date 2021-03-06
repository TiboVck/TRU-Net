import keras
import tensorflow as tf
import keras.layers as layers
import keras.activations as activations
import numpy as np
import Utils as ut
import tensorflow_probability as tfp
tfd = tfp.distributions


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

class Last_Blocks(tf.keras.Model):

    def __init__(self,out_channels, kernel, stride):


        super().__init__()

        self.conv1d = layers.Conv1D(out_channels,kernel,stride)
        
        self.norm = tf.keras.layers.BatchNormalization()
        self.relu = layers.Activation(activations.relu)

  
  


    def call (self, x):
        
        print(np.shape(x))


        output  = self.conv1d(x)
        
        output = self.relu(self.norm(output))
        
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
        self.trcnn_1d_block6 = TrCNN_Block(10,5,2)


    def call (self, out_tgru, out1_enc, out2_enc, out3_enc, out4_enc, out5_enc, out6_enc):
        
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
        print(np.shape(out1_enc))

        in6_dec = tf.concat([out1_enc,out5_dec],axis=-1)
        out6_dec = self.trcnn_1d_block6(in6_dec)
        print(np.shape(out6_dec))

        return out6_dec

# Blocks de post-processing
# Calcul des masks ?? partir des sorties du dernier layer
 
class Calcul_Mask(tf.keras.Model):
    def __init__(self):


        super().__init__()



    def call (self, z_tf, psi_beta):
        #======Calcul sigma(ztf)============
        sigma_tf = ut.sigma(z_tf)

        #psi_beta_tf = self.psi_block_beta(phi)
        #print(np.shape(beta_tf))
        
        #======Calcul BetaTF=================
        beta_tf = ut.calcul_beta(psi_beta)
        print(np.shape(beta_tf))
        
        #Calcul des upperbound de clipping pour Beta

        clip_d = tf.divide(1, tf.abs(sigma_tf[0]-sigma_tf[1]))
        clip_n = tf.divide(1, tf.abs(sigma_tf[2]-sigma_tf[3]))

        beta_tf_clip_d = tf.clip_by_value(beta_tf,tf.ones(tf.shape(beta_tf)),clip_d)
        beta_tf_clip_n = tf.clip_by_value(beta_tf,tf.ones(tf.shape(beta_tf)),clip_n)

        
        #Calcul des masques (modules)
        mask_tfd = tf.multiply(beta_tf_clip_d, sigma_tf[0])
        mask_tfid = tf.multiply(beta_tf_clip_d, sigma_tf[1])
        mask_tfn = tf.multiply(beta_tf_clip_n, sigma_tf[2])
        mask_tfin = tf.multiply(beta_tf_clip_n, sigma_tf[3])

        return mask_tfd, mask_tfid, mask_tfn, mask_tfin

class Calcul_exp_stft(tf.keras.Model):
    def __init__(self):


        super().__init__()

        #------------Additional layer for ksi------------------------------------------------------
        self.learn_temp = layers.Dense(1,use_bias=False)


        self.norm_cosinus = tf.keras.layers.BatchNormalization()

    def call (pi_ksi,  mask_tfd, mask_tfid, mask_tfn, mask_tfin):

        tirage_gumbel = tfd.Gumbel(loc=0., scale=1.)
        g_tf = tirage_gumbel.sample(tf.shape(pi_ksi))

        arg_exp_plus1 = self.learn_temp(tf.math.log(pi_ksi) + g_tf)
        arg_exp_moins1 = self.learn_temp(tf.math.log(tf.ones(tf.shape(pi_ksi))-pi_ksi) + g_tf)


        alpha_tf_plus1 = tf.math.exp(arg_exp_plus1)
        alpha_tf_moins1 = tf.math.exp(arg_exp_moins1)
        ksi_norm = alpha_tf_moins1 + alpha_tf_plus1
        y_plus1 = tf.divide(alpha_tf_plus1,ksi_norm)
        ksi_smooth = tf.multiply(2,y_plus1) - tf.one_hot(tf.shape(y_plus1))

        #first, delta theta
        cos_delta_theta_d = self.norm_cosinus(ut.calcul_theta(mask_tfd,mask_tfid))  
        cos_delta_theta_id = self.norm_cosinus(ut.calcul_theta(mask_tfid,mask_tfd))  
        cos_delta_theta_n = self.norm_cosinus(ut.calcul_theta(mask_tfn,mask_tfin))  
        cos_delta_theta_in = self.norm_cosinus(ut.calcul_theta(mask_tfin,mask_tfn))  

        delta_theta_d = tf.math.acos(cos_delta_theta_d)
        delta_theta_id = tf.math.acos(cos_delta_theta_id)
        delta_theta_n = tf.math.acos(cos_delta_theta_n)
        delta_theta_in = tf.math.acos(cos_delta_theta_in)


        #ensuite, calcul exp(j*theta) 

        approx_theta_d = tf.multiply(ksi_smooth, delta_theta_d)
        approx_theta_id = tf.multiply(ksi_smooth, delta_theta_id)
        approx_theta_n = tf.multiply(ksi_smooth, delta_theta_n)
        approx_theta_in = tf.multiply(ksi_smooth, delta_theta_in)

        #Version smooth
        exp_jtheta_d = tf.math.cos(approx_theta_d) + tf.multiply(1j,tf.math.sin(approx_theta_d))
        exp_jtheta_id = tf.math.cos(approx_theta_id) + tf.multiply(1j,tf.math.sin(approx_theta_id))
        exp_jtheta_n = tf.math.cos(approx_theta_n) + tf.multiply(1j,tf.math.sin(approx_theta_n))
        exp_jtheta_in = tf.math.cos(approx_theta_n) + tf.multiply(1j,tf.math.sin(approx_theta_in))

        return exp_jtheta_d, exp_jtheta_id, exp_jtheta_n, exp_jtheta_in


