from locale import ABDAY_1
import tensorflow as tf
import keras
import keras.layers as layers
import keras.activations as activations
import numpy as np
import tensorflow_probability as tfp

import Blocks as bl
import Utils as ut


class TRU_Net(tf.keras.Model):

'''

Architecture globale du TRU-Net.

Plusieurs variantes :

1) 2 channels en entrée : logmel spectrogram, et PCEN spectrogram
En sortie : module du mask dans l'espace temps-fréquence, on prend la phase de l'entrée.
Le module du mask est estimé avec beta_tf et les deux z_tfn. 
Loss uni-échelle que sur le module.

2) 2 channels en entrée : logmel spectrogram, et PCEN spectrogram
En sortie : module du mask dans l'espace temps-fréquence, on prend la phase de l'entrée.
Le module du mask est estimé avec beta_tf et les deux z_tfn. 
Loss multi-échelle que sur le module.

3) 2 channels en entrée : logmel spectrogram, et PCEN spectrogram
En sortie : les modules des 2 masks dans l'espace temps-fréquence, on prend la phase de l'entrée.
Les modules des mask sont estimés avec beta_tf, les deux z_tfn, les deux z_tfr.
Loss multi-échelle que sur le module.

4) 4 channels en entrée : logmel spectrogram, le PCEN spectrogram, les parties réelles et imaginaires des phases de la phase démodulé : https://www.zhinst.com/europe/en/resources/principles-of-lock-in-detection.
En sortie : les modules des 3 masks dans l'espace temps-fréquence.
Le modules des masks sont estimés avec beta_tf, les deux z_tfn, les deux z_tfr. Les phases avec ksi_tf.
Loss multi-échelle que sur le module et sur le signal.

Pour ksi: SOIT ON APPREND SEUELEMENT LES PROBAS DE +1 ET -1, SOIT ON APPREND ÉGALEMENT LA TEMPÉRATURE DE L'ESTIMATEUR

Pour estimer on fait un argmax mais on back propage grâce à la fonction y

Test de la dérevrberation : 
-wham! dataset 
https://wham.whisper.ai

-CHiME2 test set.
https://catalog.ldc.upenn.edu/LDC2017S10.



'''


    def __init__(self, channels_in):


        super().__init__()

        

#------------Encoder part----------------------------------------------------------------
        self.trunet_encoder = bl.TRUNet_Encoder(channels_in)

#------------FGRU Layer--------------------------------------------------------------------
        self.fgru_layer = bl.FGRU_Block(64)

#------------TGRU Layer--------------------------------------------------------------------------------
        self.tgru_layer = bl.TGRU_Block(64)

#------------Decoder part------------------------------------------------------------------
        self.trunet_decoder= bl.TRUNet_Decoder(64)
        
#------------Last Layer-------------------------------------------------------------------
        #self.trcnn_1d_block6 = bl.TrCNN_Block(10,5,2)
        self.psi_block_ztf = bl.Last_TrCNN_Block(5,5,2)
        self.psi_block_beta  = bl.TrCNN_Block(5,5,2)

#------------Additional layer for ksi------------------------------------------------------
        self.gumbel_sample = layers.Dense(1,use_bias=False)




    def call (self, input):

        print(np.shape(input))

#-------Encoder-----------------------------------------------------------------------------------
        
        out1_enc, out2_enc, out3_enc, out4_enc, out5_enc, out6_enc = self.trunet_encoder(input)

#-------FGRU Block--------------------------------------------------------------------------------

        out_fgru = self.fgru_layer(out6_enc)
        print('fgru',np.shape(out_fgru))

#-------TGRU Block--------------------------------------------------------------------------------

        out_tgru = self.tgru_layer(out_fgru)
        print('tgru',np.shape(out_tgru))

#-------Decoder-----------------------------------------------------------------------------------
   
        out5_dec = self.trunet_decoder(out_tgru, out2_enc, out3_enc, out4_enc, out5_enc, out6_enc)
        print(np.shape(out5_dec))

#-------Last layer of the decoder-----------------------------------------------------------------
        phi = tf.concat([out1_enc,out5_dec],axis=-1)
        z_tf = self.psi_block_ztf(phi)
        print(np.shape(z_tf))

#-------Post-processing----------------------------------------------------------------------------

        #Calcul sigma(ztf)
        sigma_tf = ut.sigma(z_tf[:,:,:4])

        #psi_beta_tf = self.psi_block_beta(phi)
        #print(np.shape(beta_tf))
        
        #Calcul BetaTF
        beta_tf = ut.calcul_beta(z_tf[:,:,4])
        print(np.shape(beta_tf))
            #calcul des upperbound de clipping pour Beta
        clip_d = tf.divide(1, tf.abs(sigma_tf[0]-sigma_tf[1]))
        
        clip_n = tf.divide(1, tf.abs(sigma_tf[2]-sigma_tf[3]))

        beta_tf_clip_d = tf.clip_by_value(beta_tf,tf.ones(tf.shape(beta_tf)),clip_d)
        beta_tf_clip_n = tf.clip_by_value(beta_tf,tf.ones(tf.shape(beta_tf)),clip_n)

        
        #Calcul des masques (modules)
        mask_tfd = tf.multiply(beta_tf_clip_d, sigma_tf[0])
        mask_tfid = tf.multiply(beta_tf_clip_d, sigma_tf[1])
        mask_tfn = tf.multiply(beta_tf_clip_n, sigma_tf[2])
        mask_tfin = tf.multiply(beta_tf_clip_n, sigma_tf[3])

        #Calcul des phases des masques 
        alpha_tf1 = z_tf[:,:,5]
        alpha_tfm1 = z_tf[:,:,6]


        #first, delta theta
        delta_theta_d = ut.calcul_theta(mask_tfd,mask_tfid)
        delta_theta_n = ut.calcul_theta(mask_tfn,mask_tfin)  

        #ensuite, calcul exp(j*theta)
        #d'abord calcul des estimations ksi=+/-1

        g1 = tfp.distributions.Gumbel(loc=0., scale=1.)
        gm1 = tfp.distributions.Gumbel(loc=0., scale=1.)
        #MODIFIE dimension des samples gumbel
        a1 = self.gumbel_sample(tf.math.log(alpha_tf1)+g1)
        am1 = self.gumbel_sample(tf.math.log(alpha_tfm1)+gm1)
        y1 = tf.divide(a1,a1+am1)
        ym1 = 1-y1


        #version smooth
        e_jtheta = tf.cos(tf.multiply(y1-ym1))
        
        return mask_tfd



if __name__=='__main__':
    
    TRU = TRU_Net(4)
    
    x = tf.constant(np.random.rand(1,256,4))
    print("input_shape:",tf.shape(x))
    y = TRU(x)

    total_params = TRU.count_params()
    print("total params:",total_params)
    print("output_shape:",tf.shape(y))


