from cmath import pi
from locale import ABDAY_1
import tensorflow as tf
import keras
import keras.layers as layers
import keras.activations as activations
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions

import Blocks as bl
import Utils as ut


class TRU_Net_learn_temp(tf.keras.Model):
        """

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
        input size : 19624192 x 80"""


        def __init__(self, channels_in):


                super().__init__()
        #------------Parameters-------------------------------------------------------------------
                self.dist = tfd.Gumbel(loc=0., scale=3.)
        #------------Encoder part-----------------------------------------------------------------
                self.trunet_encoder = bl.TRUNet_Encoder(channels_in)

        #------------FGRU Layer--------------------------------------------------------------------
                self.fgru_layer = bl.FGRU_Block(64)

        #------------TGRU Layer--------------------------------------------------------------------
                self.tgru_layer = bl.TGRU_Block(64)

        #------------Decoder part------------------------------------------------------------------
                self.trunet_decoder= bl.TRUNet_Decoder(64)

                
        #------------Last Layers-------------------------------------------------------------------
                
                self.block_z_tfk = bl.Last_Blocks(4,10,1)
                self.block_beta_tf  = bl.Last_Blocks(4,10,1)
                self.block_ksi_tf = bl.Last_Blocks(1,10,1)



        #------------Post-processing---------------------------------------------------------------
                self.calcul_mask = bl.Calcul_Mask()
                self.calcul_exp_complex = bl.Calcul_exp_stft()




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
        
                phi = self.trunet_decoder(out_tgru, out1_enc, out2_enc, out3_enc, out4_enc, out5_enc, out6_enc)
                print(np.shape(phi),"HAAAAAAA")

        #-------Last layers to output beta, z, and ksi-----------------------------------------------------------------
                
                psi_beta = self.block_beta_tf(phi)
                z_tf = self.block_z_tfk(phi)
                pi_ksi = self.block_ksi_tf(phi)

        #-------Post-processing----------------------------------------------------------------------------

                #Calcul des masques (modules)
                mask_tfd, mask_tfid, mask_tfn, mask_tfin = self.calcul_mask(z_tf,psi_beta)

                #Calcul de ksi

                exp_jtheta_d, exp_jtheta_id, exp_jtheta_n, exp_jtheta_in = self.calcul_exp_complex(pi_ksi, mask_tfd, mask_tfid, mask_tfn, mask_tfin)

                #Reconstruction stft pour loss signal

                STFT_d = tf.multiply(mask_tfd,exp_jtheta_d)
                STFT_id = tf.multiply(mask_tfid,exp_jtheta_id)
                STFT_n = tf.multiply(mask_tfn,exp_jtheta_n)
                STFT_in = tf.multiply(mask_tfin,exp_jtheta_in)
                STFT_r =  STFT_in - STFT_d
                
                return STFT_d, STFT_n, STFT_r



if __name__=='__main__':
    
    TRU = TRU_Net_learn_temp(4)
    
    x = tf.constant(np.random.rand(1,256,4))
    print("input_shape:",tf.shape(x))
    y = TRU(x)

    total_params = TRU.count_params()
    print("total params:",total_params)
    print("output_shape:",tf.shape(y))


