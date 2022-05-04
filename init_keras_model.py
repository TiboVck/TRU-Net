# -*- coding: utf-8 -*-

# Authors: T. Vacek

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
from machine_learning.keras_custom.layers import TFMasking, GlobalGain


def init_keras_model(model_name, input_shapes, output_shapes,
                     mask_clipping_value):

    if model_name == 'TRU-Net_learn_temp':

       model_input = keras.models.Input(shape=input_shapes[0][1:])

       # --- Encoder

       out1_enc, out2_enc, out3_enc, out4_enc, out5_enc, out6_enc = bl.TRUNet_Encoder(channels_in)(model_input)

       # --- FGRU Layer
       out_fgru = bl.FGRU_Block(64)(out6_enc)

       # --- TGRU Layer
       out_tgru = bl.TGRU_Block(64)(out_fgru)

       # --- Decoder
       phi = bl.TRUNet_Decoder(64)(out_tgru, out1_enc, out2_enc, out3_enc, out4_enc, out5_enc, out6_enc)

       # --- Last layers
       psi_beta = bl.Last_Blocks(4,10,1)
       z_tf = bl.Last_Blocks(4,10,1)
       pi_ksi = bl.Last_Blocks(1,10,1)

       # --- Post-processing
       #Calcul des masques (modules)
       mask_tfd, mask_tfid, mask_tfn, mask_tfin = bl.Calcul_Mask()(z_tf,psi_beta)
       

       #Calcul de l'exp avec la phase
       exp_jtheta_d, exp_jtheta_id, exp_jtheta_n, exp_jtheta_in = bl.Calcul_exp_stft()(pi_ksi, mask_tfd, mask_tfid, mask_tfn, mask_tfin)

       #Reconstruction stft pour loss signal

       STFT_d = tf.multiply(mask_tfd,exp_jtheta_d)
       STFT_id = tf.multiply(mask_tfid,exp_jtheta_id)
       STFT_n = tf.multiply(mask_tfn,exp_jtheta_n)
       STFT_in = tf.multiply(mask_tfin,exp_jtheta_in)
       STFT_r =  STFT_in - STFT_d
                
       return STFT_d, STFT_n, STFT_r

    else:

        raise TypeError('Unknown model ' + model_name)

    return model
