import keras
import tensorflow as tf
import keras.layers as layers
import keras.activations as activations
import numpy as np


def calcul_beta(psi_beta_tf):
    return 1+tf.softplus(psi_beta_tf)

def sigma(psi_ztf):
    shape = tf.shape(psi_ztf)
    sigma = tf.zeros(shape)
    

    sigma_d = tf.reshape( tf.divide(1,1 + tf.exp(-(psi_ztf[:,:,0]-psi_ztf[:,:,1]))),[shape[0],shape[1],1])
    sigma_id = tf.reshape(tf.divide(1,1 + tf.exp(-(psi_ztf[:,:,1]-psi_ztf[:,:,0]))),[shape[0],shape[1],1])
    sigma_n= tf.reshape(tf.divide(1,1 + tf.exp(-(psi_ztf[:,:,2]-psi_ztf[:,:,3]))),[shape[0],shape[1],1])
    sigma_in = tf.reshape(tf.divide(1,1 + tf.exp(-(psi_ztf[:,:,3]-psi_ztf[:,:,2]))),[shape[0],shape[1],1])

    sigma = tf.concat([sigma_d,sigma_id,sigma_n,sigma_in],axis=2)
    print(tf.shape(sigma))
    return tf.transpose(sigma,perm=[2,0,1])


def calcul_theta(mask_tf,mask_tfi):

    return tf.divide(1+tf.pow(mask_tf,2)+tf.pow(mask_tfi,2),tf.multiply(2,mask_tf))