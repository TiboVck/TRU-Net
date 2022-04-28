# -*- coding: utf-8 -*-

# Authors: T. Vacek

import tensorflow.keras as keras

from machine_learning.keras_custom.layers import TFMasking, GlobalGain


def init_keras_model(model_name, input_shapes, output_shapes,
                     mask_clipping_value):

    if model_name == 'rnn_dnn_0':
        # 3 hidden layers 256


        


    else:

        raise TypeError('Unknown model ' + model_name)

    return model
