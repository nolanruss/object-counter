import keras
import keras.backend as K
from keras.objectives import *
from keras.losses import binary_crossentropy
import keras.backend as K
import tensorflow as tf

def mean_categorical_crossentropy(y_true, y_pred):
    if K.image_data_format() == 'channels_last':
        loss = K.mean(keras.losses.categorical_crossentropy(y_true, y_pred), axis=[1, 2])
    elif K.image_data_format() == 'channels_first':
        loss = K.mean(keras.losses.categorical_crossentropy(y_true, y_pred), axis=[2, 3])
    return loss

def flatten_categorical_crossentropy(classes):
    def f(y_true, y_pred):
        y_true = K.reshape(y_true, (classes, -1))
        y_pred = K.reshape(y_pred, (-1, classes))
        return keras.losses.categorical_crossentropy(y_true, y_pred)
    return f

def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = cross_entropy

    return cross_entropy_mean

def binary_cross(y_true, y_pred):
    '''binary_cross
    Description: Mean binary cross entropy.
    Args: y_true - The expected result
          y_pred - The predicted result
    '''
    loss = K.mean(binary_crossentropy(y_true=y_true, y_pred=y_pred))
    return loss