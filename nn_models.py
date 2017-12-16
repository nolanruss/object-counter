from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.initializers import Constant
from keras.layers import Add, Activation, Flatten
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.models import Input, Model
from keras.applications.vgg16 import VGG16
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from losses import binary_cross
import keras.backend as K
import losses
import metrics
from dataset import quick_dataset, load_images

def fcn_block(nb_filters, block_nb, input_shape=None):
    '''fcn_block
    
    Description: Creates an FCN block consisting of 2 or 3 (depending upon the layer) convolutional layers, 
    followed by a maxpooling layer.
    
    Returns a model consisting of the convolution layers
    '''
    inpt = Input(shape=input_shape)
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name='block'+str(block_nb)+'_conv1')(inpt)
    x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name='block'+str(block_nb)+'_conv2')(x)
    if block_nb > 2:
        x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name='block'+str(block_nb)+'_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block'+str(block_nb)+'_pool')(x)
    model = Model(inputs=inpt, outputs=x, name='block'+str(block_nb))
    return model

def load_fcn_weights(model, wts_model=None):
    '''load_fcn_weights
    
    Description: Loads either the weights of the provided model, or the VGG16 weights into the provided model.
        The loading is accomplished by iterating through every layer in the FCN blocks, matching layer names
        in the provided wts_model, and loading them into the model.
    '''
    if wts_model == None:
        wts_model = keras.applications.vgg16.VGG16(weights='imagenet')
    weights = wts_model.get_weights()
    for lyr0 in model.layers:
        for lyr1 in model.layers:
            try:
                for lyr2 in lyr1.layers:
                    ld_weights(lyr2, lyr0)
                    #print(lyr2.name)
            except:
                ld_weights(lyr1, lyr0)
                #print(lyr1.name)
                
def ld_weights(l1, l2):
    '''ld_weights
    Description: Loads weights from l2 into l1 for matching layer names.
    '''
    if(l1.name == l2.name):
        print(l1.name, l2.name, 'weights loaded')
        l1.set_weights(l2.get_weights())
           
def upsample_block1(nb_classes, block_nb, factor=0, input_shape=None):
    '''upsample_block1
    
    Description:  Adds a scoring convolutional layer for classification of layer activations according to class. If an 
        upsampling factor is provided, the scoring layer is upsampled.'''
    inpt = Input(shape=input_shape)
    x = Conv2D(nb_classes, (1, 1), activation='relu', padding='same', name='block'+str(block_nb)+'_score')(inpt)
    if factor > 0:
        x = UpSampling2D(size=(factor, factor), name='block'+str(block_nb)+'_upscore')(x)
    model = Model(inputs=inpt, outputs=x, name='block'+str(block_nb)+'_upsample')
    return model

def upsample_block(nb_classes, block_nb, factor=0, input_shape=None):
    '''upsample_block
    
    Description:  Adds a scoring convolutional layer for classification of layer
        activations according to class. If an upsampling factor is provided, the
        scoring layer is upsampled using anipConv2DTranspose layer.'''
    
    inpt = Input(shape=input_shape)
    x = Conv2D(nb_classes, (1, 1), activation='relu', padding='same',
               name='block'+str(block_nb)+'_score')(inpt)
    if factor > 0:
        x = Conv2DTranspose(filters=nb_classes, 
                kernel_size=(factor*2, factor*2),
                strides=(factor, factor),
                padding='same',
                activation='linear',
                kernel_initializer='he_normal',
                name='block'+str(block_nb)+'_upscore')(x)
    model = Model(inputs=inpt, outputs=x, name='block'+str(block_nb)+'_upsample_factor'+str(factor))
    return model

def bilinear_upsample_block(nb_classes, block_nb, factor=0, input_shape=None):
    '''upsample_block
    
    Description:  Adds a scoring convolutional layer for classification of layer
        activations according to class. If an upsampling factor is provided, the
        scoring layer is upsampled using anipConv2DTranspose layer.'''
    
    inpt = Input(shape=input_shape)
    x = Conv2D(nb_classes, (1, 1), activation='relu', padding='same',
               name='block'+str(block_nb)+'bilinear_score')(inpt)
    if factor > 0:
        x = Conv2DTranspose(filters=nb_classes, 
                kernel_size=(factor*2, factor*2),
                strides=(factor, factor),
                padding='same',
                activation='linear',
                kernel_initializer=Constant(bilinear_upsample_weights(factor=8, number_of_classes=nb_classes)),
                name='block'+str(block_nb)+'bilinear_upscore')(x)
    model = Model(inputs=inpt, outputs=x, name='block'+str(block_nb)+'bilinear_upsample')
    return model

def freeze_weights(model, mode='freeze'):
    '''freeze_weights
    
    Description: Freezes or unfreezes all weights within a block model to allow layers to be trained.
    '''
    if mode == 'freeze':
        for lyr1 in model.layers:
            try:
                for lyr2 in lyr1.layers:
                    lyr2.trainable=False
            except:
                lyr1.trainable = False
    elif mode == 'unfreeze':
        print('Unfreezing weights...')
        for lyr1 in model.layers:
            try:
                for lyr2 in lyr1.layers:
                    lyr2.trainable=True
            except:
                lyr1.trainable = True

def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = factor*2 - factor%2
    weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes),
                       dtype=np.float32)
    upsample_kernel = upsample_filt(filter_size)
    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights
                
def build_fcn1(copy_model=None, nb_classes=1):
    '''biod_fcn
    
    Description: Follows the initial VGG16 architecture by combining 4 blocks of convolutional/max pooling VGG layers.
        It then '''
    inpt = Input(shape=(None, None,3))
    
    # Create the first 4 FCN convolutional blocks
    x = fcn_block(64, 1, input_shape=(None,None,3))(inpt) # Block 1
    x = fcn_block(128, 2, input_shape=(None,None,64))(x)  # Block 2
    x = fcn_block(256, 3, input_shape=(None,None,128))(x) # Block 3
    x = fcn_block(512, 4, input_shape=(None,None,256))(x) # Block 4
    
    # Split the model into a DAG
    x_fork = x
    
    # Create an upsampling block for the block 4 max pooling layer
    upsamp = upsample_block1(input_shape=(None,None,512), nb_classes=nb_classes, block_nb=4, factor=0)
    
    # Unite the upsampling block with the block 4 max pooling layer
    x_fork = upsamp(x_fork)
    
    # Continue building the model by adding a 5th convolutional block after block 4
    x = fcn_block(512, 5, input_shape=(None,None,512))(x) # Block 5
    
    # Upsample the block 5 maxpooling output by a factor of 2
    x = upsample_block1(block_nb=5, factor=2, input_shape=(None, None, 512), nb_classes=nb_classes)(x)
    
    # Add the block 4 and block 5 upsample scoring layer predictions (FCN 16s)
    x = Add(name='add_layer')([x, x_fork])
    
    # Apply a transpose convolution to the combined scoring layers to generate the segmentation map of the same
    # dimensions as the input
    x = Conv2DTranspose(filters=nb_classes, 
                    kernel_size=(32, 32),
                    strides=(16, 16),
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    name='deconv_layer')(x)
    model = Model(inputs=inpt, outputs=x) # Create the model
    load_fcn_weights(model, copy_model) # Load weights
    return model

def build_fcn_bilinear(copy_model=None, nb_classes=1):
    '''biod_fcn
    
    Description: Follows the initial VGG16 architecture by combining 4 blocks of convolutional/max pooling VGG layers.
        It then '''
    inpt = Input(shape=(None, None,3))
    
    # Create the first 4 FCN convolutional blocks
    x = fcn_block(64, 1, input_shape=(None,None,3))(inpt) # Block 1
    x = fcn_block(128, 2, input_shape=(None,None,64))(x)  # Block 2
    x = fcn_block(256, 3, input_shape=(None,None,128))(x) # Block 3
    x = fcn_block(512, 4, input_shape=(None,None,256))(x) # Block 4
    
    # Split the model into a DAG
    x_fork = x
    
    # Create an upsampling block for the block 4 max pooling layer
    upsamp = upsample_block(input_shape=(None,None,512), nb_classes=nb_classes, block_nb=4, factor=0)
    
    # Unite the upsampling block with the block 4 max pooling layer
    x_fork = upsamp(x_fork)
    
    # Continue building the model by adding a 5th convolutional block after block 4
    x = fcn_block(512, 5, input_shape=(None,None,512))(x) # Block 5
    
    # Upsample the block 5 maxpooling output by a factor of 2
    x = upsample_block(block_nb=5, factor=2, input_shape=(None, None, 512), nb_classes=nb_classes)(x)
    
    # Add the block 4 and block 5 upsample scoring layer predictions (FCN 16s)
    x = Add(name='add_layer')([x, x_fork])
    
    # Apply a transpose convolution to the combined scoring layers to generate the segmentation map of the same
    # dimensions as the input
    x = Conv2DTranspose(filters=nb_classes, 
                    kernel_size=(32, 32),
                    strides=(16, 16),
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    name='deconv_layer')(x)
    model = Model(inputs=inpt, outputs=x) # Create the model
    load_fcn_weights(model, copy_model) # Load weights
    return model

def build_fcn_bilinear_upsample(copy_model=None, nb_classes=1):
    '''biod_fcn
    
    Description: Follows the initial VGG16 architecture by combining 4 blocks of convolutional/max pooling VGG layers.
        It then '''
    inpt = Input(shape=(None, None,3))
    
    # Create the first 4 FCN convolutional blocks
    x = fcn_block(64, 1, input_shape=(None,None,3))(inpt) # Block 1
    x = fcn_block(128, 2, input_shape=(None,None,64))(x)  # Block 2
    x = fcn_block(256, 3, input_shape=(None,None,128))(x) # Block 3
    x = fcn_block(512, 4, input_shape=(None,None,256))(x) # Block 4
    
    # Split the model into a DAG
    x_fork = x
    
    # Create an upsampling block for the block 4 max pooling layer
    upsamp = upsample_block(input_shape=(None,None,512), nb_classes=nb_classes, block_nb=4, factor=0)
    
    # Unite the upsampling block with the block 4 max pooling layer
    x_fork = upsamp(x_fork)
    
    # Continue building the model by adding a 5th convolutional block after block 4
    x = fcn_block(512, 5, input_shape=(None,None,512))(x) # Block 5
    
    # Upsample the block 5 maxpooling output by a factor of 2
    x = upsample_block(block_nb=5, factor=2, input_shape=(None, None, 512), nb_classes=nb_classes)(x)
    
    # Add the block 4 and block 5 upsample scoring layer predictions (FCN 16s)
    x = Add(name='add_layer')([x, x_fork])
    
    # Apply a transpose convolution to the combined scoring layers to generate the segmentation map of the same
    # dimensions as the input
    x = Conv2DTranspose(filters=nb_classes, 
                    kernel_size=(32, 32),
                    strides=(16, 16),
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    name='deconv_layer')(x)
    model = Model(inputs=inpt, outputs=x) # Create the model
    transfer_bilinear_weights(model, copy_model) # Load weights
    return model

def build_fcn_bilinear_8s(copy_model=None, nb_classes=1):
    '''biod_fcn
    
    Description: Follows the initial VGG16 architecture by combining 4 blocks of convolutional/max pooling VGG layers.
        It then '''
    inpt = Input(shape=(None, None,3))
    
    # Create the first 4 FCN convolutional blocks
    x = fcn_block(64, 1, input_shape=(None,None,3))(inpt) # Block 1
    x = fcn_block(128, 2, input_shape=(None,None,64))(x)  # Block 2
    x = fcn_block(256, 3, input_shape=(None,None,128))(x) # Block 3
    
    # Split the model into a DAG
    x_fork3 = x
    # Create an upsampling block for the block 3 max pooling layer
    upsamp3 = upsample_block(input_shape=(None,None,256), nb_classes=nb_classes, block_nb=3, factor=0)
    # Unite the upsampling block with the block 3 max pooling layer
    x_fork3 = upsamp3(x_fork3)
    
    x = fcn_block(512, 4, input_shape=(None,None,256))(x) # Block 4
    
    # Split the model into a DAG
    x_fork4 = x
    # Create an upsampling block for the block 4 max pooling layer
    upsamp4 = upsample_block(input_shape=(None,None,512), nb_classes=nb_classes, block_nb=4, factor=2)
    # Unite the upsampling block with the block 4 max pooling layer
    x_fork4 = upsamp4(x_fork4)
    
    # Continue building the model by adding a 5th convolutional block after block 4
    x = fcn_block(512, 5, input_shape=(None,None,512))(x) # Block 5
    
    # Upsample the block 5 maxpooling output by a factor of 2
    x = upsample_block(block_nb=5, input_shape=(None, None, 512), nb_classes=nb_classes, factor=4)(x)
    
    # Add the block 3, 4 and block 5 upsample scoring layer predictions (FCN 8s)
    x_add_forks = Add(name='add_layer1')([x_fork4, x_fork3])
    x = Add(name='add_layer2')([x, x_add_forks])
    
    # Apply a transpose convolution to the combined scoring layers to generate the segmentation map of the same
    # dimensions as the input

    x = Conv2DTranspose(filters=nb_classes, 
                    kernel_size=(16, 16),
                    strides=(8, 8),
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer=Constant(bilinear_upsample_weights(factor=8, number_of_classes=nb_classes)),
                    name='8s_deconv_layer')(x)

    model = Model(inputs=inpt, outputs=x) # Create the model
    transfer_bilinear_weights(model, copy_model) # Load weights
    return model

def build_count_nn(copy_model=None, nb_classes=1):
    '''biod_fcn
    
    Description: Follows the initial VGG16 architecture by combining 4 blocks of convolutional/max pooling VGG layers.
        It then '''
    inpt = Input(shape=(224, 224,3))
    
    # Create the first 4 FCN convolutional blocks
    x = fcn_block(64, 1, input_shape=(None,None,3))(inpt) # Block 1
    x = fcn_block(128, 2, input_shape=(None,None,64))(x)  # Block 2
    x = fcn_block(256, 3, input_shape=(None,None,128))(x) # Block 3
    x = fcn_block(512, 4, input_shape=(None,None,256))(x) # Block 4
    x = fcn_block(512, 5, input_shape=(None,None,512))(x) # Block 5
    # Get the counts
    x = Flatten(input_shape=(None,None,512))(x)
    x = Dense(100, input_dim=nb_classes, activation='relu')(x)
    x = Dense(25, activation='relu')(x)
    x = Dense(1, activation='relu')(x)
    
    model = Model(inputs=inpt, outputs=x) # Create the model
    transfer_bilinear_weights(model, copy_model) # Load weights
    return model

def build_seg_count_nn(copy_model=None, nb_classes=1):
    '''biod_fcn
    
    Description: Follows the initial VGG16 architecture by combining 4 blocks of convolutional/max pooling VGG layers.
        It then '''
    inpt = Input(shape=(224, 224, 1))
    
    # Create the first 4 FCN convolutional blocks
    x = fcn_block(64, 1, input_shape=(None,None,1))(inpt) # Block 1
    x = fcn_block(128, 2, input_shape=(None,None,64))(x)  # Block 2
    x = fcn_block(256, 3, input_shape=(None,None,128))(x) # Block 3
    x = fcn_block(512, 4, input_shape=(None,None,256))(x) # Block 4
    x = fcn_block(512, 5, input_shape=(None,None,512))(x) # Block 5
    # Get the counts
    x = Flatten(input_shape=(None,None,512))(x)
    x = Dense(100, input_dim=nb_classes, activation='relu')(x)
    x = Dense(25, activation='relu')(x)
    x = Dense(1, activation='relu')(x)
    
    model = Model(inputs=inpt, outputs=x) # Create the model
    transfer_bilinear_weights(model, copy_model) # Load weights
    return model

def transfer_bilinear_weights(model, wts_model):
    '''
        Description: Loads either the weights of the provided model, or the VGG16 weights into the provided model.
        The loading is accomplished by iterating through every layer in the FCN blocks, matching layer names
        in the provided wts_model, and loading them into the model.
    '''
    if wts_model == None:
        wts_model = keras.applications.vgg16.VGG16(weights='imagenet')
    for wts_lyr in wts_model.layers:
        weights = wts_lyr.get_weights()
        for mod_lyr in model.layers:
            ld_weights(mod_lyr, wts_lyr)