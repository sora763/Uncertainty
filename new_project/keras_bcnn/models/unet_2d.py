from __future__ import absolute_import

import numpy as np
from math import ceil, floor
from functools import partial
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.layers import Cropping2D, Conv2DTranspose,BatchNormalization,Concatenate
from tensorflow.keras.layers import Lambda
from tensorflow.keras import initializers

import tensorflow.keras.backend as K
import tensorflow as tf
import gc
from . import ModelArchitect
from ..layers import ReflectionPadding2D
from ..initializers import bilinear_upsample
from tensorflow.keras.optimizers import Adam
from scipy.stats import entropy, norm
from tensorflow.keras.backend import categorical_crossentropy,binary_crossentropy

_batch_axis = 0
_row_axis = 1
_col_axis = 2
_channel_axis = -1


class UNet2D(ModelArchitect):
    """ Two-dimensional U-Net

    Args:
        input_shape (list): Shape of an input tensor.
        out_channels (int): Number of output channels.
        nlayer (int, optional): Number of layers.
            Defaults to 5.
        nfilter (list or int, optional): Number of filters.
            Defaults to 32.
        ninner (list or int, optional): Number of layers in UNetBlock.
            Defaults to 2.
        kernel_size (list or int): Size of convolutional kernel. Defaults to 3.
        activation (str, optional): Type of activation layer.
            Defaults to 'relu'.
        conv_init (str, optional): Type of kernel initializer for conv. layer.
            Defaults to 'he_normal'.
        upconv_init (str, optional): Type of kernel initializer for upconv. layer.
            Defaults to 'he_normal'.
        bias_init (str, optional): Type of bias initializer for conv. and upconv. layer.
        dropout (bool, optional): If True, enables the dropout.
            Defaults to False.
        drop_prob (float, optional): Ratio of dropout.
            Defaults to 0.5.
        pool_size (int, optional): Size of spatial pooling.
            Defaults to 2.
        batch_norm (bool, optional): If True, enables the batch normalization.
            Defaults to False.
    """
    def __init__(self,
                 input_shape,
                 out_channels,
                 nlayer=5,
                 nfilter=32,
                 ninner=2,
                 kernel_size=3,
                 activation='relu',
                 conv_init='he_normal',
                 upconv_init='he_normal',
                 bias_init='zeros',
                 dropout=True,
                 drop_prob=0.5,
                 pool_size=2,
                 batch_norm=False):

        super(ModelArchitect, self).__init__()

        self._args = locals()

        self._input_shape = input_shape
        self._out_channels = out_channels
        self._nlayer = nlayer
        self._nfilter = nfilter
        self._ninner = ninner
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self._kernel_size = kernel_size
        self._activation = activation
        self._conv_init = conv_init
        self._upconv_init = upconv_init
        self._bias_init = bias_init
        self._dropout = dropout
        self._drop_prob = drop_prob
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        self._pool_size = pool_size
        self._batch_norm = batch_norm

    def save_args(self, out):
        super().save_args(out, self._args)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def ninner(self):
        return self._ninner

    @property
    def nlayer(self):
        return self._nlayer

    @property
    def nfilter(self):
        return self._nfilter

    @property
    def pool_size(self):
        return self._pool_size

    @property
    def activation(self):
        return Activation(self._activation)

    @property
    def kernel_size(self):
        return self._kernel_size

    @property
    def conv_init(self):
        return initializers.get(self._conv_init)

    @property
    def upconv_init(self):

        if self._upconv_init == 'bilinear':
            return bilinear_upsample()

        return initializers.get(self._upconv_init)

    @property
    def bias_init(self):
        return self._bias_init

    @property
    def pad(self):
        return ReflectionPadding2D

    @property
    def conv(self):
        return partial(Conv2D,
                        padding='valid', use_bias=True,
                        kernel_initializer=self.conv_init,
                        bias_initializer=self.bias_init)

    @property
    def upconv(self):

        return partial(Conv2DTranspose,
                        strides=self.pool_size,
                        padding='valid', use_bias=True,
                        kernel_initializer=self.upconv_init,
                        bias_initializer=self.bias_init)

    @property
    def pool(self):
        return MaxPooling2D(pool_size=self.pool_size)


    @property
    def dropout(self):
        if self._dropout:
            return partial(Dropout(self._drop_prob))
        else:
            return lambda x: x

    @property
    def norm(self):
        if self._batch_norm:
            return BatchNormalization(axis=_channel_axis)
        else:
            return lambda x: x

    def concat(self, x1, x2):

        dx = (x1.shape[_row_axis] - x2.shape[_row_axis]) / 2
        dy = (x1.shape[_col_axis] - x2.shape[_col_axis]) / 2

        crop_size = ((floor(dx), ceil(dx)), (floor(dy), ceil(dy)))

        x12 = Concatenate(axis=_channel_axis)([Cropping2D(crop_size)(x1), x2])

        return x12

    def base_block(self, x, nfilter, ksize):

        h = x

        for _ in range(self.ninner):
            h = self.pad([(k-1)//2 for k in ksize])(h)
            h = self.conv(nfilter, ksize)(h)
            h = self.norm(h)
            h = self.activation(h)

        h = self.dropout(h)

        return h

    def contraction_block(self, x, nfilter):

        h = self.base_block(x, nfilter, self.kernel_size)
        o = self.pool(h)

        return h, o

    def expansion_block(self, x1, x2, nfilter):

        ksize = self.kernel_size

        h1 = self.pad([(k-1)//2 for k in ksize])(x1)
        h1 = self.upconv(x1.shape[_channel_axis], ksize)(x1)
        h1 = self.norm(h1)
        h1 = self.activation(h1)

        h = self.concat(h1, x2)
        h = self.base_block(h, nfilter, ksize)

        return h


    def gen_dice_loss(self,y_true, y_pred):
        '''
        computes the sum of two losses : generalised dice loss and weighted cross entropy
        '''

        #generalised dice score is calculated as in this paper : https://arxiv.org/pdf/1707.03237
        y_true_f = K.reshape(y_true,shape=(-1,3))
        y_pred_f = K.reshape(y_pred,shape=(-1,3))
        sum_p=K.sum(y_pred_f,axis=-2)
        sum_r=K.sum(y_true_f,axis=-2)
        sum_pr=K.sum(y_true_f * y_pred_f,axis=-2)
        weights=K.pow(K.square(sum_r)+K.epsilon(),-1)
        generalised_dice_numerator =2*K.sum(weights*sum_pr)
        generalised_dice_denominator =K.sum(weights*(sum_r+sum_p))
        generalised_dice_score =generalised_dice_numerator /generalised_dice_denominator
        GDL=1-generalised_dice_score
        del sum_p,sum_r,sum_pr,weights
        gc.collect()

        return GDL#+weighted_log_loss(y_true,y_pred)


    def build(self):
        inputs = Input(self.input_shape)

        store_activations = []

        # down
        h_pool = inputs
        for i in range(self.nlayer):
            nfilter = self.nfilter * (2 ** (i))
            h_conv, h_pool = self.contraction_block(h_pool, nfilter)
            store_activations.append(h_conv)

        store_activations = store_activations[::-1] # NOTE: reversed

        # up
        h = store_activations[0]
        for i in range(1, self.nlayer):
            nfilter = self.nfilter * (2 ** (self.nlayer-i-1))
            h = self.expansion_block(h, store_activations[i], nfilter)

        # out
        h = self.pad(1)(h)
        outputs = self.conv(self.out_channels, 3)(h)
        outputs = Activation('sigmoid')(outputs)
        model = Model(inputs=inputs, outputs=outputs)
        #model.compile(optimizer=Adam(lr=1e-4), loss=self.original2, metrics=['accuracy'])

        return model

