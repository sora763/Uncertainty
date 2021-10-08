import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model,Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Add, concatenate, Conv2DTranspose, MaxPooling2D, Activation, Input, Dropout, UpSampling2D,ZeroPadding2D,Reshape,add,AveragePooling2D,average,BatchNormalization,Multiply,GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.backend import categorical_crossentropy,binary_crossentropy
import gc
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.applications import Xception
import numpy as np
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.layers import Layer

from tensorflow.python.keras.layers import Layer
from tensorflow.keras import backend as K

from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow_probability as tfp

def model_xception(PATCHSIZE, out_num):
    alpha=0.5
    l2_reg = 5e-5
    inputs = Input(shape = (PATCHSIZE, PATCHSIZE, 6))
    #inputs = Input(shape=(PATCHSIZE, PATCHSIZE, 3))
    #incep = Xception(include_top=False,input_tensor=input_tensor,weights=None)
    model1 = ZeroPadding2D((1, 1), input_shape=(PATCHSIZE, PATCHSIZE, 6))(inputs)
    model1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(model1)
    model1 = ZeroPadding2D((1, 1))(model1)
    model1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(model1)
    model1 = MaxPooling2D((2, 2), strides=(2, 2))(model1)

    model1 = ZeroPadding2D((1, 1))(model1)
    model1 = Conv2D(128, (3, 3), activation='relu',  padding='same',kernel_regularizer=l2(l2_reg))(model1)
    model1 = ZeroPadding2D((1, 1))(model1)
    model1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(model1)
    model1 = MaxPooling2D((2, 2), strides=(2, 2))(model1)

    model1 = ZeroPadding2D((1, 1))(model1)
    model1 = Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg))(model1)
    model1 = ZeroPadding2D((1, 1))(model1)
    model1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(model1)
    model1 = ZeroPadding2D((1, 1))(model1)
    model1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(model1)
    model1 = MaxPooling2D((2, 2), strides=(2, 2))(model1)

    model1 = ZeroPadding2D((1, 1))(model1)
    model1 = Conv2D(512, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg))(model1)
    model1 = ZeroPadding2D((1, 1))(model1)
    model1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(model1)
    model1 = ZeroPadding2D((1, 1))(model1)
    model1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(model1)
    model1 = MaxPooling2D((2, 2), strides=(2, 2))(model1)
    model1 = Model(inputs=[inputs], outputs=[model1])
    #out = incep.output
    #x1 = Flatten()(out)

    #y = Flatten()(c10) 
    #y = Dense(1024,activation="relu")(y)
    #y = Dropout(0.5)(y)
    #y = Dense(1024,activation="relu")(y)
    #y = Dropout(0.5)(y)
    #y = Dense(out_num)(y)
    #y1 = Reshape((-1, 3))(y)  

    x1 = GlobalAveragePooling2D()(model1.output)
    x = Dense(1024,activation="relu")(x1)
    x = Dropout(0.5)(x)
    x = Dense(1024,activation="relu")(x1)
    x = Dropout(0.5)(x)
    x = Dense(PATCHSIZE*PATCHSIZE)(x)
    x = Reshape((PATCHSIZE, PATCHSIZE))(x)
    #x = Lambda(lambda xx: alpha*(xx)/tf.keras.backend.sqrt(tf.keras.backend.sum(xx**2)))(x)
    x = Arcfacelayer(PATCHSIZE, 30, 0.05)(x)
    #addition = Add()([x,c10])
    outputs = Activation("sigmoid")(x)
    model = Model(inputs=[inputs],outputs=[outputs])
    #model.compile(optimizer=Adam(lr=1e-4), loss={'activation':'binary_crossentropy','conv2d_22':'binary_crossentropy'},loss_weights={'activation':1,'conv2d_22':2},metrics=[])
    model.compile(optimizer=Adam(lr=5e-5), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def UNet(in_ch, n_class, height, width):
    inputs = Input(shape = (height, width, in_ch))
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = [inputs], outputs = [conv10])
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def model_myvgg(n_class, height, width, l2_reg, out_num):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(height, width, 3)))
    model.add(Conv2D(
        64, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        64, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        128, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        128, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        256, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        256, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        256, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        512, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        512, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(
        512, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(0.5))
    model.add(Dense(out_num))
    model.add(Reshape((-1, n_class)))
    model.add(Activation("softmax"))
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def VDSR(in_ch, n_class, height, width):
    inputs = Input(shape=(height, width, in_ch))

    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)

    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)

    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)

    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(in_ch, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    res_img = model

    outputs = add([res_img, inputs])#[:,:,:,0:3]])
    #model = Activation('softmax')(outputs)
    outputs = Conv2D(n_class, (3, 3), activation='softmax', padding='same') (outputs)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=[PSNR,'accuracy'])
    return model

