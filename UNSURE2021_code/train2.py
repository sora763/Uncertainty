#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import timeit
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import ModelCheckpoint

#import generator as grt
import models
import tools as tl
import tensorflow as tf
import image_process as ip
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam
import newgenerator
from keras_bcnn.models import BayesianUNet2D
from tensorflow.keras.backend import categorical_crossentropy,binary_crossentropy
from scipy.stats import entropy, norm
import gc


class TrainModel2(object):
    def __init__(self, method,arch,data,mode, stage,patch_size = 0):
        self.method = method
        self.arch = arch
        self.data = data
        self.patch_size = patch_size
        self.stage = stage
        self.n_class = self.get_target_class_num()
        self.model_name = arch
        self.mode = mode
        self.input_channel = 3
        if self.data == "ips":
            self.output_channel = 3
        else:
            self.output_channel = 1


    def get_target_class_num(self):
        if self.data in ['ips']:
            return 3
        elif self.data in ['melanoma']:
            return 2

    def dice_coef(self, y_true, y_pred):
        y_true = tf.keras.backend.flatten(y_true)
        y_pred = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true * y_pred)
        return 2.0 * intersection / (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + 1)


     # ロス関数
    def dice_loss(self, y_true, y_pred):
        return 1.0 - self.dice_coef(y_true, y_pred)

    def bce_dice_loss(self,y_true, y_pred):
        loss = binary_crossentropy(y_true, y_pred) + self.dice_loss(y_true, y_pred)
        return loss

    def load_model(self, out_num=0):
        if self.arch == 'unet':
            if self.method == 'unet_patch':
                #model = models.UNet(self.input_channel, self.output_channel, self.patch_size, self.patch_size)
                model = BayesianUNet2D((self.patch_size, self.patch_size, 3), self.output_channel).build()
                if self.stage == "stage1":
                    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])#'categorical_crossentropy'
                else:
                    model.compile(optimizer=Adam(lr=1e-6), loss=self.dice_loss, metrics=['accuracy'])#'categorical_crossentropy'
            return model


    def train2(self):
        if self.method == 'unet_patch':
            self.train_unet_patch()


    def train_unet_patch(self):
        batch_size = 12
        test_batch_size = 12
        epochs = 5
        if self.data in ['ips', 'melanoma']:
            for data_num in [1]:#dataset
                clear_session()#keras parameter 初期化

                #keras用GPUを複数使う処理. なくてもいい
                physical_devices = tf.config.experimental.list_physical_devices('GPU')
                if len(physical_devices) > 0:
                    for k in range(len(physical_devices)):
                        tf.config.experimental.set_memory_growth(physical_devices[k], True)
                        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
                else:
                    print("Not enough GPU hardware devices available")


                save_path = tl.get_save_path(self.data, self.method, self.model_name,
                                             self.patch_size, data_num)
                #stage毎にmodel loadの処理
                if self.stage == "stage1":
                    model = self.load_model()

                elif self.stage == "stage2":
                    save_path2 = save_path
                    save_path = save_path+"stage2/"
                    model = self.load_model()
                    model.load_weights(save_path2+'weights/weights.h5')

                elif self.stage == "stage3":
                    save_path2 = save_path+"stage2/"
                    save_path = save_path+"stage2/stage3/"
                    model = self.load_model()
                    model.load_weights(save_path2+'weights/weights.h5')

                elif self.stage == "stage4":
                    save_path2 = save_path+"stage2/stage3/"
                    save_path = save_path+"stage2/stage3/stage4/"
                    model = self.load_model()
                    model.load_weights(save_path2+'weights/weights.h5')

                #model.summary() 

                tl.make_dirs(self.data, self.method, self.model_name,
                            self.patch_size, data_num)
                train="/home/sora/new_project/crop/dataset_%d/train/"%data_num
                val="/home/sora/new_project/crop/dataset_%d/val/"%data_num
                start_time = timeit.default_timer()
                # train -----------------------------------------------------------
                #data_loaderの設定
                if self.data =="ips":
                    train_gen = newgenerator.ImageSequence(train,batch_size,"train",data_num)
                    valid_gen = newgenerator.ImageSequence(val, batch_size,"val",data_num)
                if self.data =="melanoma":
                    train_gen = newgenerator.ImageSequence_me(train,batch_size,"train",data_num)
                    valid_gen = newgenerator.ImageSequence_me(val, batch_size,"val",data_num)
                os.makedirs(save_path + 'weights', exist_ok=True)
                #val_lossを見てweightを保存するように設定
                model_checkpoint = ModelCheckpoint(
                    filepath=os.path.join(save_path,'weights', 'weights.h5'),
                    monitor="val_loss",
                    verbose=1,
                    save_best_only=True) 
                     #学習parameterをhistryに格納
                history = model.fit_generator(generator=train_gen,
                    epochs=epochs,
                    steps_per_epoch=len(train_gen),
                    verbose=1,callbacks=[model_checkpoint],
                    validation_data=valid_gen,
                    validation_steps=len(valid_gen),
                    max_queue_size=5)

                train_time = timeit.default_timer() - start_time
                     #学習parameter保存
                tl.save_parameter(save_path, self.model_name, self.patch_size,
                                  lr, epochs,
                                  batch_size, train_time)
                #lossをplot
                tl.draw_train_loss_plot(history, save_path)
                del train_gen,valid_gen,history,model_checkpoint,model
                gc.collect()
