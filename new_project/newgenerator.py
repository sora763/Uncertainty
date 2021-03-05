from pathlib import Path
import math

#from skimage.io import imread
from tensorflow.keras.utils import Sequence
import numpy as np
from PIL import Image
import cv2
from operator import itemgetter
import os
import gc
import random

class ImageSequence(Sequence):
    def __init__(self, input_path,batch_size,mode,data_num):
        self.img_path = []
        self.mode = mode
        self.data_num = data_num
        #データパスを取得
        for x in os.listdir(input_path):
            self.img_path.append("/home/sora/new_project/crop/dataset_%d/%s/"%(self.data_num,self.mode)+x)


        self.batch_size = batch_size
        self.img = random.sample(self.img_path,len(self.img_path))#data シャッフル
        tmp = len(self.img) % self.batch_size
        #data数をbatch sizeで割れるようにする
        if tmp != 0:
            for i in range(tmp):
                self.img.pop(-1)
        gc.collect()


    def __getitem__(self, idx):
        batch_x = self.img[idx * self.batch_size:(idx + 1) * self.batch_size]

        # 画像を1枚づつ読み込んで、前処理をする
        b = []
        c = []
        d = []
        e = []
        #numpyファイルload
        for file_name in batch_x:
            a=np.load(file_name)           
            b.append(a[0])#入力画像
            c.append(a[1])#教師ラベル

        b=np.array(b)
        c = np.array(c)
 
             
        return b,c

    def __len__(self):
        #データ数/batch sizeを返す
        return math.ceil(len(self.img) / self.batch_size)


class ImageSequence2(Sequence):
#entropy_loss_method_ips.ipynbで使用
    def __init__(self, input_path,batch_size,mode,data_num):
        self.img_path = []
        self.mode = mode
        self.data_num = data_num
        for x in os.listdir(input_path):
            self.img_path.append("/home/sora/new_project/crop/dataset_%d/%s/"%(self.data_num,self.mode)+x)

        self.batch_size = batch_size
        self.img = random.sample(self.img_path,len(self.img_path))
        tmp = len(self.img) % self.batch_size
        if tmp != 0:
            for i in range(tmp):
                self.img.pop(-1)

        gc.collect()


    def __getitem__(self, idx):
        batch_x = self.img[idx * self.batch_size:(idx + 1) * self.batch_size]

        # 画像を1枚づつ読み込んで、前処理をする
        b = []
        c = []
        d = []
        e = []
        
        for file_name in batch_x:
            a=np.load(file_name)
            
            b.append(a[0])
            c.append(a[1])

        v,x,y,z=np.asarray(c).shape
        b=np.array(b)
        c = np.array(c)
             
        return b,[c,c]
        

    def __len__(self):
 
        return math.ceil(len(self.img) / self.batch_size)

class ImageSequence_me(Sequence):
#melanoma用
    def __init__(self, input_path,batch_size,mode,data_num):
        self.img_path = []
        self.mode = mode
        self.data_num = data_num
        for x in os.listdir(input_path):
            self.img_path.append("/home/sora/new_project/crop/dataset_%d/%s/"%(self.data_num,self.mode)+x)


        self.batch_size = batch_size
        self.img = random.sample(self.img_path,len(self.img_path))
        tmp = len(self.img) % self.batch_size
        if tmp != 0:
            for i in range(tmp):
                self.img.pop(-1)
        #del self.lr_img_path
        gc.collect()


    def __getitem__(self, idx):
        batch_x = self.img[idx * self.batch_size:(idx + 1) * self.batch_size]


        # 画像を1枚づつ読み込んで、前処理をする
        b = []
        c = []
        d = []
        e = []
        
        for file_name in batch_x:
            a=np.load(file_name)
            
            b.append(a[0])#入力画像
            c.append(a[1])#教師ラベル
 
        b=np.array(b)
        c = np.array(c)
        #melanomaは教師label 0,1チャネルを使用
        label = c[:,:,:,1:2]

        return b,label
        

    def __len__(self):
 
        return math.ceil(len(self.img) / self.batch_size)

class ImageSequence_me2(Sequence):
#melanoma用. entropy_loss_method_melanoma.ipynbで使用
    def __init__(self, input_path,batch_size,mode,data_num):
        self.img_path = []
        self.mode = mode
        self.data_num = data_num
        for x in os.listdir(input_path):
            self.img_path.append("/home/sora/new_project/crop/dataset_%d/%s/"%(self.data_num,self.mode)+x)

        self.batch_size = batch_size
        self.img = random.sample(self.img_path,len(self.img_path))
        tmp = len(self.img) % self.batch_size
        if tmp != 0:
            for i in range(tmp):
                self.img.pop(-1)
        gc.collect()


    def __getitem__(self, idx):
        batch_x = self.img[idx * self.batch_size:(idx + 1) * self.batch_size]

        # 画像を1枚づつ読み込んで、前処理をする
        b = []
        c = []
        d = []
        e = []
        
        for file_name in batch_x:
            a=np.load(file_name)
            
            b.append(a[0])
            c.append(a[1])

        v,x,y,z=np.asarray(c).shape
        b=np.array(b)
        c = np.array(c)
        label = c[:,:,:,1:2]
             
        return b,[label,label]
        

    def __len__(self):
 
        return math.ceil(len(self.img) / self.batch_size)
