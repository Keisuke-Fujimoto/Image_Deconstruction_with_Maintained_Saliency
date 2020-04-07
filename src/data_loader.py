import glob
import os

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms


model = 'Salicon'

# Path to processed data. Created using preprocess.py
DIR_img_train = ''.join(['../data/', model, '/imgs_train'])
DIR_map_train = ''.join(['../data/', model, '/maps_train'])
DIR_img_val = ''.join(['../data/', model, '/imgs_val'])
DIR_map_val = ''.join(['../data/', model, '/maps_val'])

# Input image and saliency map size
H, W = 192, 256


class DataLoader(object):

    def __init__(self, batch_size = 5):
        # For training
        self.list_img = [k.split(os.sep)[-1].split('.')[0] for k in glob.glob(os.path.join(DIR_img_train, '*train*'))]  # Reading data list
        self.batch_size = batch_size
        self.size = len(self.list_img)  #input image number
        self.cursor = 0  #list_img iterator
        self.num_batches = int(self.size / batch_size)

        # For validation
        self.list_img_val = [k.split(os.sep)[-1].split('.')[0] for k in glob.glob(os.path.join(DIR_img_val, '*val*'))]
        self.batch_size_val = 10
        self.size_val = len(self.list_img_val)  #input image number
        self.cursor_val = 0  #list_img iterator

    def get_batch(self):
        if self.cursor + self.batch_size > self.size:
            self.cursor = 0
            np.random.shuffle(self.list_img)
            
        imgs = torch.zeros(self.batch_size, 3, H, W)
        maps = torch.zeros(self.batch_size, 1, H, W)
        
        to_tensor = transforms.ToTensor()

        for idx in range(self.batch_size):
            curr_file = self.list_img[self.cursor]
            curr_file = ''.join([curr_file, '.png'])
            full_img_path = os.path.join(DIR_img_train, curr_file)
            full_map_path = os.path.join(DIR_map_train, curr_file)
            self.cursor += 1

            img = cv2.imread(full_img_path)     # (192,256,3)
            imgs[idx] = to_tensor(img)
            
            saliency_map = cv2.imread(full_map_path, 0)
            saliency_map = np.expand_dims(saliency_map, axis=2)
            maps[idx] = to_tensor(saliency_map)
            
        return (imgs, maps)

    def get_val(self):
        if self.cursor_val + self.batch_size_val > self.size_val:
            self.cursor_val = 0
            np.random.shuffle(self.list_img_val)
            
        imgs = torch.zeros(self.batch_size_val, 3, H, W)
        maps = torch.zeros(self.batch_size_val, 1, H, W)
        
        to_tensor = transforms.ToTensor()

        for idx in range(self.batch_size_val):
            curr_file = self.list_img_val[self.cursor_val]
            curr_file = ''.join([curr_file, '.png'])
            full_img_path = os.path.join(DIR_img_val, curr_file)
            full_map_path = os.path.join(DIR_map_val, curr_file)
            self.cursor_val += 1

            img = cv2.imread(full_img_path)     # (192,256,3)
            imgs[idx] = to_tensor(img)
            
            saliency_map = cv2.imread(full_map_path, 0)
            saliency_map = np.expand_dims(saliency_map, axis=2)
            maps[idx] = to_tensor(saliency_map)
            
        return (imgs, maps)