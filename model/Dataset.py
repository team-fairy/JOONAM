import os, glob
import random

from PIL import Image
import pandas as pd 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt


torch.manual_seed(1230)
np.random.seed(1230)
random.seed(1230)


class MaskDataset(Dataset):
    def __init__(self,transform=None, augmentation=2):
        self.meta_data = path
        self.base_path = file_path
        self.transform = transform
        self.totensor = transforms.ToTensor()
        self.augmentation = augmentation
        self.file_paths = [ file for path in self.meta_data.path.values for file in self.get_files(path) ]
        
    
    def __getitem__(self, index): # index slicing 이 가능한 method
        """get_item"""
        f = self.file_paths[index]
        label = self.get_class(f)
        
        img = Image.open(f).resize((300,300))
        if self.transform:
            img = self.transform(img)
            return img, label
        else:
            return self.totensor(img), label
        
    def __len__(self):
        return len(self.file_paths)
    
    def get_files(self,path):
        # 특정 label의 size를 증가 할때 수정가능
        row_f_lst = glob.glob(os.path.join(self.base_path, path+'/*'))
        add_lst = [f for f in row_f_lst if ('incorrect' in f) or ('normal' in f)] * self.augmentation
        return row_f_lst + add_lst

    
    def is_mask(self,img):
        if 'incorrect' in img:
            return 1
        elif 'mask' in img:
            return 0
        else:
            return 2
        
    def get_age_class(self,data):
        data = int(data)
        if data < 30:
            return 0
        elif data < 60:
            return 1
        else:
            return 2 
         
    def get_gender_logit(self,img):
        return 1 if 'female' in img else 0
        
    def get_class(self,img):
        mask = self.is_mask(img)
        age = self.get_age_class( img.split('/')[-2].split('_')[-1] )
        gender = self.get_gender_logit(img)
        return mask*6 + gender*3 + age