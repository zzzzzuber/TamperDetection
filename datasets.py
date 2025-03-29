from torch.utils.data import Dataset,DataLoader
import torch
import os
import numpy as np 
from PIL import Image

class Imgprocessor(object):
    def __init__(self,row):
        self.row = row
    @property
    def path(self):
        return self.row[0]
    
    @property
    def label(self):
        return int(self.row[1])

class TamperDetectionDataset(Dataset):
    def __init__(self,file,transform):
        self.file = file
        self.samples = self.get_img_path()
        self.transform = transform

    def get_img_path(self):
        with open(self.file,'r')as f:
            lines = f.readlines()
        samples = [Imgprocessor(line.replace('\n','').split(' ')) for line in lines]
        
        return samples
    
    def readImg(self,path):
        # img = cv2.imread(path)
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = Image.open(path).convert('RGB')
        process_img = self.transform(img)
        return process_img

    def __getitem__(self,idx):
        sample = self.samples[idx]
        img = self.readImg(sample.path)
        label = sample.label
        return img,label

    def __len__(self):
        return len(self.samples)