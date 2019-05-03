import os
import re
import pandas as pd
import numpy as np
import torch
import glob
import cv2
from imgaug import augmenters as iaa
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from utils.project import Global as G
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils.project import Global as G
from utils.data import BaseDataset, BaseDataLoader

class GLRDataset(BaseDataset):
    def __init__(self, *args,
            **kwargs):
        super().__init__(*args, **kwargs)

    def resample_df(self, upper_count=100, lower_count=10,frac=0.1, random_state=0):
        G.logger.info(f"resample random state: {random_state}")
        img_count = self.df.landmark_id.value_counts()
        img_count.name = 'img_count'
        self.df = self.df.join(img_count, on='landmark_id')
        img_count = self.df['img_count'].values
        # weight = np.log(img_count+1)/np.log(1.05)
        weight = ((img_count>upper_count)*upper_count) + \
                ((img_count<lower_count)*0) + \
                (((img_count<=upper_count)*(img_count>=lower_count))*img_count) + 1e-8

        # if frac larger than remain data frac, use remain data frac
        remain = (weight.squeeze()>1e-6).sum()/weight.squeeze().shape[0]
        if remain < frac:
            frac = remain

        if frac<1.0:
            self.df = self.df.sample(frac=frac, weights=weight.squeeze(), random_state=random_state)
            self.df = self.df.reset_index(drop=True)
        remain_class = self.df.landmark_id.value_counts()
        remain_class.name = 'img_count'
        G.logger.info(f'resample upper_count: {upper_count}')
        G.logger.info(f'resample loser_count: {lower_count}')
        G.logger.info(f'number of class after resample: {len(remain_class)}')
        G.logger.info(f'data reamin ratio after filtering: {remain}')
        G.logger.info(f'data reamin number after filtering: {len(self.df)}')


class GLRDataLoader(BaseDataLoader):
    def __init__(self, *args, split_ratio=[0.8,0.1,0.1], **kwargs):
        super().__init__(*args, **kwargs)
        self.split_ratio = split_ratio


    # override for different dataset needed
    def resample_dataset(self, upper_count=100, lower_count=10,frac=0.1, random_state=0):
        self.train_ds = GLRDataset(self.df_train,
                self.files_path,
                mode='train')
        self.train_ds.resample_df(upper_count=upper_count*self.split_ratio[0],
                lower_count=lower_count*self.split_ratio[0],
                frac=frac,
                random_state=random_state)

        self.val_ds = GLRDataset(self.df_val,
                self.files_path,
                mode='val')
        self.val_ds.resample_df(upper_count=upper_count*self.split_ratio[1],
                lower_count=lower_count*self.split_ratio[1],
                frac=frac,
                random_state=random_state)

        self.test_ds = GLRDataset(self.df_test,
                self.files_path,
                mode='test')
        self.test_ds.resample_df(upper_count=999999*self.split_ratio[2],
                lower_count=0*self.split_ratio[2],
                frac=frac,
                random_state=random_state)

