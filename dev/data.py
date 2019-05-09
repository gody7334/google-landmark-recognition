import sys
sys.path.insert(0, "../")

import os
import re
import pandas as pd
import numpy as np
import random
import torch
import glob
import cv2
from imgaug import augmenters as iaa
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils.project import ArgParser as A
from utils.project import Global as G
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils.project import Global as G
from utils.data import BaseDataset, BaseDataLoader
from utils.data import *

def triplet_collate(batch):
    batch_size = len(batch)
    input = []
    labels = []
    for b in range(batch_size):
        input.extend(batch[b][0])
        labels.extend(batch[b][1])

    input = np.array(input).transpose(0,3,1,2)
    input = torch.from_numpy(input).float()

    labels = np.array(labels)
    labels = torch.from_numpy(labels).long()
    return input, labels


class TripLetDataset(BaseDataset):
    def __init__(self, *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.label_idxs = {}
        self.labels = []

        self.build_label_idx()

    def __getitem__(self, idx):
        label = self.df.loc[idx, 'landmark_id']
        pos_idx = random.choice(self.label_idxs[label])
        while True:
            neg1_label = random.choice(self.labels)
            neg2_label = random.choice(self.labels)
            if neg1_label != label and neg2_label != label:
                break
        neg1_idx = random.choice(self.label_idxs[neg1_label])
        neg2_idx = random.choice(self.label_idxs[neg2_label])
        # pos_idx = self.df[self.df['landmark_id']==label].sample(n=1).index.values[0]
        # neg1_idx = self.df[self.df['landmark_id']!=label].sample(n=1).index.values[0]
        # neg2_idx = self.df[self.df['landmark_id']!=label].sample(n=1).index.values[0]
        anch_img, anch_label = super().__getitem__(idx)
        pos_img, pos_label = super().__getitem__(pos_idx)
        neg1_img, neg1_label = super().__getitem__(neg1_idx)
        neg2_img, neg2_label = super().__getitem__(neg2_idx)
        return [anch_img, pos_img, neg1_img, neg2_img], \
                [anch_label, pos_label, neg1_label, neg2_label]

    def build_label_idx(self):
        self.label_idxs = {}
        self.labels = []
        indexs = self.df.index.values
        labels = self.df['landmark_id'].values
        for l, i in zip(labels, indexs):
            if l not in self.label_idxs.keys():
                self.label_idxs[l] = [i]
            else:
                self.label_idxs[l].append(i)
        self.labels = list(self.label_idxs.keys())


class GLRDataset(TripLetDataset):
    def __init__(self, *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weight=None

        self.get_class_freq()

    def get_class_freq(self):
        img_count = self.df.landmark_id.value_counts()
        img_count.name = 'img_count'
        self.df = self.df.join(img_count, on='landmark_id')

    def remove_low_freq_sample(self, lower_count=20):
        self.df = self.df[self.df['img_count']>=lower_count]
        self.df = self.df.reset_index(drop=True)
        self.build_label_idx()
        remain_class = self.df.landmark_id.value_counts()
        remain_class.name = 'img_count'
        G.logger.info(f'loser_count: {lower_count}')
        G.logger.info(f'number of class after resample: {len(remain_class)}')
        G.logger.info(f'data reamin number after cutoff: {len(self.df)}')

    def reweight_sample(self, upper_count=100, lower_count=10):
        img_count = self.df['img_count'].values
        # weight = np.log(img_count+1)/np.log(1.05)
        self.sample_weight = ((img_count>upper_count)*upper_count) + \
                ((img_count<lower_count)*0) + \
                (((img_count<=upper_count)*(img_count>=lower_count))*img_count) + 1e-8


    def resample_df(self, upper_count=100, lower_count=10,frac=0.1, random_state=0):
        G.logger.info(f"resample random state: {random_state}")

        self.reweight_sample(upper_count, lower_count)
        # if frac larger than remain data frac, use remain data frac
        remain = (self.sample_weight.squeeze()>1e-6).sum()/self.sample_weight.squeeze().shape[0]
        if remain < frac:
            frac = remain
        if frac<1.0 and A.dev_exp!='DEV':
            self.df = self.df.sample(frac=frac, weights=weight.squeeze(), random_state=random_state)
            self.df = self.df.reset_index(drop=True)

        G.logger.info(f'rebuild label_idx for resampling')
        self.build_label_idx()

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
    def get_dataset(self):
        # test only need base dataset,
        # as we care final score more related to bce
        self.train_ds = GLRDataset(self.df_train,
                self.files_path,
                mode='train',
                collate_fn=triplet_collate)
        self.val_ds = GLRDataset(self.df_val,
                self.files_path,
                mode='val',
                collate_fn=base_collate)
        self.test_ds = BaseDataset(self.df_test,
                self.files_path,
                mode='test',
                collate_fn=base_collate)

    def cutoff_dataset(self, lower_count=20):
        self.get_dataset()
        self.train_ds.remove_low_freq_sample(lower_count*self.split_ratio[0])
        self.val_ds.remove_low_freq_sample(lower_count*self.split_ratio[1])

    # override for different dataset needed
    def resample_dataset(self, upper_count=100, lower_count=10,frac=[0.1,0.1,0.1], random_state=0):
        self.get_dataset()

        self.train_ds.resample_df(upper_count=upper_count*self.split_ratio[0],
                lower_count=lower_count*self.split_ratio[0],
                frac=frac[0],
                random_state=random_state)
        self.train_ds.resample_df(upper_count=upper_count*self.split_ratio[1],
                lower_count=lower_count*self.split_ratio[1],
                frac=frac[1],
                random_state=random_state)
        self.val_ds.resample_df(frac=frac[1], random_state=random_state)
        self.test_ds.resample_df(frac=frac[2], random_state=random_state)


def ut_base_dataloader():

    def sample_data_from_ds(ds):
        num = len(ds)
        for m in range(5):
            i = np.random.choice(num)
            row = ds.df.loc[i]
            # text = insert_tag(row)
            img, label = ds[i]
            print(label)
            print(len(img))
            print(img[0].shape)
            import ipdb; ipdb.set_trace();

    file_part = [os.path.basename(x).replace('.jpg','') \
            for x in glob.glob('/home/gody7334/google-landmark/input/trn-256/*.jpg')]
    df_all = pd.read_csv('/home/gody7334/google-landmark/input/train.csv')
    df_part = df_all[df_all['id'].isin(file_part)]

    df_train, df_val = train_test_split(df_part, test_size=0.2)
    df_val, df_test = train_test_split(df_val, test_size=0.5)

    dl=GLRDataLoader(df_train.reset_index(drop=True),
            df_val.reset_index(drop=True),
            df_test.reset_index(drop=True),
            None,
            files_path='/home/gody7334/google-landmark/input/trn-256/')
    dl.resample_dataset()

    sample_data_from_ds(dl.train_ds)
    sample_data_from_ds(dl.val_ds)
    sample_data_from_ds(dl.test_ds)

if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    ut_base_dataloader()
    print('success!')
