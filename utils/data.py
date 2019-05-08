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
from utils.project import ArgParser as A

def base_collate(batch):
    batch_size = len(batch)
    input = []
    labels = []
    for b in range(batch_size):
        input.append(batch[b][0])
        if len(batch[b]) > 1:
            labels.append(batch[b][1])

    input = np.array(input).transpose(0,3,1,2)
    input = torch.from_numpy(input).float()

    if len(labels) is 0:
        return input

    labels = np.array(labels)
    labels = torch.from_numpy(labels).long()
    return input, labels

class BaseDataset(Dataset):

    def __init__(self, df, files_path, w=256, h=256,
            mode='train', labeled=True, collate_fn=base_collate):
        self.df = df
        self.files_path = files_path
        self.w = w
        self.h = h
        self.mode = mode
        self.labeled = labeled
        self.augmentor = self.augment_pipe()
        self.collate_fn = collate_fn

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.loc[idx, 'id']
        label = self.df.loc[idx, 'landmark_id']
        img = cv2.imread(os.path.join(self.files_path, filename+'.jpg'))

        if self.mode=='train':
            img = self.augmentor.augment_image(img)

        # resize
        img = cv2.resize(img,(self.w,self.h))

        # normalize on image net statistic
        mean=[0.485, 0.456, 0.406] #rgb
        std =[0.229, 0.224, 0.225]
        img = img/255.0
        img = (img-mean)/std

        if self.labeled:
            return img, label
        return img

    def augment_pipe(self):
        augmentor = iaa.Sequential(
            [
                # iaa.SomeOf((0, None),
                iaa.OneOf(
                [
                    iaa.CropAndPad(percent=(-0.05, 0.05)),
                    # iaa.CoarseDropout((0.0, 0.10), size_percent=(0.05, 0.25)),
                    iaa.Affine(scale=(0.95, 1.05)),
                    # iaa.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}),
                    iaa.ContrastNormalization((0.8, 1.2)),
                    iaa.Multiply((0.8, 1.2)),
                    iaa.Affine(rotate=(-5,5)),
                    # iaa.Affine(shear=(-5, 5)),
                    iaa.Fliplr(0.5),
                    # iaa.Flipud(0.5),

                # ],random_order=True),
                ]),
            ], random_order=True)
        return augmentor

    def resample_df(self, frac=0.1, random_state=0):
        G.logger.info(f"resample random state: {random_state}")

        if frac<1.0 and A.dev_exp!='DEV':
            self.df = self.df.sample(frac=frac, random_state=random_state)
            self.df = self.df.reset_index(drop=True)

        remain_class = self.df.landmark_id.value_counts()
        remain_class.name = 'img_count'
        G.logger.info(f'number of class after resample: {len(remain_class)}')
        G.logger.info(f'data reamin ratio after filtering: {frac}')
        G.logger.info(f'data reamin number after filtering: {len(self.df)}')



class BaseDataLoader():
    def __init__(self,
            df_train,
            df_val,
            df_test,
            sample_sub,
            files_path='',
            train_size=20,
            val_size=128,
            test_size=128):

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.sample_sub = sample_sub
        self.files_path = files_path
        self.df_submission = None

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.submission_ds = None
        self.get_dataset()

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.submission_loader = None
        self.update_batch_size(train_size,val_size,test_size)

    # override for different dataset needed
    def get_dataset(self):
        self.train_ds = BaseDataset(self.df_train,
                self.files_path,
                mode='train',
                collate_fn=base_collate)
        self.val_ds = BaseDataset(self.df_val,
                self.files_path,
                mode='val',
                collate_fn=base_collate)
        self.test_ds = BaseDataset(self.df_test,
                self.files_path,
                mode='test',
                collate_fn=base_collate)

    def set_submission_dataloader(self, df_submission):
        self.df_submission = df_submission
        self.submission_ds = BaseDataset(self.df_submission, labeled=False)
        self.submission_loader = DataLoader(
            self.submission_ds,
            collate_fn = base_collate,
            batch_size=self.test_size,
            num_workers=4,
            pin_memory=True,
            shuffle=False
        )

    def update_batch_size(self,
            train_size=20,
            val_size=128,
            test_size=128):
        # need update whole dataset for diff image resize
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.get_dataloader()

    def get_dataloader(self):
        self.train_loader = DataLoader(
            self.train_ds,
            collate_fn = self.train_ds.collate_fn,
            batch_size=self.train_size,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )
        # set val loader to be shuffle for fast validate
        self.val_loader = DataLoader(
            self.val_ds,
            collate_fn = self.val_ds.collate_fn,
            batch_size=self.val_size,
            num_workers=4,
            pin_memory=True,
            shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_ds,
            collate_fn = self.test_ds.collate_fn,
            batch_size=self.test_size,
            num_workers=4,
            pin_memory=True,
            shuffle=False
        )

def ut_base_dataloader():

    def sample_data_from_ds(ds):
        num = len(ds)
        for m in range(5):
            i = np.random.choice(num)
            row = ds.df.loc[i]
            # text = insert_tag(row)
            img, label = ds[i]
            print(label)
            print(img.shape)
            import ipdb; ipdb.set_trace();

    file_part = [os.path.basename(x).replace('.jpg','') \
            for x in glob.glob('/home/gody7334/google-landmark/input/trn-256/*.jpg')]
    df_all = pd.read_csv('/home/gody7334/google-landmark/input/train.csv')
    df_part = df_all[df_all['id'].isin(file_part)]

    df_train, df_val = train_test_split(df_part, test_size=0.2)
    df_val, df_test = train_test_split(df_val, test_size=0.5)

    dl=BaseDataLoader(df_train.reset_index(drop=True),
            df_val.reset_index(drop=True),
            df_test.reset_index(drop=True),
            None,
            files_path='/home/gody7334/google-landmark/input/trn-256/')

    sample_data_from_ds(dl.train_ds)
    sample_data_from_ds(dl.val_ds)
    sample_data_from_ds(dl.test_ds)

if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    ut_base_dataloader()
    print('success!')
