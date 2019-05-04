from utils.project import ArgParser as A;
from utils.project import Global as G

from pathlib import Path
from pprint import pprint as pp
import itertools
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.batchnorm import _BatchNorm

from utils.bot import OneCycle
from utils.lr_finder import LRFinder
from utils.lr_scheduler import TriangularLR

from utils.bot import *
from dev.glr_bot import *
from utils.data import *
from dev.data import GLRDataLoader
from utils.model import *
from utils.pipeline import BasePipeline
from dev.bcnn import BCNN, BCNN_CP, BCNN_MP, BCNN_CP_HP

class GLRPipeline(BasePipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_pipeline_params(self):
        self.stage_params = GLRPipelineParams(self.model).half_precision()

    def init_model(self):
        self.model = BCNN_CP_HP(num_classes=203094, bi_vector_dim= 2048,
            cnn_type1='resnet34', cnn_type2='resnet34')

    def keep_class_by_freq(self, df, freq):
        img_count = df.landmark_id.value_counts()
        img_count.name = 'img_count'
        df = df.join(img_count, on='landmark_id')
        df = df[df['img_count']>100]

    def do_cycles_train(self):
        G.logger.info("start cycle training")
        stage=0
        while(stage<len(self.stage_params)):
            params = self.stage_params[stage]
            G.logger.info("Start stage %s", str(stage))

            if A.dev_exp=='DEV': frac=1.0

            if params['batch_size'] is not None:
                self.dl.get_dataset()
                # update dataset df for resample dataset
                self.dl.resample_dataset(random_state=stage,
                        upper_count=params['resample'][0],
                        lower_count=params['resample'][1],
                        frac=params['resample'][2])
                # update dataloader for new dataset
                self.dl.update_batch_size(
                        train_size=params['batch_size'][0],
                        val_size=params['batch_size'][1],
                        test_size=params['batch_size'][2])

            self.oc.update_bot(optimizer = params['optimizer'],
                    scheduler=params['scheduler'],
                    unfreeze_layers=params['unfreeze_layers'],
                    freeze_layers=params['freeze_layers'],
                    dropout_ratio=params['dropout_ratio'],
                    n_epoch=params['epoch'],
                    stage=str(stage),
                    train_loader=self.dl.train_loader,
                    val_loader=self.dl.val_loader,
                    )
            self.oc.train_one_cycle()
            self.do_prediction('')
            stage+=1

            if A.dev_exp=="DEV" and stage==5:
                break

class GLRPipelineParams():
    def __init__(self,model):
        self.model = model
        self.params = []
        pass

    def half_precision(self):
        self.params = \
        [
            [
                {
                    'optimizer': Adam(
                        [{'params':self.model.feat1.parameters(),'lr':1e-4},
                         {'params':self.model.feat2.parameters(),'lr':1e-4},
                         {'params':self.model.com_bi_pool.parameters(),'lr':1e-3},
                         {'params':self.model.fc.parameters(),'lr':1e-3, 'eps':1e-5},],
                        weight_decay=1e-4),
                    'resample': [1000,50,0.2], # uppser_count, lower_count, fraction
                    'batch_size': [4,64,64],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 1 if A.dev_exp=="EXP" else 1,
                }
            ]*2,
            [
                {
                    'optimizer': Adam(
                        [{'params':self.model.feat1.parameters(),'lr':1e-4},
                         {'params':self.model.feat2.parameters(),'lr':1e-4},
                         {'params':self.model.com_bi_pool.parameters(),'lr':5e-4},
                         {'params':self.model.fc.parameters(),'lr':5e-4, 'eps':1e-5}],
                        weight_decay=1e-4),
                    'resample': [1000,20,0.2], # uppser_count, lower_count, fraction
                    'batch_size': [16,64,64],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.fc, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 1 if A.dev_exp=="EXP" else 1,
                }
            ]*2
        ]
        self.params = [j for sub in self.params for j in sub]
        return self.params

    def simple(self):
        '''
        '''
        self.params = \
        [
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4),
                    'resample': [500,200,0.1], # uppser_count, lower_count, fraction
                    'batch_size': [16,64,64],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 1 if A.dev_exp=="EXP" else 1,
                }
            ]*3,
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4),
                    'resample': [500,100,0.1], # uppser_count, lower_count, fraction
                    'batch_size': [16,64,64],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.fc, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 1 if A.dev_exp=="EXP" else 1,
                }
            ]*5,
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=5e-4,weight_decay=1e-4),
                    'resample': [500,70,0.1], # uppser_count, lower_count, fraction
                    'batch_size': [16,64,64],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.fc, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 1 if A.dev_exp=="EXP" else 1,
                }
            ]*5,
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=5e-4,weight_decay=1e-4),
                    'resample': [500,50,0.1], # uppser_count, lower_count, fraction
                    'batch_size': [16,64,64],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.fc, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 2 if A.dev_exp=="EXP" else 1,
                }
            ]*5,
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=2e-4,weight_decay=1e-4),
                    'resample': [500,30,0.1], # uppser_count, lower_count, fraction
                    'batch_size': [16,64,64],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.fc, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 2 if A.dev_exp=="EXP" else 1,
                }
            ]*8,
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=2e-4,weight_decay=1e-4),
                    'resample': [500,20,0.1], # uppser_count, lower_count, fraction
                    'batch_size': [16,64,64],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.fc, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 2 if A.dev_exp=="EXP" else 1,
                }
            ]*8,



        ]
        self.params = [j for sub in self.params for j in sub]
        return self.params