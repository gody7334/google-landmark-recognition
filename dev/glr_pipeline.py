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
from dev.glr_bot import GLRBot
from utils.model import *
from utils.pipeline import BasePipeline
from dev.bcnn import *

class GLRPipeline(BasePipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_pipeline_params(self):
        # self.stage_params = GLRPipelineParams(self.model).simple_cutoff()
        pass

    def init_model(self):
        self.model = sBCNN_CP_HP(
                num_classes=203100,
                bi_vector_dim= 2048,
                cnn_type1='resnet50',
                # cnn_type2='resnet34',
                )

    def init_dataloader(self):
        self.dl=GLRDataLoader(self.train_df,
                self.val_df,
                self.holdout_df,
                None,
                files_path=self.files_path,
                split_ratio=self.split_ratio)

    def init_bot(self):
        self.bot = GLRBot(
            self.model, self.dl.train_loader, self.dl.val_loader,
            optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-3),
            criterion=torch.nn.CrossEntropyLoss(),
            echo=True,
            use_tensorboard=True,
            avg_window=25,
            snapshot_policy='validate',
            folds = 0,
            fold = 0
        )

    def keep_class_by_freq(self, df, freq):
        img_count = df.landmark_id.value_counts()
        img_count.name = 'img_count'
        df = df.join(img_count, on='landmark_id')
        df = df[df['img_count']>100]

    def get_stage_params(self):
        base_lr = float(input('base lr: '))
        G.logger.info(f"base lr: {base_lr}")

        cutoff = int(input('class count for cutoff: '))
        G.logger.info(f"cutoff count: {cutoff}")

        eval_interval = int(input('eval interval step: '))
        G.logger.info(f"eval interval step: {eval_interval}")

        eval_step = int(input('steps in eval: '))
        G.logger.info(f"steps in eval: {eval_step}")

        params = {
                    'optimizer_init':[
                        [
                            [{'params':self.model.feat1.parameters(),'lr':base_lr*0.1},
                             # {'params':self.model.feat2.parameters(),'lr':base_lr*0.1},
                             {'params':self.model.com_bi_pool.parameters(),'lr':base_lr},
                             {'params':self.model.fc.parameters(),'lr':base_lr, 'eps':1e-5},]
                        ],
                        {'weight_decay':1e-4}
                    ],
                    'cutoff': cutoff,
                    'batch_size': [4,16,16],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 1 if A.dev_exp=="EXP" else 1,
                    'eval_interval': eval_interval,
                    'eval_step': eval_step,
                }
        return params

    def do_cycles_train(self):
        G.logger.info("start cycle training")
        stage=0

        # while(stage<len(self.stage_params)):
            # params = self.stage_params[stage]

        while(True):
            self.load_stage_model()
            params = self.get_stage_params()
            G.logger.info("Start stage %s", str(stage))

            # eval every 5000 step as one epoch is too long
            # eval only 100 step as eval too much eval data
            eval_interval=params['eval_interval']
            eval_step=params['eval_step']

            if A.dev_exp=='DEV': frac=1.0

            if params['batch_size'] is not None:
                self.dl.get_dataset()
                # update dataset df for resample dataset
                # self.dl.resample_dataset(random_state=stage,
                        # upper_count=params['resample'][0],
                        # lower_count=params['resample'][1],
                        # frac=params['resample'][2])

                # remove lower freq class data in train for fast convergen
                lower_count = 1 if A.dev_exp=='DEV' else params['cutoff']
                self.dl.cutoff_dataset(lower_count=lower_count)

                # update dataloader for new dataset
                self.dl.update_batch_size(
                        train_size=params['batch_size'][0],
                        val_size=params['batch_size'][1],
                        test_size=params['batch_size'][2])

            self.oc.update_bot(optimizer_init = params['optimizer_init'],
                    scheduler=params['scheduler'],
                    unfreeze_layers=params['unfreeze_layers'],
                    freeze_layers=params['freeze_layers'],
                    dropout_ratio=params['dropout_ratio'],
                    n_epoch=params['epoch'],
                    stage=str(stage),
                    train_loader=self.dl.train_loader,
                    val_loader=self.dl.val_loader,
                    eval_interval=eval_interval,
                    eval_step=eval_step
                    )
            self.oc.train_one_cycle()
            # self.do_prediction('')
            stage+=1

            if A.dev_exp=="DEV" and stage==10:
                break

    def do_prediction(self, target_path=''):
        if target_path != '':
            self.bot.load_model(target_path)

        preds, confs, targets, loss = self.bot.predict(self.dl.test_loader, return_y=True)

        G.logger.info("holdout validation loss: %.6f", loss)
        G.logger.tb_scalars("losses", {"Holdout": loss}, self.bot.step)

        score, accu = self.bot.metrics(preds, confs, targets)
        G.logger.info("holdout, gap: %.6f, accu: %.6f", score, accu)
        G.logger.tb_scalars(
            "gap", {"holdout": score},  self.bot.step)
        G.logger.tb_scalars(
            "accu", {"holdout": accu},  self.bot.step)


class GLRPipelineParams():
    def __init__(self,model):
        self.model = model
        self.params = []
        pass

    def simple_cutoff(self):
        self.params = \
        [
            [
                {
                    'optimizer_init':[
                        [
                            [{'params':self.model.feat1.parameters(),'lr':1e-4},
                             {'params':self.model.feat2.parameters(),'lr':1e-4},
                             {'params':self.model.com_bi_pool.parameters(),'lr':2e-4},
                             {'params':self.model.fc.parameters(),'lr':2e-4, 'eps':1e-5},]
                        ],
                        {'weight_decay':1e-4}
                    ],
                    'cutoff': 100,
                    'batch_size': [4,16,16],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 1 if A.dev_exp=="EXP" else 1,
                }
            ]*1,
            [
                {
                    'optimizer_init':[
                        [
                            [{'params':self.model.feat1.parameters(),'lr':1e-4},
                             {'params':self.model.feat2.parameters(),'lr':1e-4},
                             {'params':self.model.com_bi_pool.parameters(),'lr':2e-4},
                             {'params':self.model.fc.parameters(),'lr':2e-4, 'eps':1e-5},]
                        ],
                        {'weight_decay':1e-4}
                    ],
                    'cutoff': 50,
                    'batch_size': [4,16,16],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 1 if A.dev_exp=="EXP" else 1,
                }
            ]*2,
        ]

        self.params = [j for sub in self.params for j in sub]
        return self.params

    def simple_resample(self):
        self.params = \
        [
            [
                {
                    'optimizer_init':[
                        [
                            [{'params':self.model.feat1.parameters(),'lr':1e-4},
                             {'params':self.model.feat2.parameters(),'lr':1e-4},
                             {'params':self.model.com_bi_pool.parameters(),'lr':1e-3},
                             {'params':self.model.fc.parameters(),'lr':1e-3, 'eps':1e-5},]
                        ],
                        {'weight_decay':1e-4}
                    ],
                    'resample': [1000,50,[0.1,0.05,0.1]], # uppser_count, lower_count, [train, val, test fraction]
                    'batch_size': [4,16,16],
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
                    'optimizer_init':[
                        [
                            [{'params':self.model.feat1.parameters(),'lr':1e-4},
                             {'params':self.model.feat2.parameters(),'lr':1e-4},
                             {'params':self.model.com_bi_pool.parameters(),'lr':1e-3},
                             {'params':self.model.fc.parameters(),'lr':1e-3, 'eps':1e-5},]
                        ],
                        {'weight_decay':1e-4}
                    ],
                    'resample': [500,20,[0.1,0.05,0.1]], # uppser_count, lower_count, fraction
                    'batch_size': [4,16,16],
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
                    'optimizer_init':[
                        [
                            [{'params':self.model.feat1.parameters(),'lr':1e-4},
                             {'params':self.model.feat2.parameters(),'lr':1e-4},
                             {'params':self.model.com_bi_pool.parameters(),'lr':1e-3},
                             {'params':self.model.fc.parameters(),'lr':1e-3, 'eps':1e-5},]
                        ],
                        {'weight_decay':1e-4}
                    ],
                    'resample': [100,10,[0.1,0.05,0.1]], # uppser_count, lower_count, fraction
                    'batch_size': [4,16,16],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 1 if A.dev_exp=="EXP" else 1,
                }
            ]*5,
            [
                {
                    'optimizer_init':[
                        [
                            [{'params':self.model.feat1.parameters(),'lr':1e-4},
                             {'params':self.model.feat2.parameters(),'lr':1e-4},
                             {'params':self.model.com_bi_pool.parameters(),'lr':5e-4},
                             {'params':self.model.fc.parameters(),'lr':5e-4, 'eps':1e-5},]
                        ],
                        {'weight_decay':1e-4}
                    ],
                    'resample': [100,5,[0.1,0.05,0.1]], # uppser_count, lower_count, fraction
                    'batch_size': [4,16,16],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 1 if A.dev_exp=="EXP" else 1,
                }
            ]*5,
            [
                {
                    'optimizer_init':[
                        [
                            [{'params':self.model.feat1.parameters(),'lr':1e-4},
                             {'params':self.model.feat2.parameters(),'lr':1e-4},
                             {'params':self.model.com_bi_pool.parameters(),'lr':2e-4},
                             {'params':self.model.fc.parameters(),'lr':2e-4, 'eps':1e-5},]
                        ],
                        {'weight_decay':1e-4}
                    ],
                    'resample': [100,1,[0.1,0.05,0.1]], # uppser_count, lower_count, fraction
                    'batch_size': [4,16,16],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 1 if A.dev_exp=="EXP" else 1,
                }
            ]*5,

        ]
        self.params = [j for sub in self.params for j in sub]
        return self.params

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
                    'batch_size': [8,16,16],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 1 if A.dev_exp=="EXP" else 1,
                }
            ]*1,
            [
                {
                    'optimizer': Adam(
                        [{'params':self.model.feat1.parameters(),'lr':1e-4},
                         {'params':self.model.feat2.parameters(),'lr':1e-4},
                         {'params':self.model.com_bi_pool.parameters(),'lr':5e-4},
                         {'params':self.model.fc.parameters(),'lr':5e-4, 'eps':1e-5}],
                        weight_decay=1e-4),
                    'resample': [1000,20,0.2], # uppser_count, lower_count, fraction
                    'batch_size': [8,16,16],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 1 if A.dev_exp=="EXP" else 1,
                }
            ]*1,
            [
                {
                    'optimizer': Adam(
                        [{'params':self.model.feat1.parameters(),'lr':1e-4},
                         {'params':self.model.feat2.parameters(),'lr':1e-4},
                         {'params':self.model.com_bi_pool.parameters(),'lr':1e-3},
                         {'params':self.model.fc.parameters(),'lr':1e-3, 'eps':1e-5},],
                        weight_decay=1e-4),
                    'resample': [1000,50,0.2], # uppser_count, lower_count, fraction
                    'batch_size': [8,16,16],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 1 if A.dev_exp=="EXP" else 1,
                }
            ]*1,

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
