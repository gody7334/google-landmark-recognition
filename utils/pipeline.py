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
from utils.model import *


class BasePipeline:
    def __init__(self, train_csv, test_csv='', files_path='', random_state=2019):
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.files_path = files_path
        self.train_df = None
        self.val_df = None
        self.holdout_df = None
        self.sample_sub = None
        self.random_state=random_state

    def init_pipeline(self):
        G.logger.info("cv split")
        self.do_cv_split()

        G.logger.info("load model")
        self.model = cnn_model(num_class=203094)

        G.logger.info("load data loader")
        self.dl=BaseDataLoader(self.train_df,
                self.val_df,
                self.holdout_df,
                None,
                files_path=self.files_path)

        G.logger.info("create bot")
        self.bot = GLRBot(
            self.model, self.dl.train_loader, self.dl.val_loader,
            optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-3),
            criterion=torch.nn.CrossEntropyLoss(),
            echo=True,
            use_tensorboard=True,
            avg_window=25,
            snapshot_policy='last',
            folds = 0,
            fold = 0
        )

        G.logger.info("create onecycle")
        self.oc = OneCycle(self.bot)

        path = ''
        continue_step = 0
        if path != '':
            G.logger.info("continue from %s, steps $d", path, continue_step)
        self.oc.update_bot(pretrained_path=path, continue_step=continue_step, n_step=0)

        G.logger.info("load cycle train pipeline params")
        self.stage_params = PipelineParams(self.model).simple()

    def do_cv_split(self):
        G.logger.info(f"random state: {self.random_state}")

        # TODO too slow....
        file_part = [os.path.basename(x).replace('.jpg','') \
                for x in glob.glob(os.path.join(self.files_path,'*.jpg'))]
        df_all = pd.read_csv(self.train_csv)
        df_part = df_all[df_all['id'].isin(file_part)]

        df_train, df_val = train_test_split(df_part, test_size=0.5,random_state=self.random_state)
        df_val, df_test = train_test_split(df_val, test_size=0.5,random_state=self.random_state)

        self.train_df = df_train.reset_index(drop=True)
        self.val_df = df_val.reset_index(drop=True)
        self.holdout_df = df_test.reset_index(drop=True)

    def do_cycles_train(self):
        G.logger.info("start cycle training")
        stage=0
        while(stage<len(self.stage_params)):
            params = self.stage_params[stage]
            G.logger.info("Start stage %s", str(stage))

            if params['batch_size'] is not None:
                # update dataset df for resample dataset
                self.dl.get_dataset(random_state=stage)
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

            if A.dev_exp=="DEV" and stage==2:
                break

    def do_prediction(self, target_path=''):
        if target_path != '':
            self.bot.load_model(target_path)

        preds, confs, targets, loss = self.bot.predict(self.dl.test_loader, return_y=True)

        G.logger.info("holdout validation loss: %.6f", loss)
        G.logger.tb_scalars("losses", {"Holdout": loss}, self.bot.step)

        score = self.bot.metrics(preds, confs, targets)
        G.logger.info("holdout gap: %.6f", score)
        G.logger.tb_scalars("gap", {"Holdout": score}, self.bot.step)


class PipelineParams():
    def __init__(self,model):
        self.model = model
        self.params = []
        pass

    def simple(self):
        '''
        baseline, one cycle train, with reducing lr after one cycle
        '''
        self.params = \
        [
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4),
                    'batch_size': [24,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.model.fc, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 1 if A.dev_exp=="EXP" else 1,
                }
            ]*1,
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4),
                    'batch_size': [16,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.model, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 5 if A.dev_exp=="EXP" else 1,
                }
            ]*2,
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-4),
                    'batch_size': [16,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.model, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 5 if A.dev_exp=="EXP" else 1,
                }
            ]*2
        ]
        self.params = [j for sub in self.params for j in sub]
        return self.params
