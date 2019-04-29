############ Setup Project ######################
import os
import matplotlib
from utils.project import ArgParser as A; A();
from utils.project import Global as G
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ['CUDA_VISIBLE_DEVICES'] = A.gpu_id
matplotlib.use('Agg')

EXPERIMENT=A.version
DISCRIPTION='create baseline'
G(EXPERIMENT, DISCRIPTION, A.gpu_id)
G.logger.info('arg params:%s', str(A.args))

mode=A.dev_exp

##################################################

import logging
from pathlib import Path
import colored_traceback.always
from matplotlib.pyplot import *
from pprint import pprint as pp
import itertools
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from pytorch_pretrained_bert.modeling import BertModel, BertLayerNorm

from utils.bot import OneCycle
from utils.lr_finder import LRFinder
from utils.lr_scheduler import TriangularLR

from utils.bot import *
from dev.glr_bot import *
from utils.data import *
from utils.model import *
from utils.pipeline import *

####################################################


def run_gap():
    if A.mode == 'train':
        G.reset_logger(folds=A.split, fold=A.fold)
        gappl = GAPPipeline(fold=A.fold, folds=A.split, holdout_ratio=A.holdout)
        gappl.do_cycles_train()

    ###### ensemble prediction ########
    if A.mode == 'ensemble_eval':
        # path =  '/home/gody7334/gender-pronoun/input/result/002_CV_ALL/'\
                # +'2019-04-09_09-52-02'+'/check_point/'
                # +'cv0-5_stage3_snapshot_basebot_0.348980.pth'
        # pattern = 'cv[0-4]-5_stage*_snapshot_basebot_*.pth'
        path = A.checkpoint_path
        pattern = A.models_pattern
        gappl = GAPPipeline(fold=A.fold, folds=A.split, holdout_ratio=A.holdout)
        gappl.do_ensemble(path, pattern, eval=True)

    if A.mode == 'ensemble_pred':
        path = A.checkpoint_path
        pattern = A.models_pattern
        gappl = GAPPipeline(fold=A.fold, folds=A.split, holdout_ratio=A.holdout)
        gappl.do_ensemble(path, pattern, eval=False)
        gappl.do_submission()

    if A.mode == 'blending_pred':
        path = A.checkpoint_path
        pattern = A.models_pattern
        gappl = GAPPipeline(fold=A.fold, folds=A.split, holdout_ratio=A.holdout)
        gappl.do_blending(path, pattern)


def run_base():
    if A.dev_exp=='EXP':
        train_csv = '/home/gody7334/google-landmark/input/train.csv'
    elif A.dev_exp=='DEV':
        train_csv = '/home/gody7334/google-landmark/input/train_dev.csv'
    elif A.dev_exp=='DEVL':
        train_csv = '/home/gody7334/google-landmark/input/train_exp.csv'

    pl = BasePipeline(train_csv,
            files_path='/home/gody7334/google-landmark/input/trn-256')
    pl.init_pipeline()
    pl.do_cycles_train()

if __name__ == '__main__':
    G.logger.info( '%s: calling main function ... ' % os.path.basename(__file__))
    run_base()
    G.logger.info('success!')
