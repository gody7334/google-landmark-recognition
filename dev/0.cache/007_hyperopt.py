
# coding: utf-8

# ## https://www.kaggle.com/hung96ad/pytorch-starter
# * LSTM, gru,
# * bidirection,
# * attention on 2 gru,
# * max pool and global pool on last layer,
# * concatinate 2 attention and 2 pool
# * fn, relu, fn, sigmoid
# * blank augmentation r=0.05
# * weighted sampler lablel[0,1], original_dist[15,1], rebalance_weight[1,2],

# In[1]:


import ipdb
import re
import time
import gc
import random
import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import sklearn

from pprint import pprint as pp
from tqdm import tqdm
tqdm.pandas()
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.utils.data.sampler import *

from importlib import reload
import xgboost as xgb
from xgboost import XGBClassifier
from hyperopt import hp
from sklearn.ensemble import RandomForestClassifier
from hyperopt.pyll.stochastic import sample
from hyperopt import rand, tpe
from hyperopt import Trials
from hyperopt import fmin

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# MODE = 'DEVELOPMENT'; sample_size=10000
MODE = 'EXPERIMENT'


# In[2]:


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# In[3]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[4]:


def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result


# In[5]:


train_preds_all = np.load('train_preds_all.npy')
test_local_pred_models = np.load('test_local_pred_models.npy')
test_local_target_models = np.load('test_local_target_models.npy')
train_y = np.load('train_y.npy')
print(train_preds_all.shape)
print(np.array(test_local_pred_models).shape)
print(np.array(test_local_target_models).shape)
print(train_y.shape)


# In[6]:


def RF_search(X_train, y_train, X_test, y_test):
    def Objective(hyperparams):
        clf = RandomForestClassifier(n_estimators=hyperparams['n_estimators'],
                                 criterion='gini',
                                 max_depth=hyperparams['max_depth'],
                                 random_state=0,
                                 n_jobs=10,
                                 min_samples_split=50,
                                 min_samples_leaf=100,
                                 min_weight_fraction_leaf=0,
                                 max_features='auto',
                                 max_leaf_nodes=None,
                                 min_impurity_decrease=0,
                                 bootstrap=True,
                                 oob_score=False,
                                 verbose=False,
                                 warm_start=False,
                                 class_weight=None
                                )
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_train)
        threshold = threshold_search(y_train, y_prob[:,1])
        y_prob = clf.predict_proba(X_test)[:,1]
        f1 = metrics.f1_score(y_test, y_prob > threshold['threshold'])
        print(f1)

        return 1-f1

    # Create the domain space
    hyperparams = {
        'n_estimators': hp.choice('n_estimators', np.arange(10,200)),
        'max_depth': hp.choice('max_depth',np.arange(5,20))
    }
    tpe_algo = tpe.suggest
    tpe_trials = Trials()

    # Run 2000 evals with the tpe algorithm
    tpe_best = fmin(fn=Objective, space=hyperparams, algo=tpe_algo, trials=tpe_trials,
                    max_evals=100, rstate= np.random.RandomState(50))

    print(tpe_best)


# In[7]:


epoch = 4
X_train = train_preds_all[epoch,:,:].transpose([1,0])
y_train = train_y
X_test = test_local_pred_models.transpose([1,2,0])[epoch,:,:]
y_test = test_local_target_models.transpose([1,2,0])[epoch,:,0]
RF_search(X_train, y_train, X_test, y_test)


# In[46]:


# Create the domain space
hyperparams = {
    'n_estimators': hp.uniform('x', 10, 1000)
    'max_depth': hp.uniform('x', 5, 20)
}

samples = []
# Sample 10000 values from the range
for _ in range(10000):
    samples.append(sample(hyperparams['n_estimators']))
# Histogram of the values
plt.hist(samples, bins = 20, edgecolor = 'black');
plt.xlabel('x'); plt.ylabel('Frequency'); plt.title('Domain Space');


# In[52]:


# Create the algorithms
tpe_algo = tpe.suggest
rand_algo = rand.suggest

# Create two trials objects
tpe_trials = Trials()
rand_trials = Trials()


# In[54]:


epoch = 4
X_train = train_preds_all[epoch,:,:].transpose([1,0])
y_train = train_y
X_test = test_local_pred_models.transpose([1,2,0])[epoch,:,:]
y_test = test_local_target_models.transpose([1,2,0])[epoch,:,0]

RF_Objective(X_train=X_train,
             y_train=y_train,
             X_test=X_test,
             y_test=y_test,
             n_estimators=10,
             max_depth=10)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
for i in range(train_preds_all.shape[0]):
    clf = RandomForestClassifier(n_estimators=100,
                                 max_depth=10,
                                 random_state=0,
                                 n_jobs=10,
                                 min_samples_split=50,
                                 min_samples_leaf=100)
    X_train = train_preds_all[i].transpose([1,0])
    y_train = train_y
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_train)
    threshold = threshold_search(y_train, y_prob[:,1])

    X_test = np.array(test_local_pred_models).transpose([1,2,0])[i]
    y_test = np.array(test_local_target_models).transpose([1,2,0])[i,:,0]
    y_prob = clf.predict_proba(X_test)[:,1]
    print(metrics.f1_score(y_test, y_prob > threshold['threshold']))


# In[ ]:




useTrainCV = True
cv_folds = 5
early_stopping_rounds = 100

# for i in range(1):
for i in range(train_preds_all.shape[0]):

    params = { 'tree_method':'gpu_hist', 'predictor':'gpu_predictor' }
    alg = XGBClassifier(learning_rate=0.01, n_estimators=500, max_depth=8,
                    min_child_weight=1.0, gamma=0.2, subsample=0.6, colsample_bytree=0.2,
                    objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27, **params)

    X_train = train_preds_all[i].transpose([1,0])
    y_train = train_y

    if useTrainCV:
        print("Start Feeding Data")
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        # xgtest = xgb.DMatrix(X_test.values, label=y_test.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # print('Start Training')
    alg.fit(X_train, y_train, eval_metric='auc', verbose=True)
    y_prob = alg.predict_proba(X_train)
    threshold = threshold_search(y_train, y_prob[:,1])

    # print("Start Predicting")
    X_test = np.array(test_local_pred_models).transpose([1,2,0])[i]
    y_test = np.array(test_local_target_models).transpose([1,2,0])[i,:,0]
    pred_proba = alg.predict_proba(X_test)[:, 1]
    predictions = pred_proba > threshold['threshold']
#     print("\n关于现在这个模型")
#     print("准确率 : %.4g" % metrics.accuracy_score(y_test, predictions))
#     print("AUC 得分 (训练集): %f" % metrics.roc_auc_score(y_test, pred_proba))
    print("F1 Score 得分 (训练集): %f" % metrics.f1_score(y_test, predictions))


# In[ ]:


import lightgbm as lgb

for i in range(train_preds_all.shape[0]):
    X_train = train_preds_all[i].transpose([1,0])
    y_train = train_y

    lgb_params = {
        'objective': '',
        'num_leaves': 58,
        'subsample': 0.6143,
        'colsample_bytree': 0.6453,
        'min_split_gain': np.power(10, -2.5988),
        'reg_alpha': np.power(10, -2.2887),
        'reg_lambda': np.power(10, 1.7570),
        'min_child_weight': np.power(10, -0.1477),
        'verbose': -1,
        'seed': 3,
        'boosting_type': 'gbdt',
        'max_depth': -1,
        'learning_rate': 0.05,
        'metric': 'l2',
        'num_boost_round':10000,
    }
    clf = lgb.LGBMClassifier(**lgb_params)
    clf.fit(
        X_train,
        y_train
    )

    y_prob = clf.predict_proba(X_train)
    threshold = threshold_search(y_train, y_prob[:,1])

    X_test = np.array(test_local_pred_models).transpose([1,2,0])[i]
    y_test = np.array(test_local_target_models).transpose([1,2,0])[i,:,0]
    pred_proba = clf.predict_proba(X_test)[:, 1]
    predictions = pred_proba > threshold['threshold']
    print("F1 Score 得分 (训练集): %f" % metrics.f1_score(y_test, predictions))


# In[ ]:


thresholds = []
for i in range(len(train_preds)):
    thresholds.append(threshold_search(train_y, sigmoid(train_preds[i])))
pp(thresholds)

test_local_f1s = []
test_local_pred_models_mean = sigmoid(np.array(test_local_pred_models)).mean(axis=0)
test_local_target_ = np.array(test_local_target_models)[0,0,:].squeeze()

for i in range(len(test_local_pred_models_mean)):
    test_local_f1s.append(f1_score(test_local_target_, np.array(test_local_pred_models_mean[i]) > thresholds[i]['threshold']))
pp(test_local_f1s)


# In[ ]:


test_local_pred_models_ = sigmoid(np.array(test_local_pred_models)).transpose([1,0,2])
for i in range(test_local_pred_models_.shape[0]):
    test_local_pred_corr = pd.DataFrame(test_local_pred_models_[i].transpose([1,0])).corr()
    pp(test_local_pred_corr)


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

class_names = ['sincere','non-sincere']
for i in range(len(test_local_pred_models_mean)):
    cnf_matrix = confusion_matrix(test_local_target_,
                                  np.array(test_local_pred_models_mean[i]) > thresholds[i]['threshold'])
    plt.figure(i*2)
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    plt.figure(i*2+1)
    plot_confusion_matrix(cnf_matrix, classes=class_names,normalize=True,
                          title='Confusion matrix, without normalization')


# In[ ]:


preds = train_preds > search_result['threshold']
train_df['pred'] = preds.astype(int)
error_pred = train_df.loc[train_df['target'] != train_df['pred']]
error_pred.to_csv('error_pred.csv')


# In[ ]:


sub = pd.read_csv('./data/sample_submission.csv')
sub.prediction = test_preds > search_result['threshold']
sub.to_csv("submission.csv", index=False)

