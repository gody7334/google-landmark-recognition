import logging
import sklearn
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import numpy as np
import glob
import pickle
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
from utils.bot import BaseBot, ValueBuffer
from utils.project import Global as G
from utils.project import ArgParser as A
import xgboost as xgb
from xgboost import XGBClassifier
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperopt import rand, tpe
from hyperopt import Trials
from hyperopt import fmin

class GLRBot(BaseBot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_format = "%.6f"
        self.min_logloss = 999.9
        self.best_xgboost = None

    def extract_prediction(self, tensor):
        return tensor

    def GAP_vector(self, pred, conf, true, return_x=False):
        '''
        Compute Global Average Precision (aka micro AP), the metric for the
        Google Landmark Recognition competition.
        This function takes predictions, labels and confidence scores as vectors.
        In both predictions and ground-truth, use None/np.nan for "no label".

        Args:
            pred: vector of integer-coded predictions
            conf: vector of probability or confidence scores for pred
            true: vector of integer-coded labels for ground truth
            return_x: also return the data frame used in the calculation

        Returns:
            GAP score
        '''
        x = pd.DataFrame({'pred': pred, 'conf': conf, 'true': true})
        x.sort_values('conf', ascending=False, inplace=True, na_position='last')
        x['correct'] = (x.true == x.pred).astype(int)
        x['prec_k'] = x.correct.cumsum() / (np.arange(len(x)) + 1)
        x['term'] = x.prec_k * x.correct
        gap = x.term.sum() / x.true.count()
        if return_x:
            return gap, x
        else:
            return gap

    def metrics(self, preds, confs, targets):
        '''
        override if needed for different metrics
        '''
        gap, x = self.GAP_vector(preds, confs, targets, return_x=True)
        accu = accuracy_score(targets, preds)
        return gap, accu

    def eval(self, loader):
        self.model.eval()
        confs, preds, y_global = [], [], []
        self.eval_am.reset()
        confs = ValueBuffer()
        preds = ValueBuffer()
        targs = ValueBuffer()

        self.logger.info("start eval, plz wait...")

        with torch.set_grad_enabled(False):
            for *input_tensors, y_local in tqdm(loader):
                output = self.eval_one_step(input_tensors, y_local)
                outputs_softmax = nn.functional.softmax(output,dim=1)
                conf, pred = torch.max(outputs_softmax, 1)
                confs.concat(conf.data.cpu().numpy().astype(np.float16))
                preds.concat(pred.data.cpu().numpy().astype(np.int32))
                targs.concat(y_local.data.cpu().numpy().astype(np.int32))

        loss = self.eval_am.avg
        loss_str = self.loss_format % loss
        self.logger.info("val loss %s", loss_str)
        self.logger.tb_scalars(
            "losses", {"val": loss},  self.step)

        score, accu = self.metrics(preds.ndarray(),
                confs.ndarray(),
                targs.ndarray())
        self.logger.info("val, gap: %.6f, accu: %.6f", score, accu)
        self.logger.tb_scalars(
            "gap", {"val": score},  self.step)
        self.logger.tb_scalars(
            "accu", {"val": accu},  self.step)

        return loss

    def predict(self, loader, *, return_y=False):
        '''
        test set has label which can be used to investigate manually
        '''
        self.model.eval()
        # reuse eval loss meter
        self.eval_am.reset()
        confs = ValueBuffer()
        preds = ValueBuffer()
        targs = ValueBuffer()
        self.logger.info("start predict, plz wait...")

        with torch.set_grad_enabled(False):
            for *input_tensors, y_local in tqdm(loader):
                input_tensors = [x.to(self.device) for x in input_tensors]
                output = self.predict_batch(input_tensors)
                outputs_softmax = nn.functional.softmax(output,dim=1)
                conf, pred = torch.max(outputs_softmax, 1)
                confs.concat(conf.data.cpu().numpy().astype(np.float16))
                preds.concat(pred.data.cpu().numpy().astype(np.int32))

                if return_y:
                    batch_loss = self.criterion(
                        self.extract_prediction(output), y_local.to(self.device))
                    self.eval_am.append(batch_loss.data.cpu().numpy(), y_local.size(self.batch_idx))
                    targs.concat(y_local.data.cpu().numpy().astype(np.int32))

        if return_y:
            loss = self.eval_am.avg
            return preds.ndarray(), confs.ndarray(), targs.ndarray(), loss
        return preds.ndarray(), confs.ndarray()


    def submission(self, outputs, sample_sub):
        G.logger.info( 'Generate final Submission!')
        sample_sub["A"] = outputs[:,0]
        sample_sub["B"] = outputs[:,1]
        sample_sub["NEITHER"] = outputs[:,2]
        sample_sub.to_csv(G.proj.files+"submission.csv")

    # def metrics(self, outputs, targets):
        # '''
        # override if needed for different metrics
        # '''
        # criterion_scores = self.criterion(outputs, targets).data.cpu().numpy()
        # score = np.mean(criterion_scores)
        # G.logger.info("holdout validation score: %.6f", score)
        # G.logger.tb_scalars("losses", {"Holdout": score}, self.step)

        # for t in np.arange(0.9,1.0,0.01):
            # import ipdb; ipdb.set_trace();
            # outputs_sm = nn.functional.softmax(outputs,dim=1)
            # outputs_t_idx = torch.sum((outputs_sm>t).float()*1,dim=1).unsqueeze(1)

            # outputs_t = ((outputs_sm > t).float() * 0.999) + ((outputs_sm <= t).float() * 0.0005)+1e-8
            # outputs_t = outputs_t * outputs_t_idx + outputs_sm * (1-outputs_t_idx)

            # outputs_t = torch.log(outputs_t)
            # loss = nn.NLLLoss()
            # criterion_scores = loss(outputs_t, targets).data.cpu().numpy()
            # score = np.mean(criterion_scores)
            # G.logger.info("threshold: %.2f, holdout validation score: %.6f", t, score)

        # return score


    ## keep best
    # def snapshot(self, loss):
        # """Override the snapshot method because Kaggle kernel has limited local disk space."""
        # loss_str = self.loss_format % loss
        # self.logger.info("Snapshot loss %s", loss_str)
        # self.logger.tb_scalars(
            # "losses", {"val": loss},  self.step)
        # target_path =(
            # self.checkpoint_dir /
            # "snapshot_{}_{}.pth".format(self.name, loss_str))

        # if not self.best_performers or (self.best_performers[0][0] > loss) or self.snapshot_policy=='last':
            # torch.save(self.model.state_dict(), target_path)
            # self.best_performers = [(loss, target_path, self.step)]
        # self.logger.info("Saving checkpoint %s...", target_path)
        # assert Path(target_path).exists()
        # return loss
