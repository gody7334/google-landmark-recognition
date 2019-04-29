import logging
import sklearn
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
        return gap

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

        score = self.metrics(preds.ndarray(),
                confs.ndarray(),
                targs.ndarray())
        self.logger.info("val gap %s", score)
        self.logger.tb_scalars(
            "gap", {"val": score},  self.step)

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
                confs.concat(conf.data.cpu().numpy().astype(np.int32))
                preds.concat(pred.data.cpu().numpy().astype(np.float16))

                if return_y:
                    batch_loss = self.criterion(
                        self.extract_prediction(output), y_local.to(self.device))
                    self.eval_am.append(batch_loss.data.cpu().numpy(), y_local.size(self.batch_idx))
                    targs.concat(y_local.data.cpu().numpy().astype(np.int32))

        if return_y:
            loss = self.eval_am.avg
            return preds.ndarray(), confs.ndarray(), targs.ndarray(), loss
        return pred.ndarray(), conf.ndarray()




    def predict_avg(self, loader, checkpoint_path, pattern='', eval=False):
        '''
        avg ensemble
        '''
        preds = []
        targets = glob.glob(checkpoint_path+pattern)

        # Iterating through checkpoints
        for target in targets:
            self.logger.info("Loading %s", format(target))
            self.load_model(target)
            if eval:
                outputs, ys = self.predict(loader, return_y=eval)
            else:
                outputs = self.predict(loader, return_y=eval)
            preds.append(outputs.unsqueeze(0))
        outputs_avg = torch.cat(preds, dim=0).mean(dim=0)

        if eval:
            return outputs_avg, ys
        else:
            return outputs_avg

    def blending(self, dl, checkpoint_path, pattern):
        '''
        '''
        blend_train_xs = []
        blend_train_ys = []
        blend_test_xs = []
        blend_test_ys = []
        blend_sub_xs = []
        targets = glob.glob(checkpoint_path+pattern)
        for target in targets:
            self.logger.info("Loading %s", format(target))
            self.load_model(target)
            train_preds, train_ys = self.predict(dl.train_loader, return_y=True)
            val_preds, val_ys = self.predict(dl.val_loader, return_y=True)
            test_preds, test_ys = self.predict(dl.test_loader, return_y=True)
            submission_preds = self.predict(dl.submission_loader, return_y=False)
            preds = torch.cat((train_preds, val_preds), dim=0)
            ys = torch.cat((train_ys, val_ys,), dim=0)
            blend_train_xs.append(preds)
            blend_test_xs.append(test_preds)
            blend_sub_xs.append(submission_preds)

        blend_train_ys = ys.cpu().numpy()
        blend_test_ys = test_ys.cpu().numpy()
        blend_train_xs = nn.functional.softmax(torch.cat(blend_train_xs, dim=1),dim=1).cpu().numpy()
        blend_test_xs = nn.functional.softmax(torch.cat(blend_test_xs, dim=1),dim=1).cpu().numpy()
        blend_sub_xs = nn.functional.softmax(torch.cat(blend_sub_xs, dim=1), dim=1).cpu().numpy()
        self.xgboost_search(blend_train_xs, blend_train_ys, blend_test_xs, blend_test_ys)

        submission_pred = self.best_xgboost.predict_proba(blend_sub_xs)
        self.submission(submission_pred, dl.sample_sub)

    def xgboost_search(self, X_train, y_train, X_test, y_test):
        useTrainCV = True
        cv_folds = 5
        early_stopping_rounds = 100
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        # y_train = np.eye(3)[y_train]
        # y_test = np.eye(3)[y_test]

        def Objective(hyperparams):
            params = { 'tree_method':'gpu_hist', 'predictor':'gpu_predictor', 'num_class':3,}
            alg = XGBClassifier(
                                learning_rate=0.01,
                                n_estimators=int(hyperparams['n_estimators']),
                                max_depth=int(hyperparams['max_depth']),
                                min_child_weight=hyperparams['min_child_weight'],
                                gamma=hyperparams['gamma'],
                                subsample=hyperparams['subsample'],
                                colsample_bytree=hyperparams['colsample_bytree'],
                                colsample_bylevel=hyperparams['colsample_bylevel'],
                                objective='multi:softmax',
                                reg_alpha=0,
                                reg_lambda=1,
                                max_delta_step=0,
                                nthread=4,
                                base_score=0.5,
                                silent=True,
                                # scale_pos_weight=hyperparams['scale_pos_weight'],
                                seed=27, **params)
            if useTrainCV:
                G.logger.info("Start Feeding Data")
                xgb_param = alg.get_xgb_params()
                xgtrain = xgb.DMatrix(X_train, label=y_train)
                # xgtest = xgb.DMatrix(X_test.values, label=y_test.values)
                cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=int(alg.get_params()['n_estimators']), nfold=cv_folds,
                                  early_stopping_rounds=early_stopping_rounds)
                alg.set_params(n_estimators=cvresult.shape[0])

            # print('Start Training')
            alg.fit(X_train, y_train, eval_metric='logloss', verbose=True)
            y_prob = alg.predict_proba(X_test)
            logloss = sklearn.metrics.log_loss(y_test, y_prob)

            G.logger.info('hyperparams: %s', str(hyperparams))
            G.logger.info( 'holdout logloss: %.6f', (logloss))

            if logloss < self.min_logloss:
                G.logger.info( '! save xgboost model, logloss: %.6f', (logloss))
                pickle.dump(alg, open(G.proj.files+"xgboost.pickle.dat", "wb"))
                self.best_xgboost = alg
                self.min_logloss = logloss

            return logloss

        # Create the domain space
        hyperparams = {
            'max_depth': hp.quniform('max_depth', 2, 5, 1),
            'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
            'learning_rate': hp.uniform('learning_rate', 0.001, 0.01),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1.0),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.1, 1.0),
            'min_child_weight': hp.uniform('min_child_weight', 0.5, 2.0),
            'gamma': hp.uniform('gamma', 0.0, 0.5),
        }

        tpe_algo = tpe.suggest
        tpe_trials = Trials()

        # Run 2000 evals with the tpe algorithm
        tpe_best = fmin(fn=Objective, space=hyperparams, algo=tpe_algo, trials=tpe_trials,
                        max_evals=A.hyperopt_trials, rstate= np.random.RandomState(50))
        G.logger.info('best hyperparams: %s', str(tpe_best))

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
