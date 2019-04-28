import os
import numpy as np
import torch
import random
from datetime import datetime
import shutil
import argparse
import logging
from pathlib import Path

PROJECT_NAME = 'google-landmark-recognition'
PROJECT_PATH = '/home/gody7334/project/google-landmark/cnn-landmark/'
PROJECT_BACKUP_FOLDER = '/home/gody7334/project/google-landmark/result'
IDENTIFIER   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

class ArgParser():
    gpu_id = '0'
    version = '000_BASELINE'
    dev_exp = 'EXP'
    mode = 'train'
    split = 5
    fold = 0
    holdout = 0.1
    checkpoint_path = ''
    models_pattern = ''
    predict_csv = ''
    hyperopt_trials = 0
    args = None

    def __init__(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-g", "--gpu_id", default='0', type=str, help="gpu id")
        ap.add_argument("-v","--version", default='000_BASELINE',
                help="version label")
        ap.add_argument("-de","--dev_exp", default='EXP',
                choices=['DEV', 'EXP'],
                help="development or experimenet mode")
        ap.add_argument("-m","--mode", default='train',
                choices=['train','ensemble_eval','ensemble_pred','blending_pred'],
                help="mode to run")
        ap.add_argument("-s", "--split", default=5, help="how many splits in CV")
        ap.add_argument("-f", "--fold", default=0, type=int, help="which fold in CV")
        ap.add_argument("-ho", "--holdout", default=0.1, type=float, help="how many data as holdout set")
        ap.add_argument("-cp", "--checkpoint_path", default="", help="checkpoint path for eval or pred")
        ap.add_argument("-mp", "--models_pattern", default="*.pth", help="linux wildscard file pattern")
        ap.add_argument("-p", "--predict_csv", default='', help="predict_csv")
        ap.add_argument("-t", "--hyperopt_trials", default=100, type=int, help="how many trials hyperopts will run")

        args = vars(ap.parse_args())
        ArgParser.args = args

        ArgParser.gpu_id = args['gpu_id']
        ArgParser.version = args['version']
        ArgParser.dev_exp = args['dev_exp']
        ArgParser.mode = args['mode']
        ArgParser.split = args['split']
        ArgParser.fold = args['fold']
        ArgParser.holdout = args['holdout']
        ArgParser.checkpoint_path = args['checkpoint_path']
        ArgParser.models_pattern = args['models_pattern']
        ArgParser.predict_csv = args['predict_csv']
        ArgParser.hyperopt_trials = args['hyperopt_trials']

        if args['mode'] in ['ensemble_eval','ensemble_pred','blending_pred']:
            assert args['checkpoint_path'] != ''; 'model checkpoint path for eval/pred'
        if args['mode'] in ['ensemble_pred','blending_pred']:
            assert args['predict_csv'] != ''; 'csv file for prediction'

class Global():
    proj = None
    logger = None
    exp = None

    def __init__(self, experiment, description, gpu_id='0',*kwargs):
        Environment(gpu=gpu_id)
        Global.exp = experiment
        Global.proj = Project(experiment, description)
        Global.proj.backup_project_as_zip()
        Global.logger = Logger(experiment, Global.proj.proj_save_path, logging.INFO,
                             use_tensorboard=True, echo=True)
    def reset_logger(folds, fold):
        Global.logger = Logger(Global.exp, Global.proj.proj_save_path, logging.INFO,
                             use_tensorboard=True, echo=True, folds=folds, fold=fold)

class Environment():
    '''
    setup gpu device
    setup random seed
    print environment information
    '''
    def __init__(self, gpu=''):
        print('@%s:  ' % PROJECT_NAME)

        if gpu != '': os.environ['CUDA_VISIBLE_DEVICES'] =  gpu

        print('@%s:  ' % os.path.basename(__file__))

        if 1:
            SEED = 35202  #123  #int(time.time()) #
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            print ('\tset random seed')
            print ('\t\tSEED=%d'%SEED)

        if 1:
            torch.backends.cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
            torch.backends.cudnn.enabled   = True
            print ('\tset cuda environment')
            print ('\t\ttorch.__version__              =', torch.__version__)
            print ('\t\ttorch.version.cuda             =', torch.version.cuda)
            print ('\t\ttorch.backends.cudnn.version() =', torch.backends.cudnn.version())
            try:
                print ('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =',os.environ['CUDA_VISIBLE_DEVICES'])
                NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
            except Exception:
                print ('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =','None')
                NUM_CUDA_DEVICES = 1

            print ('\t\ttorch.cuda.device_count()      =', torch.cuda.device_count())
            # print ('\t\ttorch.cuda.current_device()    =', torch.cuda.current_device())
        print('')

class Project():
    def __init__(self, experiment, description, mode='EXP'):
        self.proj_path = PROJECT_PATH
        self.proj_save_path = f'{PROJECT_BACKUP_FOLDER}/{experiment}/{IDENTIFIER}'
        self.experiment = experiment
        self.backup = self.proj_save_path + '/backup/'
        self.check_point = self.proj_save_path + '/check_point/'
        self.files = self.proj_save_path + '/files/'
        self.description = description
        print(self.__str__())
        self.mkdir()
        return

    def __str__(self):
        string = ''\
        + '\tproject              = %s\n'%PROJECT_NAME \
        + '\texperiment           = %s\n'%self.experiment \
        + '\tdiscription          = %s\n'%self.description \
        + '\tproject path         = %s\n'%self.proj_path \
        + '\tproject save path    = %s\n'%self.proj_save_path \
        + '\tbackup folder        = %s\n'%self.backup \
        + '\tfiles folder         = %s\n'%self.files \
        + '\tcheck point          = %s\n'%self.check_point \
        + '\n'
        return string

    def mkdir(self):
        os.makedirs(self.proj_save_path, exist_ok=True)
        os.makedirs(self.backup, exist_ok=True)
        os.makedirs(self.check_point, exist_ok=True)
        os.makedirs(self.files, exist_ok=True)
        return

    def backup_project_as_zip(self):
        zip_file = self.backup +'/code.train.%s.zip'%IDENTIFIER
        assert(os.path.isdir(self.proj_path))
        assert(os.path.isdir(os.path.dirname(zip_file)))
        shutil.make_archive(zip_file.replace('.zip',''), 'zip', self.proj_path)
        pass

    def backup_files(self, file_paths=[]):
        for fp in file_paths:
            file_name = os.path.basename(fp)
            shutil.copyfile(fp, self.files + file_name)
        return

class Logger:
    def __init__(self, model_name, log_dir,
            level=logging.INFO, use_tensorboard=False,
            echo=False, folds=5, fold=0):
        self.log_dir=log_dir
        self.model_name = model_name
        (Path(log_dir) / "summaries").mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime('%Y%m%d_%H%M')
        log_file = 'log_cv{}-{}_{}.txt'.format(fold, folds, date_str)
        formatter = logging.Formatter(
            '[[%(asctime)s]] %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        self.logger = logging.getLogger("bot")
        # Remove all existing handlers
        self.logger.handlers = []
        # Initialize handlers
        fh = logging.FileHandler(
            Path(log_dir) / Path(log_file))
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        if echo:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)
        self.logger.setLevel(level)
        self.logger.propagate = False
        self.tbwriter = None
        if use_tensorboard:
            from tensorboardX import SummaryWriter
            # Tensorboard
            self.tbwriter = SummaryWriter(
                log_dir + "/summaries/" + "cv{}-{}_{}_{}".format(fold, folds, self.model_name, date_str)
            )

    def info(self, msg, *args):
        self.logger.info(msg, *args)

    def warning(self, msg, *args):
        self.logger.warning(msg, *args)

    def debug(self, msg, *args):
        self.logger.debug(msg, *args)

    def error(self, msg, *args):
        self.logger.error(msg, *args)

    def tb_scalars(self, key, value, step):
        if self.tbwriter is None:
            self.debug("Tensorboard writer is not enabled.")
        else:
            self.tbwriter.add_scalars(key, value, step)

    def tb_export_scalars(self):
        if self.tbwriter is None:
            self.debug("Tensorboard writer is not enabled.")
        else:
            self.tbwriter.export_scalars_to_json(self.log_dir+"/summaries/all_scalars.json")


class Output():
    def __init__():
        return

class Graph():
    def __init__():
        return

if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    Environment(gpu='0')
    print(Project('000_test','test create'))

