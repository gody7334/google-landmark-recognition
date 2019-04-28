

class GAPPipeline:
    def __init__(self, fold=0, folds=5, holdout_ratio=0.1, cv_random_state=2019):
        '''
        Gradient Checkpoint model need turn off Dropout, and BatchNorm
        Accumulated Gradient need turn off BatchNorm
        '''
        self.fold = fold
        self.folds = folds
        self.holdout_ratio = holdout_ratio
        self.cv_random_state = cv_random_state
        self.train_df = None
        self.val_df = None
        self.holdout_df = None
        self.sample_sub = None
        self.submission_df = None
        self.outputs = None

        assert fold < folds; "fold cannot be larger than total folds"
        G.logger.info("cv split")
        self.do_cv_split()

        G.logger.info("load model")
        # self.model = GAPModel_CheckPoint(BERT_MODEL, torch.device("cuda:0"))
        # self.model = GAPModel(BERT_MODEL, torch.device("cuda:0"))
        self.model = score_model(BERT_MODEL)

        G.logger.info("load gapdl")
        self.gapdl = GAPDataLoader(self.train_df, self.val_df, self.holdout_df, self.sample_sub)

        if A.predict_csv!='':
            self.gapdl.set_submission_dataloader(self.submission_df)

        G.logger.info("create bot")
        self.bot = GAPBot(
            self.model, self.gapdl.train_loader, self.gapdl.val_loader,
            optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-3),
            criterion=torch.nn.CrossEntropyLoss(),
            echo=True,
            use_tensorboard=True,
            avg_window=25,
            snapshot_policy='last',
            folds = self.folds,
            fold = self.fold
        )

        G.logger.info("create onecycle")
        self.oc = OneCycle(self.bot)
        # load pretained model for continue training
        # 04-05_10-48: 0.472 -> 04-06_11-41:0.426 ->
        # path = '/home/gody7334/gender-pronoun/input/result/000_BASELINE/'\
                # +'2019-04-06_11-41-43'+'/check_point/'+'stage3_snapshot_basebot_0.348980.pth'
        # continue_step = 50500
        path = ''
        continue_step = 0
        self.oc.update_bot(pretrained_path=path, continue_step=continue_step, n_step=0)

        self.stage_params = PipelineParams(self.model).baseline_then_finetune()

    def do_cv_split(self):

        def extract_target(df):
            df["Neither"] = 0
            df.loc[~(df['A-coref'] | df['B-coref']), "Neither"] = 1
            df["target"] = 0
            df.loc[df['B-coref'] == 1, "target"] = 1
            df.loc[df["Neither"] == 1, "target"] = 2
            print(df.target.value_counts())
            return df

        G.logger.info(f"cv random state: {self.cv_random_state}")
        G.logger.info(f"cv fold:{self.fold}, total:{self.folds}")

        gap_train = extract_target(pd.read_csv("~/gender-pronoun/input/dataset/gap-test.csv",index_col=0))
        gap_val = extract_target(pd.read_csv("~/gender-pronoun/input/dataset/gap-validation.csv",index_col=0))
        gap_test = extract_target(pd.read_csv("~/gender-pronoun/input/dataset/gap-development.csv",index_col=0))
        # sample_sub = pd.read_csv("~/gender-pronoun/input/dataset/sample_submission_stage_1.csv",index_col = "ID")
        sample_sub = pd.read_csv("~/gender-pronoun/input/dataset/sample_submission_stage_2.csv",index_col = "ID")

        if self.holdout_ratio==0:
            #### CV train val, use original develop set as test set
            train = pd.concat([gap_train, gap_val],ignore_index=True)
            train = train.reset_index(drop=True)
            holdout = gap_test.reset_index(drop=True)
        else:
            ##### CV all
            gap_all = pd.concat([gap_train, gap_val, gap_test],ignore_index=True)
            train, holdout = train_test_split(gap_all, test_size=self.holdout_ratio,
                    random_state=self.cv_random_state, shuffle=True, stratify=gap_all['target'])
            train = train.reset_index(drop=True)
            holdout = holdout.reset_index(drop=True)

        Kfold = StratifiedKFold(n_splits=self.folds,
                random_state=self.cv_random_state,shuffle=True).split(train, train['target'])
        self.holdout_df = holdout
        # self.submission_df = pd.read_csv(A.predict_csv, index_col=0) if A.predict_csv!='' else None
        self.submission_df = pd.read_csv(A.predict_csv, sep='\t') if A.predict_csv!='' else None
        self.sample_sub = sample_sub

        for n_fold, (train_index, val_index) in enumerate(Kfold):
            train_df = train.loc[train_index]
            train_df = train_df.reset_index(drop=True)
            val_df   = train.loc[val_index]
            val_df   = val_df.reset_index(drop=True)
            if n_fold == self.fold:
                self.train_df = train_df
                self.val_df = val_df
                self.train_df.to_csv(G.proj.files+\
                        f"{self.fold}_{self.folds}_train_df.csv",index=False)
                self.val_df.to_csv(G.proj.files+\
                        f"{self.fold}_{self.folds}_val_df.csv",index=False)
                self.holdout_df.to_csv(G.proj.files+\
                        f"{self.fold}_{self.folds}_holdout_df.csv",index=False)
                return

    def do_cycles_train(self):
        stage=0
        while(stage<len(self.stage_params)):
            params = self.stage_params[stage]
            G.logger.info("Start stage %s", str(stage))

            if params['batch_size'] is not None:
                self.gapdl.update_batch_size(
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
                    train_loader=self.gapdl.train_loader,
                    val_loader=self.gapdl.val_loader,
                    )
            self.oc.train_one_cycle()
            self.do_prediction('')
            stage+=1

            if mode=="DEV" and stage==2:
                break

    def do_cycles_train_old(self):
        'stage1'
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
        oc.update_bot(optimizer=optimizer,
                scheduler="Default Triangular",
                unfreeze_layers=[model.head],
                n_epoch=10,
                stage='1',
                )
        oc.train_one_cycle()

        'stage2'
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
        oc.update_bot(optimizer = optimizer,
                scheduler="Default Triangular",
                unfreeze_layers=[model.head],
                n_epoch=20,
                stage='2',
                )
        oc.train_one_cycle()

        'stage3'
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
        oc.update_bot(optimizer = optimizer,
                scheduler="Default Triangular",
                unfreeze_layers=[model.head],
                n_epoch=20,
                stage='3',
                )
        oc.train_one_cycle()


        'stage4'
        gapdl.update_batch_size(train_size=8,val_size=128,test_size=128)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-2)
        oc.update_bot(optimizer = optimizer,
                scheduler="Default Triangular",
                train_loader=gapdl.train_loader,
                val_loader=gapdl.val_loader,
                unfreeze_layers=[model.head, model.bert.encoder],
                n_epoch=20,
                stage='4',
                )
        oc.train_one_cycle()

    def do_prediction(self, target_path=''):
        if target_path != '':
            self.bot.load_model(target_path)

        self.outputs, targets = self.bot.predict(self.gapdl.test_loader, return_y=True)
        self.bot.metrics(self.outputs, targets)

    def do_ensemble(self, checkpoint_path='', pattern='', eval=False):
        if eval:
            self.outputs, targets = self.bot.predict_avg\
                    (self.gapdl.test_loader, checkpoint_path, pattern, eval)
            self.bot.metrics(self.outputs, targets)
        else:
            self.outputs = self.bot.predict_avg\
                    (self.gapdl.submission_loader, checkpoint_path, pattern, eval)

    def do_blending(self, checkpoint_path='', pattern=''):
        self.bot.blending(self.gapdl, checkpoint_path, pattern)

    def do_submission(self):
        # Write the prediction to file for submission
        self.bot.submission(nn.functional.softmax(self.outputs,dim=1), self.sample_sub)

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
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.model.fc, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 5 if mode=="EXP" else 1,
                }
            ]*3,
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.model.fc, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }
            ]*3,
        ]
        self.params = [j for sub in self.params for j in sub]
        return self.params


    def step_scheduler(self):
        '''
        simple step schedulre as baseline
        '''
        adam = Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4)
        self.params = \
            [
                {
                    'optimizer': adam,
                    'batch_size': [20,128,128],
                    'scheduler': StepLR(adam, 1000, gamma=0.5, last_epoch=-1),
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 50 if mode=="EXP" else 1,
                },
            ]
        return self.params

    def increase_dropout(self):
        '''
        remove BERT dropout, if don't train BERT
        slowly decrease dropout ratio in HEAD when finetune
        maybe final finetune...
        '''
        self.params = \
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4),
                    'batch_size': [40,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [(self.model.bert, 0.0)],
                    'accu_gradient_step': None,
                    'epoch': 5 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-4),
                    'batch_size': [40,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [(self.model.bert, 0.0),
                                      (self.model.head, 0.60)],
                    'accu_gradient_step': None,
                    'epoch': 10 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-4),
                    'batch_size': [40,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [(self.model.bert, 0.00),
                                      (self.model.head, 0.60)],
                    'accu_gradient_step': None,
                    'epoch': 10 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-5,weight_decay=1e-4),
                    'batch_size': [40,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio':  [(self.model.bert, 0.00),
                                      (self.model.head, 0.60)],
                    'accu_gradient_step': None,
                    'epoch': 10 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-5,weight_decay=1e-4),
                    'batch_size': [40,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio':  [(self.model.bert, 0.0),
                                      (self.model.head, 0.7)],
                    'accu_gradient_step': None,
                    'epoch': 10 if mode=="EXP" else 1,
                },
            ]
        return self.params

    def baseline(self):
        '''
        baseline, one cycle train, with reducing lr after one cycle
        '''
        self.params = \
        [
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.embedding, nn.Module),
                                        (self.model.span_extractor, nn.Module),
                                        (self.model.pair_score, nn.Module),
                                        ],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 5 if mode=="EXP" else 1,
                }
            ]*1,
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.embedding, nn.Module),
                                        (self.model.span_extractor, nn.Module),
                                        (self.model.pair_score, nn.Module),
                                        ],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }
            ]*3,
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=5e-4,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.embedding, nn.Module),
                                        (self.model.span_extractor, nn.Module),
                                        (self.model.pair_score, nn.Module),
                                        ],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }
            ]*3,
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=2.5e-4,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.embedding, nn.Module),
                                        (self.model.span_extractor, nn.Module),
                                        (self.model.pair_score, nn.Module),
                                        ],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }
            ]*3
        ]
        self.params = [j for sub in self.params for j in sub]
        return self.params

    def baseline_continue(self):
        '''
        baseline continue training
        '''
        self.params = itertools.chain(\
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.embedding, nn.Module),
                                        (self.model.span_extractor, nn.Module),
                                        (self.model.pair_score, nn.Module),
                                        ],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }
            ]*1,
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=5e-4,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.embedding, nn.Module),
                                        (self.model.span_extractor, nn.Module),
                                        (self.model.pair_score, nn.Module),
                                        ],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }
            ]*2,
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=2.5e-4,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.embedding, nn.Module),
                                        (self.model.span_extractor, nn.Module),
                                        (self.model.pair_score, nn.Module),
                                        ],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }
            ]*3
        )
        self.params = itertools.chain(self.params, self.params)
        return self.params

    def baseline_then_finetune(self):
        '''
        baseline, one cycle train, with reducing lr after one cycle
        '''
        self.params = \
        [
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.embedding, nn.Module),
                                        (self.model.span_extractor, nn.Module),
                                        (self.model.pair_score, nn.Module),
                                        ],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 5 if mode=="EXP" else 1,
                }
            ]*1,
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.embedding, nn.Module),
                                        (self.model.span_extractor, nn.Module),
                                        (self.model.pair_score, nn.Module),
                                        ],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }
            ]*3,
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=5e-4,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.embedding, nn.Module),
                                        (self.model.span_extractor, nn.Module),
                                        (self.model.pair_score, nn.Module),
                                        ],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }
            ]*3,
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=2.5e-4,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.embedding, nn.Module),
                                        (self.model.span_extractor, nn.Module),
                                        (self.model.pair_score, nn.Module),
                                        ],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }
            ]*3,
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.embedding, nn.Module),
                                        (self.model.span_extractor, nn.Module),
                                        (self.model.pair_score, nn.Module),
                                        ],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }
            ]*1,
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=5e-4,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.embedding, nn.Module),
                                        (self.model.span_extractor, nn.Module),
                                        (self.model.pair_score, nn.Module),
                                        ],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }
            ]*2,
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=2.5e-4,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.embedding, nn.Module),
                                        (self.model.span_extractor, nn.Module),
                                        (self.model.pair_score, nn.Module),
                                        ],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }
            ]*3,
            [
                {
                    'optimizer': Adam([{'params':self.model.bert.parameters(),'lr':1e-6},],
                        lr=1e-5, weight_decay=1e-3),
                    'batch_size': [8,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),],
                    'dropout_ratio': [(self.model.bert, 0.35)],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }
            ]*4,
            [
                {
                    'optimizer': Adam([{'params':self.model.bert.parameters(),'lr':5e-7},],
                        lr=5e-6, weight_decay=1e-4),
                    'batch_size': [8,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),],
                    'dropout_ratio': [(self.model.bert, 0.35)],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }
            ]*4,
            [
                {
                    'optimizer': Adam([{'params':self.model.bert.parameters(),'lr':2e-7},],
                        lr=2e-6, weight_decay=1e-4),
                    'batch_size': [8,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),],
                    'dropout_ratio': [(self.model.bert, 0.35)],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }
            ]*4,
        ]
        self.params = [j for sub in self.params for j in sub]
        return self.params

    def accumulated_gradient(self):
        '''
        warm up head and init BN without accu_gradient
        then turn off BN train and using accu_gradient to reduce var
        without training BERT
        '''
        self.params = \
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-3),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 5 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-3),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert, nn.Module),
                                      (self.model.head, _BatchNorm)],
                    'dropout_ratio': [],
                    'accu_gradient_step': 10,
                    'epoch': 10 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-3),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert, nn.Module),
                                      (self.model.head, _BatchNorm)],
                    'dropout_ratio': [],
                    'accu_gradient_step': 10,
                    'epoch': 20 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-3),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert, nn.Module),
                                      (self.model.head, _BatchNorm)],
                    'dropout_ratio': [],
                    'accu_gradient_step': 10,
                    'epoch': 20 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-5,weight_decay=1e-3),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert, nn.Module),
                                      (self.model.head, _BatchNorm)],
                    'dropout_ratio': [],
                    'accu_gradient_step': 10,
                    'epoch': 20 if mode=="EXP" else 1,
                },
            ]
        return self.params

    def unfreeze_bert(self):
        '''
        inital warm up training head,
        then unfreeze bert to train all model
        if using gradient checkpoint, need to turn off dropout
        if using accumulated gradient, need to turn off BN
        '''
        self.params = \
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-3),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 2 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(
                        [{'params':self.model.head.parameters(),'lr':1e-4},
                         {'params':self.model.bert.parameters(),'lr':1e-5},],
                        weight_decay=1e-3),
                    'batch_size': [6,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module)],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 10 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(
                        [{'params':self.model.head.parameters(),'lr':1e-5},
                         {'params':self.model.bert.parameters(),'lr':1e-6},],
                        weight_decay=1e-3),
                    'batch_size': [6,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                },
                # {
                    # 'optimizer': Adam(
                        # [{'params':self.model.head.parameters(),'lr':1e-4},
                         # {'params':self.model.bert.parameters(),'lr':1e-5},],
                        # weight_decay=1e-3),
                    # 'batch_size': [6,128,128],
                    # 'scheduler': "Default Triangular",
                    # 'unfreeze_layers': [],
                    # 'freeze_layers': [(self.model.bert.embeddings,nn.Module),],
                    # 'dropout_ratio': [],
                    # 'accu_gradient_step': None,
                    # 'epoch': 20 if mode=="EXP" else 1,
                # },
                # {
                    # 'optimizer': Adam(
                        # [{'params':self.model.head.parameters(),'lr':1e-5},
                         # {'params':self.model.bert.parameters(),'lr':1e-6},],
                        # weight_decay=1e-3),
                    # 'batch_size': [6,128,128],
                    # 'scheduler': "Default Triangular",
                    # 'unfreeze_layers': [],
                    # 'dropout_ratio': [],
                    # 'freeze_layers': [(self.model.bert.embeddings,nn.Module),],
                    # 'accu_gradient_step': None,
                    # 'epoch': 20 if mode=="EXP" else 1,
                # },
            ]
        return self.params

    def unfreeze_bert_with_accu_gradient(self):
        '''
        !!!! do not inital warm up training head,
        !!!! it will cause fail to train..
        then unfreeze bert to train all model
        freeze batch norm, bert layer norm
        for using accu gradient to reduce variance as batch size is too small
        BERT is very sensitive, need to train under very small LR
        add dropout adjustment, as previous have huge performance gap
        '''
        self.params = \
            [
                # {
                    # 'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-3),
                    # 'batch_size': [40,128,128],
                    # 'scheduler': "Default Triangular",
                    # 'unfreeze_layers': [(self.model.head, nn.Module)],
                    # 'freeze_layers': [],
                    # 'dropout_ratio': [],
                    # 'accu_gradient_step': None,
                    # 'epoch': 5 if mode=="EXP" else 1,
                # },
                {
                    'optimizer': Adam(
                        [{'params':self.model.head.parameters(),'lr':1e-3},
                         {'params':self.model.bert.parameters(),'lr':1e-4},],
                        weight_decay=1e-3),
                    'batch_size': [4,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),
                                      (self.model,(_BatchNorm, BertLayerNorm))],
                    'dropout_ratio': [],
                    'accu_gradient_step': 100,
                    'epoch': 20 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(
                        [{'params':self.model.head.parameters(),'lr':1e-3},
                         {'params':self.model.bert.parameters(),'lr':1e-4},],
                        weight_decay=1e-3),
                    'batch_size': [4,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),
                                      (self.model,(_BatchNorm, BertLayerNorm))],
                    'dropout_ratio': [],
                    'accu_gradient_step': 100,
                    'epoch': 20 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(
                        [{'params':self.model.head.parameters(),'lr':1e-4},
                         {'params':self.model.bert.parameters(),'lr':1e-5},],
                        weight_decay=1e-3),
                    'batch_size': [4,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),
                                      (self.model,(_BatchNorm, BertLayerNorm))],
                    'dropout_ratio': [],
                    'accu_gradient_step': 100,
                    'epoch': 20 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(
                        [{'params':self.model.head.parameters(),'lr':1e-4},
                         {'params':self.model.bert.parameters(),'lr':1e-5},],
                        weight_decay=1e-3),
                    'batch_size': [4,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),
                                      (self.model,(_BatchNorm,BertLayerNorm))],
                    'dropout_ratio': [],
                    'accu_gradient_step': 100,
                    'epoch': 20 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(
                        [{'params':self.model.head.parameters(),'lr':1e-5},
                         {'params':self.model.bert.parameters(),'lr':1e-6},],
                        weight_decay=1e-3),
                    'batch_size': [4,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),
                                      (self.model,(_BatchNorm,BertLayerNorm))],
                    'dropout_ratio': [],
                    'accu_gradient_step': 100,
                    'epoch': 20 if mode=="EXP" else 1,
                },

            ]
        return self.params

    def finetune_bert(self):
        '''
        one cycle train as baseline,
        only fine tune whole model in last cycle
        its better not to turn off dropout, as it will cause overfitting
        '''
        self.params = itertools.chain(\
            [
               {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 5 if mode=="EXP" else 1,
                }
            ]*1,
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 10 if mode=="EXP" else 1,
                }
            ]*3,
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }
            ]*3,
            [
                {
                    'optimizer': Adam(
                        [{'params':self.model.head.parameters(),'lr':1e-5},
                         {'params':self.model.bert.parameters(),'lr':1e-6},],
                        weight_decay=1e-4),
                    'batch_size': [6,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'dropout_ratio': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }
            ]*3,
            [
                {
                    'optimizer': Adam(
                        [{'params':self.model.head.parameters(),'lr':1e-6},
                         {'params':self.model.bert.parameters(),'lr':1e-7},],
                        weight_decay=1e-4),
                    'batch_size': [6,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'dropout_ratio': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }
            ]*3
        )
        return self.params

    def finetune_bert_continue(self):
        '''
        only fine tune whole model in last cycle
        its better not to turn off dropout, as it will cause overfitting
        '''
        self.params = \
        [
            [
                {
                    'optimizer': Adam([{'params':self.model.bert.parameters(),'lr':5e-7},],
                        lr=5e-6, weight_decay=1e-4),
                    'batch_size': [8,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),],
                    'dropout_ratio': [(self.model.bert, 0.35)],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }
            ]*2,
            [
                {
                    'optimizer': Adam([{'params':self.model.bert.parameters(),'lr':2e-7},],
                        lr=2e-6, weight_decay=1e-4),
                    'batch_size': [8,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),],
                    'dropout_ratio': [(self.model.bert, 0.35)],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }
            ]*2,
        ]

        self.params = [j for sub in self.params for j in sub]
        self.params = self.params*2
        return self.params



