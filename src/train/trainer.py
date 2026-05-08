import copy
import logging

import numpy as np
# import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from src.dataset import dataset
from src.dataset.dataset import load_specified_dataset
from src.dataset.data_processor import DataProcessor
from src.evaluation.estimator import Estimator
from src.utils.recorder import Recorder
import src.model as model
from src.train.config import experiment_hyper_load, config_override
from src.utils.utils import set_seed, tensor_to_device
import pytorch_warmup as warmup


def load_trainer(config):
    if config.model in ['FHMEX']:
        return BMRTrainer(config)
    return Trainer(config)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.model_name = config.model
        self._config_override(self.model_name, config)
        # pretraining
        self.pretraining_model = None
        self.do_pretraining = self.config.do_pretraining
        self.pretraining_task = self.config.pretraining_task
        self.pretraining_epoch = self.config.pretraining_epoch
        self.pretraining_batch = self.config.pretraining_batch
        self.pretraining_lr = self.config.pretraining_lr
        self.pretraining_l2 = self.config.pretraining_l2

        # training
        self.training_model = None
        self.num_worker = self.config.num_worker
        self.train_batch = self.config.train_batch
        self.eval_batch = self.config.eval_batch
        self.lr = self.config.learning_rate
        self.l2 = self.config.l2
        self.epoch_num = self.config.epoch_num
        self.dev = torch.device(self.config.device)
        self.split_type = self._set_split_mode(self.config.split_type)
        self.do_test = self.split_type == 'valid_and_test'
        self.do_test_with_eval = self.config.do_test_with_eval and self.do_test

        # set random seed
        set_seed(self.config.seed)

        # components
        self.data_processor = DataProcessor(self.config)
        self.estimator = Estimator(self.config)
        self.recorder = Recorder(self.config)

        # # set random seed
        # set_seed(self.config.seed)

        # preparing data
        data_dict, additional_data_dict = self.data_processor.prepare_data()
        self.data_dict = data_dict  # store standard train/eval/test data
        self.additional_data_dict = additional_data_dict  # extra data (model specified)
        # self.estimator.load_item_popularity(self.data_processor.popularity)

    def start_training(self):
        if self.do_pretraining:
            self.pretrain()
        self.train()

    def pretrain(self):
        if self.pretraining_task in ['MISP', 'MIM', 'PID']:
            pretrain_dataset = getattr(dataset, f'{self.pretraining_task}PretrainDataset')
            pretrain_dataset = pretrain_dataset(self.config, self.data_dict['train'],
                                                self.additional_data_dict)
        else:
            raise NotImplementedError(f'No such pretraining task: {self.pretraining_task}, '
                                      f'choosing from [MIP, MIM, PID]')
        train_loader = DataLoader(pretrain_dataset, batch_size=self.train_batch, collate_fn=pretrain_dataset.collate_fn,
                                  shuffle=True, num_workers=0, drop_last=False)
        # data_ele to device
        tensor_to_device(self.additional_data_dict, self.dev)

        pretrain_model = self._load_model()

        opt = torch.optim.Adam(filter(lambda x: x.requires_grad, pretrain_model.parameters()),
                               self.pretraining_lr, weight_decay=self.pretraining_l2)

        self.experiment_setting_verbose(pretrain_model, training=False)

        logging.info('Start pretraining...')
        for epoch in range(self.pretraining_epoch):
            pretrain_model.train()
            self.recorder.epoch_restart()
            self.recorder.tik_start()
            train_iter = tqdm(enumerate(train_loader), total=len(train_loader))
            train_iter.set_description(f'pretraining  epoch: {epoch}')
            for i, batch_dict in train_iter:
                tensor_to_device(batch_dict, self.dev)
                loss = getattr(pretrain_model, f'{self.pretraining_task}_pretrain_forward')(batch_dict)
                opt.zero_grad()
                loss.backward()
                opt.step()

                self.recorder.save_batch_loss(loss.item())
            self.recorder.tik_end()
            self.recorder.train_log_verbose(len(train_loader))

        self.pretraining_model = pretrain_model
        logging.info('Pre-training is over, prepare for training...')

    def train(self):
        SpecifiedDataSet = load_specified_dataset(self.model_name, self.config)
        train_dataset = SpecifiedDataSet(self.config, self.data_dict['train'],
                                         self.additional_data_dict)
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch,
                                  collate_fn=train_dataset.collate_fn, shuffle=True)

        eval_dataset = SpecifiedDataSet(self.config, self.data_dict['eval'],
                                        self.additional_data_dict, mode='eval')
        eval_loader = DataLoader(eval_dataset, batch_size=self.eval_batch,
                                 collate_fn=eval_dataset.collate_fn, shuffle=False)

        # data_ele to device
        tensor_to_device(self.additional_data_dict, self.dev)
        self.training_model = self._load_model()

        opt = torch.optim.AdamW(filter(lambda x: x.requires_grad, self.training_model.parameters()),
                                betas=(0.9, 0.98), lr=self.lr, weight_decay=self.l2)
        self.recorder.reset()
        self.experiment_setting_verbose(self.training_model)

        logging.info('Start training...')
        for epoch in range(self.epoch_num):
            self.training_model.train()
            self.recorder.epoch_restart()
            self.recorder.tik_start()
            train_iter = tqdm(enumerate(train_loader), total=len(train_loader))
            train_iter.set_description('training  ')
            for step, batch_dict in train_iter:
                # training forward
                batch_dict['epoch'] = epoch
                batch_dict['step'] = step
                tensor_to_device(batch_dict, self.dev)
                loss = self.training_model.calc_loss(batch_dict)
                if torch.is_tensor(loss) and loss.requires_grad:
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.training_model.parameters(), max_norm=1.0)
                    opt.step()
                    self.recorder.save_batch_loss(loss.item())
            self.recorder.tik_end()
            self.recorder.train_log_verbose(len(train_loader))

            # evaluation
            self.recorder.tik_start()
            eval_metric_result, eval_loss = self.estimator.evaluate(eval_loader, self.training_model)
            self.recorder.tik_end(mode='eval')
            self.recorder.log_verbose_and_save(eval_metric_result, eval_loss, self.training_model)

            if self.do_test_with_eval:
                test_metric_res = self.test_model(self.data_dict['test'])
                self.recorder.report_test_result(test_metric_res)

            if self.recorder.early_stop:
                break

        self.recorder.report_best_res()
        # test model
        if self.do_test:
            test_metric_res, test_loss = self.test_model(self.data_dict['test'])
            self.recorder.report_test_result(test_metric_res)

    def _set_split_mode(self, split_mode):
        assert split_mode in ['valid_and_test', 'valid_only'], f'Invalid split mode: {split_mode} !'
        return split_mode

    def _load_model(self):
        if self.do_pretraining and self.pretraining_model is not None:  # return pretraining model
            return self.pretraining_model

        # return new model
        if self.config.model_type.upper() == 'SEQUENTIAL':
            return self._load_sequential_model()
        elif self.config.model_type.upper() in ['GRAPH', 'KNOWLEDGE']:
            return self._load_model_with_additional_data()
        else:
            return self._load_sequential_model()

    def _load_sequential_model(self):
        Model = getattr(model, self.model_name)
        specified_seq_model = Model(self.config, self.additional_data_dict).to(self.dev)
        return specified_seq_model

    def _load_model_with_additional_data(self):
        Model = getattr(model, self.model_name)
        specified_model = Model(self.config, self.additional_data_dict).to(self.dev)
        return specified_model

    def _config_override(self, model_name, cmd_config):
        self.model_config = getattr(model, f'{model_name}_config')()
        self.config = config_override(self.model_config, cmd_config)
        # capitalize
        self.config.model_type = self.config.model_type.upper()
        self.config.graph_type = [g_type.upper() for g_type in self.config.graph_type]

    def experiment_setting_verbose(self, model, training=True):
        if self.do_pretraining and training:
            return
        # model config
        logging.info('[1] Model Hyper-Parameter '.ljust(47, '-'))
        model_param_set = self.model_config.keys()
        for arg in vars(self.config):
            if arg in model_param_set:
                logging.info(f'{arg}: {getattr(self.config, arg)}')
        # experiment config
        logging.info('[2] Experiment Hyper-Parameter '.ljust(47, '-'))
        # verbose_order = ['Data', 'Training', 'Evaluation', 'Save']
        hyper_types, exp_setting = experiment_hyper_load(self.config)
        for i, hyper_type in enumerate(hyper_types):
            hyper_start_log = (f'[2-{i + 1}] ' + hyper_type.lower() + ' hyper-parameter ').ljust(47, '-')
            logging.info(hyper_start_log)
            for hyper, value in exp_setting[hyper_type].items():
                logging.info(f'{hyper}: {value}')
        # data statistic
        self.data_processor.data_log_verbose(3)
        # model architecture
        self.report_model_info(model)

    def report_model_info(self, model):
        # model architecture
        logging.info('[1] Model Architecture '.ljust(47, '-'))
        logging.info(f'total parameters: {model.calc_total_params()}')
        logging.info(model)

    def test_model(self, test_data_pair=None):
        SpecifiedDataSet = load_specified_dataset(self.model_name, self.config)
        test_dataset = SpecifiedDataSet(self.config, test_data_pair,
                                        self.additional_data_dict, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=self.eval_batch,
                                 collate_fn=test_dataset.collate_fn, shuffle=False)
        # load the best model
        self.recorder.load_best_model(self.training_model)
        self.training_model.eval()

        # test_metric_result = self.estimator.test(test_loader, self.training_model)
        test_metric_result = self.estimator.test(test_loader, self.training_model)

        return test_metric_result

    def start_test(self):
        # data_ele to device
        tensor_to_device(self.additional_data_dict, self.dev)
        self.training_model = self._load_model()
        self.experiment_setting_verbose(self.training_model)
        test_metric_res = self.test_model(self.data_dict['test'])
        self.recorder.report_test_result(test_metric_res)


class BMRTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        logging.info('Use BMRTrainer.')

    def train(self):
        df_eval = self.data_dict['eval']
        df_test = self.data_dict['test']
        eq = df_eval.equals(df_test)
        print('Is df_eval equals df_test?', eq)

        SpecifiedDataSet = load_specified_dataset(self.model_name, self.config)
        train_dataset = SpecifiedDataSet(self.config, self.data_dict['train'],
                                         self.additional_data_dict)
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch,
                                  collate_fn=train_dataset.collate_fn, shuffle=True)

        eval_dataset = SpecifiedDataSet(self.config, self.data_dict['eval'],
                                        self.additional_data_dict, mode='eval')
        eval_loader = DataLoader(eval_dataset, batch_size=self.eval_batch,
                                 collate_fn=eval_dataset.collate_fn, shuffle=False)

        # data elements to device
        tensor_to_device(self.additional_data_dict, self.dev)
        self.training_model = self._load_model()

        optim_params_normal, optim_params_fast = [], []
        name_params_normal, name_params_fast = [], []

        finetune_encoders = False
        for k, v in self.training_model.named_parameters():
            if v.requires_grad:
                if "image_model" in k or "text_model" in k:
                    finetune_encoders = True
                    name_params_normal.append(k)
                    optim_params_normal.append(v)
                else:
                    name_params_fast.append(k)
                    optim_params_fast.append(v)
        if finetune_encoders:
            logging.info('Finetune image/text encoders.')
        num_steps = int(len(train_loader) * self.epoch_num * 1.1)
        if finetune_encoders:
            encoder_lr = 1e-5
            optimizer_normal = torch.optim.AdamW(
                optim_params_normal, lr=encoder_lr, betas=(0.9, 0.999), weight_decay=0.01
            )
            scheduler_normal = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_normal, T_max=num_steps)
            warmup_scheduler_normal = warmup.UntunedLinearWarmup(optimizer_normal)

        lr_fast = self.lr
        optimizer_fast = torch.optim.AdamW(
            optim_params_fast,
            lr=lr_fast,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        scheduler_fast = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_fast, T_max=num_steps
        )
        warmup_scheduler_fast = warmup.UntunedLinearWarmup(optimizer_fast)

        self.recorder.reset()
        self.experiment_setting_verbose(self.training_model)

        logging.info('Start training...')
        for epoch in range(self.epoch_num):
            self.training_model.train()
            self.recorder.epoch_restart()
            self.recorder.tik_start()
            train_iter = tqdm(enumerate(train_loader), total=len(train_loader))
            train_iter.set_description('training  ')
            for step, batch_dict in train_iter:
                # training forward
                batch_dict['epoch'] = epoch
                batch_dict['step'] = step
                tensor_to_device(batch_dict, self.dev)
                loss = self.training_model.calc_loss(batch_dict)

                # update gradients
                if torch.is_tensor(loss) and loss.requires_grad:
                    if finetune_encoders:
                        optimizer_normal.zero_grad()
                    optimizer_fast.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.training_model.parameters(), max_norm=1)
                    if epoch >= 10 and finetune_encoders:  # fine-tune MAE and BERT from the 10th epoch
                        optimizer_normal.step()
                    optimizer_fast.step()
                    if finetune_encoders:
                        with warmup_scheduler_normal.dampening():
                            scheduler_normal.step()
                    with warmup_scheduler_fast.dampening():
                        scheduler_fast.step()

                    self.recorder.save_batch_loss(loss.item())
            self.recorder.tik_end()
            self.recorder.train_log_verbose(len(train_loader))

            # evaluation
            self.recorder.tik_start()
            eval_metric_result, eval_loss = self.estimator.evaluate(eval_loader, self.training_model)
            self.recorder.tik_end(mode='eval')
            self.recorder.log_verbose_and_save(eval_metric_result, eval_loss, self.training_model)

            if self.do_test_with_eval:
                test_metric_res = self.test_model(self.data_dict['test'])
                self.recorder.report_test_result(test_metric_res)

            if self.recorder.early_stop:
                break

        self.recorder.report_best_res()
        # test model
        if self.do_test:
            test_metric_res, test_loss = self.test_model(self.data_dict['test'])
            self.recorder.report_test_result(test_metric_res)


if __name__ == '__main__':
    pass
