# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collection of wrapper for training and evaluting pytorch model.
"""
from typing import List, Tuple
import warnings 
import pandas as pd
from loguru import logger
from apex import amp
import numpy as np
import torch
import torch.optim as optim
import optuna
from ninstd.check import is_imported
from .utils import AvgMeter, torch_cpu_or_gpu

__all__ = [
    'Trainer',
    'HalfTrainer'
]


class Trainer(object):
    """Class responsed for training and testing.
    """
    def __init__(
            self, model=None, optim=None, loss_func=None,
            train_loader=None, valid_loader=None,
            test_loader=None, scheduler=None, writer=None,
            *args, **kwargs):

        self.model = model
        self.optim = optim
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.scheduler = scheduler
        self.writer = writer
        self.device = torch_cpu_or_gpu()
        self.epoch_idx = 0
        self.dfs = {}

    @staticmethod
    def gen_empty_df(cols: List[str]) -> pd.DataFrame:
        """Generate empty dataframe given column names.
        """
        df = pd.DataFrame(columns=cols)
        return df

    def get_best(self, name_df: str, col: str, direction: str = 'max') -> float:
        """Return the best value in either `max` or `min` direction from self.dfs.
        """
        assert isinstance(name_df, str)
        assert isinstance(col, str)
        assert isinstance(direction, str)
        if direction.lower() == 'max':
            return np.amax(self.dfs[name_df][col])
        elif direction.lower() == 'min':
            return np.amin(self.dfs[name_df][col])
        else:
            raise NotImplementedError(
                f'Direction: {direction} is not `max` or `min`.')

    def dfs_append_row(self, name_df: str, **kwargs) -> None:
        """Given name_df or key of dfs, access df within dfs.
        Append a row from the dict kwargs to that df.
        """
        keys = list(kwargs)
        wrapped_kwargs = {key: [kwargs[key]] for key in keys}
        df = pd.DataFrame(wrapped_kwargs)
        self.dfs[name_df] = self.dfs[name_df].append(df)

    def check_df_exists(self, name_df: str, cols: List[str]) -> None:
        """Check is name_df is in the dfs or not.
        If it is not exist, generate new df in self.dfs.
        Else replace the variable in self.dfs to empty df.
        """
        assert isinstance(name_df, str)
        if name_df not in self.dfs.keys():
            df = self.gen_empty_df(cols)
            self.dfs.update({name_df: df})

    def _check_train(self) -> None:
        assert self.train_loader is not None
        assert self.model is not None
        assert self.optim is not None
        assert self.loss_func is not None

    def _check_valid(self) -> None:
        assert self.valid_loader is not None
        assert self.model is not None
        assert self.loss_func is not None

    def _check_test(self) -> None:
        assert self.test_loader is not None
        assert self.model is not None
        assert self.loss_func is not None

    def log_info(
            self, header: str, epoch: int, acc: float, loss: float):
        if self.optim is not None:
            lr = self.optim.param_groups[0]['lr']
        else:
            raise NotImplementedError('self.optim is not defined.')
        try:
            logger.info(
                f'{header} Epoch: {epoch}, Accuracy: {acc}, Loss: {loss}, Lr: {lr}')
        except NameError:
            # If cannot detect the logger, then using print out.
            print(
                f'{header} Epoch: {epoch}, Accuracy: {acc}, Loss: {loss}, Lr: {lr}')
        
    @staticmethod
    def log_info_any(header: str, *args, **kwargs):
        accum_string = f'{header} '
        for kw in kwargs.keys():
            accum_string += f'{kw.capitalize()} {kwargs[kw]} '
        try:
            logger.info(accum_string)
        except NameError:
            print(accum_string)
        
    def add_scalars(
        self, group_name: str, updating_dict: dict, idx: int) -> None:
        """Adding information into a writer, Tensorboard.
        """
        assert self.writer is not None
        self.writer.add_scalars(
            group_name, updating_dict, global_step=idx)
        
    def forwarding(self, data, label):
        """For forwarding a model.
        Need to using with the with torch.no_grad() in case evalutation.
        """
        batch = label.data.size(0)
        data, label = data.to(self.device), label.to(self.device)
        pred = self.model.forward(data)
        loss = self.loss_func(pred, label)
        _, max_pred = torch.max(pred.data, 1)
        correct = (max_pred == label).sum().item()
        return correct, loss, batch
    
    def forwarding_and_updating(self, data, label):
        """For training only wrapper of forwarding.
        Adding necessary function for back-propagation.
        """
        self.optim.zero_grad()
        correct, loss, batch = self.forwarding(data, label)
        loss.backward()
        self.optim.step()
        return correct, loss, batch

    def train_an_epoch(self, verbose: int=0):
        self._check_train()
        HEADER: str = 'training'
        self.epoch_idx += 1
        avg_acc = AvgMeter()
        avg_loss = AvgMeter()
        self.check_df_exists(HEADER, ['acc', 'loss'])

        self.model.train()
        for train_data, train_label in self.train_loader:
            correct, loss, batch = self.forwarding_and_updating(
                train_data, train_label)
            avg_acc(correct, batch)
            avg_loss(loss.item(), batch)

        self.dfs_append_row(HEADER, acc=avg_acc.avg, loss=avg_loss.avg)

        if self.scheduler is not None:
            self.scheduler.step()

        if self.writer is not None:
            self.add_scalars(
                group_name=HEADER,
                updating_dict={
                    'train_acc': avg_acc.avg,
                    'train_loss': avg_loss.avg},
                idx=self.epoch_idx)

        if verbose > 0:
            self.log_info(
                header=HEADER, epoch=self.epoch_idx,
                acc=avg_acc.avg, loss=avg_loss.avg)
        return avg_acc.avg, avg_loss.avg

    def test_an_epoch(self, verbose: int=0):
        self._check_test()
        avg_acc = AvgMeter()
        avg_loss = AvgMeter()
        HEADER: str = 'testing'
        self.check_df_exists(HEADER, ['acc', 'loss'])

        self.model.eval()
        with torch.no_grad():
            for test_data, test_label in self.test_loader:
                correct, loss, batch = self.forwarding(
                    test_data, test_label)
                avg_acc(correct, batch)
                avg_loss(loss.item(), batch)

        self.dfs_append_row(HEADER, acc=avg_acc.avg, loss=avg_loss.avg)
        
        if self.writer is not None:
            self.add_scalars(
                group_name=HEADER,
                updating_dict={
                    'test_acc': avg_acc.avg,
                    'test_loss': avg_loss.avg},
                idx=self.epoch_idx)

        if verbose > 0:
            self.log_info(
                header=HEADER, epoch=self.epoch_idx,
                acc=avg_acc.avg, loss=avg_loss.avg)
        return avg_acc.avg, avg_loss.avg

    def valid_an_epoch(self, verbose: int=0):
        self._check_valid()
        avg_acc = AvgMeter()
        avg_loss = AvgMeter()
        HEADER: str = 'validation'
        self.check_df_exists(HEADER, ['acc', 'loss'])

        self.model.eval()
        with torch.no_grad():
            for valid_data, valid_label in self.valid_loader:
                correct, loss, batch = self.forwarding(
                    valid_data, valid_label)
                avg_acc(correct, batch)
                avg_loss(loss.item(), batch)

        self.dfs_append_row(HEADER, acc=avg_acc.avg, loss=avg_loss.avg)

        if self.writer is not None:
            self.add_scalars(
                group_name=HEADER,
                updating_dict={'valid_acc': avg_acc.avg, 'valid_loss': avg_loss.avg},
                idx=self.epoch_idx)

        if verbose > 0:
            self.log_info(
                header=HEADER, epoch=self.epoch_idx,
                acc=avg_acc.avg, loss=avg_loss.avg)

        return avg_acc.avg, avg_loss.avg

    def warm_up_lr(self, terminal_lr: float, verbose: int = 0):
        """Fix as the wrapper of the train_an_epoch.
        Using warm up learning from:
            https://stackoverflow.com/questions/55933867/what-does-learning-rate-warm-up-mean
        """
        self._check_train()
        avg_acc = AvgMeter()
        avg_loss = AvgMeter()
        HEADER: str = 'warmup'
        self.check_df_exists(HEADER, ['acc', 'loss'])

        self.model.train()
        for idx, (train_data, train_label) in enumerate(self.train_loader):
            warm_up_lr = terminal_lr*(idx/len(self.train_loader))
            self.optim.param_groups[0]['lr'] = warm_up_lr
            correct, loss, batch = self.forwarding_and_updating(
                train_data, train_label)
            avg_acc(correct, batch)
            avg_loss(loss.item(), batch)
       
        self.dfs_append_row(HEADER, acc=avg_acc.avg, loss=avg_loss.avg)

        if self.scheduler is not None:
            self.scheduler.step()

        if verbose > 0:
            self.log_info(
               header=HEADER, epoch=self.epoch_idx,
                acc=avg_acc.avg, loss=avg_loss.avg)

        return avg_acc.avg, avg_loss.avg

    def swap_to_sgd(self, lr: float = None) -> None:
        """Swap into SGD with provided value.
        """
        if lr is None:
            if self.optim is None:
                # TODO: making my own exception?
                raise RuntimeError('')
            else:
                lr = self.optim.param_groups[0]['lr']
        self.optim = optim.SGD(self.model.parameters(), lr)


class HalfTrainer(Trainer):
    """Trainer with the supporting of the mixed precision training.
    """
    def __init__(self, *args, **kwargs):
        super(HalfTrainer, self).__init__(*args, **kwargs)
        self._half_flag = False
        self.opt_level = None
        
    def _check_half(self):
        assert self.model is not None
        assert self.optim is not None
        assert self.loss_func is not None
        assert is_imported('apex.amp')
    
    def to_half(self, opt_level: str = 'O1', verbose: int = 1):
        """To half precision using Apex module.
        More details: https://nvidia.github.io/apex/amp.html
        TODO: checking with regularly trained model which one is faster.
        TODO: gradient overflow problem very 2 epoches.
        """
        assert isinstance(opt_level, str)
        assert isinstance(verbose, int)
        assert opt_level in ['O0', 'O1', 'O2', 'O3']
        self._check_half()
        self.model, self.optim = amp.initialize(
            self.model, self.optim, opt_level=opt_level, verbosity=verbose)
        if self.scheduler is not None:
            self.scheduler.optimizer = self.optim
        # Setting for detect the half precision during training, testing or saving.
        self._half_flag = True
        self.opt_level = opt_level
    
    def forwarding_and_updating(self, data, label) -> Tuple[str, str, str]:
        """
        """
        batch = label.data.size(0)
        data, label = data.to(self.device), label.to(self.device)
        self.optim.zero_grad()
        pred = self.model.forward(data)
        if not self._half_flag:
            loss = self.loss_func(pred, label)
            loss.backward()
        else:
            loss = self.loss_func(pred, label)
            with amp.scale_loss(loss, self.optim) as scaled_loss:
                scaled_loss.backward()
        _, max_pred = torch.max(pred.data, 1)
        correct = (max_pred == label).sum().item()
        self.optim.step()
        return correct, loss, batch
    
    def swap_to_sgd(self, lr: float = None) -> None:
        """Swap into SGD with provided value.
        """
        if lr is None:
            if self.optim is None:
                # TODO: making my own exception?
                raise RuntimeError(f'')
            else:
                lr = self.optim.param_groups[0]['lr']
 
        self.optim = optim.SGD(self.model.parameters(), lr)
        if self._half_flag:
            self.model, self.optim = amp.initialize(
                self.model, self.optim, opt_level=self.opt_level)

    def train_eval_epoches(
        self, epoch: int, eval_every_epoch: int,
        verbose: int, return_best: bool, trial_funct, trial) -> float:
        """Train and test for certain epoches with an option
        to turn on the hyper parameter tunning.
        For trial_funct, please follow nintorch.hyper.default_trial.
        Having problem with optuna, with class.method the constructor is not
        called for each trial of optuna. Solving by using self.init_model keep
        for reseting the model.
        """
        self._check_train()
        HEADER: str = 'Train'
        self.check_df_exists(HEADER, ['acc', 'loss'])
        if self.valid_or_test:
            self._check_valid()
            HEADER_EVAL: str = 'Validation'
        else:
            self._check_test()
            HEADER_EVAL: str = 'Test'
        kwargs = trial_funct(trial)
        self.check_df_exists(HEADER_EVAL, ['acc', 'loss'])

        # TODO: cover more than lr, weight decay and momentum.
        if 'lr' in kwargs:
            self.optim.param_groups[0]['lr'] = kwargs['lr']
        if 'weight_decay' in kwargs:
            self.optim.param_groups[0]['weight_decay'] = kwargs['weight_decay']
        if 'momentum' in kwargs:
            self.optim.param_groups[0]['momentum'] = kwargs['momentum']
        if len(kwargs) > 3:
            warnings.warn('kwargs is more than 3 might not supported.', UserWarning)

        pbar = range(epoch)
        if verbose > 0:
            pbar = tqdm(pbar)
        
        for i in pbar:
            train_acc, train_loss = self.train_an_epoch()
            
            if verbose > 0:
                self.log_info(
                   header=HEADER, epoch=self.epoch_idx,
                    acc=train_acc, loss=train_loss)
            self.dfs_append_row(HEADER, acc=train_acc, loss=train_loss)

            if i % eval_every_epoch == 0 and i != 0:
                if self.valid_or_test:
                    eval_acc, eval_loss = self.valid_an_epoch()
                else:
                    eval_acc, eval_loss = self.test_an_epoch()

                if verbose > 0:
                    self.log_info(
                       header=HEADER_EVAL, epoch=self.epoch_idx,
                        acc=eval_acc, loss=eval_loss)

                self.dfs_append_row(HEADER_EVAL, acc=eval_acc, loss=eval_loss)
                trial.report(eval_acc, i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if return_best:
            best_acc = self.get_best(HEADER_EVAL, 'acc', 'max')
            return best_acc
        else:
            return eval_acc


