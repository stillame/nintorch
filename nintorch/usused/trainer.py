 # !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collection of wrapper for training and evaluting pytorch model.
"""
import warnings 
from typing import List, Tuple, Callable

import optuna
from tqdm import tqdm
from apex import amp
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from loguru import logger
from sklearn.metrics import confusion_matrix

from ninstd.check import is_imported
from .utils import AvgMeter, torch_cpu_or_gpu

__all__ = [
    'Trainer',
    'HyperTrainer',
    'HalfTrainer',
]


class Trainer(object):
    """Class responsed for training and evaluating.
    verbose: 0 for nothing, 1 for logging and 2 for progress bar.
    TODO: not automatically create df from training evaluating. Self create, let the user defines.
    """
    def __init__(
            self,
            model=None,
            optim=None,
            loss_func=None,
            train_loader=None,
            valid_loader=None,
            test_loader=None,
            scheduler=None,
            writer=None,
            *args,
            **kwargs) -> None:

        self.optim = optim
        self.loss_func = loss_func
        self.loaders: dict = {
            'train': train_loader, 
            'valid': valid_loader,
            'test': test_loader}
        self.scheduler = scheduler
        self.writer = writer
        self.device = torch_cpu_or_gpu()
        self.model = model.to(self.device)
        self.epoch_idx: int = 0
        self.dfs: dict = {}
        self.best_acc: float = 0.0

    def _check_train(self) -> None:
        assert self.loaders['train'] is not None
        assert self.model is not None
        assert self.optim is not None
        assert self.loss_func is not None

    def _check_eval(self) -> None:
        assert self.loaders['test'] is not None or self.loaders['valid'] is not None  
        assert self.model is not None
        assert self.loss_func is not None

    @staticmethod
    def gen_empty_df(cols: List[str]) -> pd.DataFrame:
        """Generate empty dataframe given column names.
        """
        df = pd.DataFrame(columns=cols)
        return df
    
    # TODO: arbitary supporting other name of loaders.
    # def add_loader(self, name_loader: str, loader) -> None:
    #     """Adding loader to self.loaders.
    #     """
    #     assert isinstance(name_loader, str)
    #     self.loaders.update({name_loader: loader})

    def get_best(self, name_df: str, col: str, direction: str = 'max') -> float:
        """Return the best value in either `max` or `min` direction from self.dfs.
        """
        assert isinstance(name_df, str)
        assert isinstance(col, str)
        assert isinstance(direction, str)
        
        print(self.dfs)
        
        if direction.lower() == 'max' or direction.lower() == 'maximum':
            return np.amax(self.dfs[name_df][col])
        elif direction.lower() == 'min' or direction.lower() == 'minimum':
            return np.amin(self.dfs[name_df][col])
        else:
            raise ValueError(
                f'Direction: {direction} is not `max`, `minimum`, `minimum` or `min`.')

    def dfs_append_row(self, name_df: str, **kwargs) -> None:
        """Given name_df or key of dfs, access df within dfs.
        Append a row from the dict kwargs to that df.
        """
        keys = list(kwargs)
        if hasattr(kwargs[keys[0]], 'numpy'):
            # Checking first variable is torch.Tensor or not.
            wrapped_kwargs = {key: [kwargs[key].numpy()] for key in keys}
        else:
            wrapped_kwargs = {key: [kwargs[key]] for key in keys}
        df = pd.DataFrame(wrapped_kwargs)
        self.dfs[name_df] = self.dfs[name_df].append(df)
        # TODO: Adding resetting the index!

    def check_df_exists(self, name_df: str, cols: List[str]) -> None:
        """Check is name_df is in the dfs or not.
        If it is not exist, generate new df in self.dfs.
        """
        assert isinstance(name_df, str)
        if name_df not in self.dfs.keys():
            df = self.gen_empty_df(cols)
            self.dfs.update({name_df: df})

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
 
    def predicting(self, data):
        """Predict data given string of name_loader.
        """
        assert self.model is not None
        with torch.no_grad():
            data = data.to(self.device)
            pred = self.model.forward(data)
        return pred

    def forwarding(self, data, label=None):
        """For forwarding a model.
        Need to using with the with torch.no_grad() in case evalutation.
        Designed to use for both training and evaluating.
        """
        batch = data.data.size(0)
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
        # Reusing from the self.forwarding.
        correct, loss, batch = self.forwarding(data, label)
        loss.backward()
        self.optim.step()
        return correct, loss, batch

    def train_an_epoch(self, header: str = 'train', verbose: int = 1):
        self._check_train()
        self.epoch_idx += 1
        avg_acc = AvgMeter()
        avg_loss = AvgMeter()
        self.check_df_exists(header, ['acc', 'loss'])
        
        self.model.train()
        for train_data, train_label in self.loaders[header]:
            correct, loss, batch = self.forwarding_and_updating(
                train_data, train_label)
            avg_acc(correct, batch)
            avg_loss(loss.item(), batch)
        self.dfs_append_row(
            header, acc=avg_acc.avg, loss=avg_loss.avg)

        if self.scheduler is not None:
            self.scheduler.step()

        if self.writer is not None:
            self.add_scalars(
                group_name=header,
                updating_dict={
                    'acc': avg_acc.avg,
                    'loss': avg_loss.avg},
                idx=self.epoch_idx)

        if verbose == 1:
            self.log_info(
                header=header, epoch=self.epoch_idx,
                acc=avg_acc.avg, loss=avg_loss.avg)
        return avg_acc.avg, avg_loss.avg

    def eval_an_epoch(self, header: str = 'test', verbose: int = 0):
        """For validation or testing, designed for none-gradient processing.
        """
        self._check_eval()
        assert isinstance(header, str)
        assert header in ['valid', 'test']
        avg_acc = AvgMeter()
        avg_loss = AvgMeter()
        self.check_df_exists(header, ['acc', 'loss'])
        
        self.model.eval()
        with torch.no_grad():
            for data, label in self.loaders[header]:
                correct, loss, batch = self.forwarding(data, label)
                avg_acc(correct, batch)
                avg_loss(loss.item(), batch)
        self.dfs_append_row(header, acc=avg_acc.avg, loss=avg_loss.avg)
        
        if self.writer is not None:
            self.add_scalars(
                group_name=header,
                updating_dict={
                    'acc': avg_acc.avg,
                    'loss': avg_loss.avg},
                idx=self.epoch_idx)

        if verbose == 1:
            self.log_info(
                header=header, epoch=self.epoch_idx,
                acc=avg_acc.avg, loss=avg_loss.avg)
        return avg_acc.avg, avg_loss.avg
    
    def warm_up_lr(self, terminal_lr: float, header: str = 'train', verbose: int = 1):
        """Fix as the wrapper of the train_an_epoch.
        Using warm up learning from:
            https://stackoverflow.com/questions/55933867/what-does-learning-rate-warm-up-mean
        """
        self._check_train()
        assert isinstance(terminal_lr, float)
        assert isinstance(header, str)
        avg_acc = AvgMeter()
        avg_loss = AvgMeter()
        # This is the special case of HEADER and NAME_LOADER, please not follow this.
        self.check_df_exists(header, ['acc', 'loss'])
        self.epoch_idx += 1
        
        self.model.train()
        for idx, (train_data, train_label) in enumerate(self.loaders[header]):
            warm_up_lr = terminal_lr*(idx/len(self.loaders[header]))
            self.optim.param_groups[0]['lr'] = warm_up_lr
            correct, loss, batch = self.forwarding_and_updating(
                train_data, train_label)
            avg_acc(correct, batch)
            avg_loss(loss.item(), batch)
        self.dfs_append_row(header, acc=avg_acc.avg, loss=avg_loss.avg)

        if self.scheduler is not None:
            self.scheduler.step()

        if verbose == 1:
            self.log_info(
               header=header, epoch=self.epoch_idx,
               acc=avg_acc.avg, loss=avg_loss.avg)
        return avg_acc.avg, avg_loss.avg

    @staticmethod
    def batch2dataset(batches: list) -> np.ndarray:
        """ Converting list of batches to dataset.
        """
        if hasattr(batches[0], 'cpu'):
            # Converting from cuda to cpu.
            batches = [batch.cpu() for batch in batches]
        np_batches = [batch.numpy() for batch in batches]
        # [[], [], ...] -> []
        np_dataset = np.concatenate(np_batches)
        return np_dataset

    def predicting_an_epoch(self, name_loader: str) -> np.ndarray:
        """Predicting or validation or testing without labels.
        Therefore, no loss and metrics.
        """
        self._check_eval()
        assert isinstance(name_loader, str)
        assert name_loader in ['valid', 'test']
        HEADER: str = 'predict'
        self.check_df_exists(HEADER, ['acc', 'loss'])
        self.model.eval()

        list_pred: list = []
        with torch.no_grad():
            for data in self.loaders[name_loader]:
                if type(data) == list:
                    # More than one data with label or mask?
                    # Should be torch.Tensor in case data only.
                    data = data[0]
                pred = self.predicting(data)
                list_pred.append(pred)
        dataset_pred = self.batch2dataset(list_pred)
        dataset_pred = np.argmax(dataset_pred, axis=1)
        return dataset_pred
    
    def confusion_mat(
            self, name_loader: str, order: int, save: bool):
        """ Confusion matrix.
        """
        pred = self.predicting_an_epoch(name_loader)
        true = self.get_label_dataset(name_loader, order)
        # TODO: adding seaborn saving confusion matrix.
        return confusion_matrix(true, pred)
    
    def get_label_dataset(
            self, name_loader: str, order: int = 1) -> np.ndarray:
        """Get labels from loaders given the name_loader and order within the output of loader. 
        """
        list_data: list = []
        for data in self.loaders[name_loader]:
            if type(data) == list:
                data = data[order]
            list_data.append(data)
        dataset_label = self.batch2dataset(list_data)
        return dataset_label

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
        
    def train_eval_epoches(
        self, epoch: int, 
        eval_every_epoch: int,
        name_loader: str,
        return_best: bool,
        verbose: int = 1) -> float:
        self._check_train()
        self._check_eval()
        assert isinstance(epoch, int)
        assert isinstance(eval_every_epoch, int)
        assert isinstance(name_loader, str)
        assert isinstance(return_best, bool)
        assert epoch >= eval_every_epoch
        
        HEADER: str = 'train'
        self.check_df_exists(HEADER, ['acc', 'loss'])
        self.check_df_exists(name_loader, ['acc', 'loss'])
        pbar = range(1, epoch + 1)
        if verbose == 2:
            pbar = tqdm(pbar)
        for i in pbar:
            train_acc, train_loss = self.train_an_epoch(verbose=verbose)
            if i % eval_every_epoch == 0 and i != 0:
                eval_acc, eval_loss = self.eval_an_epoch(
                    name_loader, verbose=verbose)
                if verbose == 2:
                    pbar.set_description(
                        f'train acc: {train_acc}, eval acc: {eval_acc}')
        if return_best:
            best_acc = self.get_best(name_loader, 'acc', 'max')
            return best_acc
        else:
            return eval_acc


class HyperTrainer(Trainer):
    """ To make this run-able with trainer. 
    The model and optim are required to be reconstructed every tunning round.
    Therefore instead of receive the instances, need to receive callable and input.
    To construct within this module instead.
    """
    def __init__(
            self,
            model_kwargs: dict, 
            optim_kwargs: dict,
            *args, **kwargs):
        super(HyperTrainer, self).__init__(*args, **kwargs)
        # Get the type of model and optim.
        self.model_type = type(kwargs['model'])
        self.model_kwargs = model_kwargs
        self.optim_type = type(kwargs['optim'])
        self.optim_kwargs = optim_kwargs
        
    def hyper_init(self):
        """Recreate the model and optim in every trial.
        """
        self.model = self.model_type(**self.model_kwargs)
        self.optim_kwargs['params'] = self.model.parameters()
        self.optim = self.optim_type(**self.optim_kwargs)
        self.model.to(self.device)
        
    def train_eval_epoches(
        self, epoch: int,
        eval_every_epoch: int,
        name_loader: str,
        trial_func: Callable, 
        trial,
        return_best: bool = True, 
        verbose: int = 1) -> float:
        """Train and eval designed for optuna hyper parameter seaching.
        """
        self._check_train()
        self._check_eval()
        assert isinstance(epoch, int)
        assert isinstance(eval_every_epoch, int)
        assert isinstance(name_loader, str)
        assert isinstance(return_best, bool)
        assert epoch >= eval_every_epoch
        
        HEADER: str = 'train'
        self.check_df_exists(HEADER, ['acc', 'loss'])
        self.check_df_exists(name_loader, ['acc', 'loss'])
        self.hyper_init()

        hyper_kwargs = trial_func(trial)
        # TODO: cover more than lr, weight decay and momentum.
        if 'lr' in hyper_kwargs:
            self.optim.param_groups[0]['lr'] = hyper_kwargs['lr']
        if 'weight_decay' in hyper_kwargs:
            self.optim.param_groups[0]['weight_decay'] = hyper_kwargs['weight_decay']
        if 'momentum' in hyper_kwargs:
            self.optim.param_groups[0]['momentum'] = hyper_kwargs['momentum']
        if len(hyper_kwargs) > 3:
            warnings.warn('kwargs is more than 3 might not supported.', UserWarning)

        pbar = range(1, epoch + 1)
        if verbose == 2:
            pbar = tqdm(pbar)
        for i in pbar:
            train_acc, train_loss = self.train_an_epoch(verbose=verbose)
            if i % eval_every_epoch == 0 and i != 0:
                eval_acc, eval_loss = self.eval_an_epoch(
                    name_loader, verbose=verbose)
                if verbose == 2:
                    pbar.set_description(
                        f'train acc: {train_acc}, eval acc: {eval_acc}')
                trial.report(eval_acc, i)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        if return_best:
            best_acc = self.get_best(name_loader, 'acc', 'max')
            return best_acc
        else:
            return eval_acc


class HalfTrainer(Trainer):
    """Trainer with the supporting of the mixed precision training.
    """
    def __init__(self, *args, **kwargs):
        super(HalfTrainer, self).__init__(*args, **kwargs)
        self.opt_level = None
        
    def _check_half(self):
        assert self.model is not None
        assert self.optim is not None
        assert self.loss_func is not None
        assert is_imported('apex.amp')
    
    def to_half(self, opt_level: str = 'O1', verbose: int = 0) -> None:
        """To half precision using Apex module.
        More details: https://nvidia.github.io/apex/amp.html
        TODO: checking with regularly trained model which one is faster.
        TODO: Checking why warnning the gradient overflow problem very 2 epoches.
        """
        assert isinstance(opt_level, str)
        assert isinstance(verbose, int)
        assert opt_level in ['O0', 'O1', 'O2', 'O3']
        self._check_half()
        self.model, self.optim = amp.initialize(
            self.model, self.optim, opt_level=opt_level, verbosity=verbose)

        # Checking that this is necessary or not.
        #if self.scheduler is not None:
        #     self.scheduler.optimizer = self.optim
        self.opt_level = opt_level
        

    def forwarding_and_updating(self, data, label) -> Tuple[str, str, str]:
        """
        """
        batch = label.data.size(0)
        data, label = data.to(self.device), label.to(self.device)
        self.optim.zero_grad()
        pred = self.model.forward(data)
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
        self.model, self.optim = amp.initialize(
            self.model, self.optim, opt_level=self.opt_level)
