# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collection of wrapper for training and evaluting pytorch model.
"""
import warnings
import pandas as pd
from loguru import logger
from apex import amp
import optuna
import torch
import torch.optim as optim
from ninstd.check import is_imported
from .utils import AvgMeter, torch_cpu_or_gpu


__all__ = [
    'Trainer',
    'HalfTrainer',
    'HyperTrainer']


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
        self._epoch_idx = 0
        # self.record = self.gen_recorder()
        self.record = None

    @staticmethod
    def gen_record() -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            {'train_acc': [], 'test_acc': [], 'valid_acc': [],
             'train_loss': [], 'test_loss': [], 'valid_loss': []})

    def _check_train(self):
        assert self.train_loader is not None
        assert self.model is not None
        assert self.optim is not None
        assert self.loss_func is not None

    def _check_valid(self):
        assert self.valid_loader is not None
        assert self.model is not None
        assert self.loss_func is not None

    def _check_test(self):
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
            
    def log_info_any(self, header: str, *args, **kwargs):
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
        self._epoch_idx += 1
        avg_acc = AvgMeter()
        avg_loss = AvgMeter()
        header = 'Training'

        self.model.train()
        for train_data, train_label in self.train_loader:
            correct, loss, batch = self.forwarding_and_updating(
                train_data, train_label)
            avg_acc(correct, batch)
            avg_loss(loss.item(), batch)

        self.record['train_acc'][self._epoch_idx] = avg_acc.avg
        self.record['train_loss'][self._epoch_idx] = avg_loss.avg

        if self.scheduler is not None:
            self.scheduler.step()

        if self.writer is not None:
            self.add_scalars(
                group_name=header,
                updating_dict={
                    'train_acc': avg_acc.avg,
                    'train_loss': avg_loss.avg},
                idx=self._epoch_idx)

        if verbose > 0:
            self.log_info(
                header=header, epoch=self._epoch_idx,
                acc=avg_acc.avg, loss=avg_loss.avg)

        return avg_acc.avg, avg_loss.avg

    def test_an_epoch(self, verbose: int=0):
        self._check_test()
        avg_acc = AvgMeter()
        avg_loss = AvgMeter()
        header = 'Testing'

        self.model.eval()
        with torch.no_grad():
            for test_data, test_label in self.test_loader:
                correct, loss, batch = self.forwarding(
                    test_data, test_label)
                avg_acc(correct, batch)
                avg_loss(loss.item(), batch)

        self.record['test_acc'][self._epoch_idx] = avg_acc.avg
        self.record['test_loss'][self._epoch_idx] = avg_loss.avg

        if self.writer is not None:
            self.add_scalars(
                group_name=header,
                updating_dict={
                    'test_acc': avg_acc.avg,
                    'test_loss': avg_loss.avg},
                idx=self._epoch_idx)

        if verbose > 0:
            self.log_info(
                header=header, epoch=self._epoch_idx,
                acc=avg_acc.avg, loss=avg_loss.avg)

        return avg_acc.avg, avg_loss.avg

    def valid_an_epoch(self, verbose: int=0):
        self._check_valid()
        avg_acc = AvgMeter()
        avg_loss = AvgMeter()
        header = 'Validation'

        self.model.eval()
        with torch.no_grad():
            for valid_data, valid_label in self.valid_loader:
                correct, loss, batch = self.forwarding(
                    valid_data, valid_label)
                avg_acc(correct, batch)
                avg_loss(loss.item(), batch)

        self.record['valid_acc'][self._epoch_idx] = avg_acc.avg
        self.record['valid_loss'][self._epoch_idx] = avg_loss.avg

        if self.writer is not None:
            self.add_scalars(
                group_name=header,
                updating_dict={'valid_acc': avg_acc.avg, 'valid_loss': avg_loss.avg},
                idx=self._epoch_idx)

        if verbose > 0:
            self.log_info(
                header='Validation', epoch=self._epoch_idx,
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
        header = 'Warmup'

        self.model.train()
        for idx, (train_data, train_label) in enumerate(self.train_loader):
            warm_up_lr = terminal_lr*(idx/len(self.train_loader))
            self.optim.param_groups[0]['lr'] = warm_up_lr
            correct, loss, batch = self.forwarding_and_updating(
                train_data, train_label)
            avg_acc(correct, batch)
            avg_loss(loss.item(), batch)

        self.record['train_acc'][self._epoch_idx] = avg_acc.avg
        self.record['train_loss'][self._epoch_idx] = avg_loss.avg

        if self.scheduler is not None:
            self.scheduler.step()

        if verbose > 0:
            self.log_info(
                header=header, epoch=self._epoch_idx,
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



class HyperTrainer(Trainer):
    def __init__(self, valid_or_test: bool = False, *args, **kwargs):
        super(HyperTrainer, self).__init__(*args, **kwargs)
        self.valid_or_test = valid_or_test

    def train_eval_epoches(
        self, epoch: int, eval_every_epoch: int, trial_funct, trial):
        """Train and test for certain epoches with an option
        to turn on the hyper parameter tunning.
        For trial_funct, please follow nintorch.hyper.default_trial.
        """
        self._check_train()
        if self.valid_or_test:
            self._check_valid()
        else:
            self._check_test()
        kwargs = trial_funct(trial)

        # TODO: cover more than lr and weight decay.
        if 'lr' in kwargs:
            self.optim.param_groups[0]['lr'] = kwargs['lr']
        if 'weight_decay' in kwargs:
            self.optim.param_groups[0]['weight_decay'] = kwargs['weight_decay']
        if len(kwargs) > 2:
            warnings.warn('kwargs is more than 2 might not supported.', UserWarning)

        for i in range(epoch):
            train_acc, train_loss = self.train_an_epoch()
            if self.valid_or_test:
                valid_acc, valid_loss = self.valid_an_epoch()
            else:
                test_acc, test_loss = self.test_an_epoch()
            self.record['train_acc'][i] = train_acc
            self.record['train_loss'][i] = train_loss

            if i % eval_every_epoch == 0 and i != 0:
                if self.valid_or_test:
                    self.record['valid_acc'][i] = valid_acc
                    self.record['valid_loss'][i] = valid_loss
                    trial.report(valid_acc, i)
                else:
                    self.record['test_acc'][i] = test_acc
                    self.record['test_loss'][i] = test_loss
                    trial.report(test_acc, i)

            if trial.should_prune():
                raise optuna.exception.TrialPruned()


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
    
    def forwarding_and_updating(self, data, label):
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
                raise RuntimeError('')
            else:
                lr = self.optim.param_groups[0]['lr']
        if not self._half_flag:
            self.optim = optim.SGD(self.model.parameters(), lr)
        else:
            self.optim = optim.SGD(self.model.parameters(), lr)
            self.model, self.optim = amp.initialize(
                self.model, self.optim, opt_level=self.opt_level)

