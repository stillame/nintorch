#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import pytest
import numpy as np
import torch.nn as nn
import torch.optim as optim

from ninstd.path import del_dir_or_file
from nintorch.trainer import Trainer, HyperTrainer
from nintorch.dataset import load_dataset
from nintorch.model_zoo import LeNet5
from nintorch.hyper import start_tuning, default_trial, default_objective

THRESHOLD: float = 0.3


class TestTrainer(object):
    @staticmethod
    def get_kwargs():
        """ Get kwargs for trainer or to reset the model every test case.
        """
        model = LeNet5(in_chl=1)
        model.init_weight_bias()
        train_loader, test_loader = load_dataset(
            128, 128, 0, 8, 'mnist', './tmp')
        loss_func = nn.CrossEntropyLoss()
        optim_ = optim.AdamW(model.parameters(), lr=1e-3)
        model_kwargs = {'in_chl': 1}
        optim_kwargs = {'params': model.parameters(), 'lr': 1e-3}
        
        basic_kwargs = {
            'model': model,
            'optim': optim_, 
            'loss_func': loss_func, 
            'train_loader': train_loader,
            'valid_loader': None, 
            'test_loader': test_loader,
            'model_kwargs': model_kwargs,
            'optim_kwargs': optim_kwargs,
            }
        return basic_kwargs
        
    def test_train_an_epoch(self):
        kwargs = self.get_kwargs()
        trainer = Trainer(**kwargs)
        acc, _ = trainer.train_an_epoch(verbose=0)
        # An epoches, training acc should be: 0.6467833333333334d
        assert acc > THRESHOLD
        
    def test_eval_an_epoch(self):
        kwargs = self.get_kwargs()
        trainer = Trainer(**kwargs)
        acc, _ = trainer.eval_an_epoch('test', verbose=0)
        assert acc < THRESHOLD
        
        acc, _ = trainer.train_an_epoch(verbose=0)
        acc, _ = trainer.eval_an_epoch('test', verbose=0)
        assert acc > THRESHOLD
        
    def test_predicting_an_epoch(self):
        kwargs = self.get_kwargs()
        trainer = Trainer(**kwargs)
        pred = trainer.predicting_an_epoch('test')
        labels = trainer.get_label_dataset('test', 1)
        correct = np.equal(pred, labels).sum()
        acc = correct/len(pred)
        assert len(pred) == 10_000
        assert len(labels) == 10_000
        assert acc < THRESHOLD
        
    def test_warm_up_lr(self):
        kwargs = self.get_kwargs()
        trainer = Trainer(**kwargs)
        acc, _ = trainer.warm_up_lr(1e-3, verbose=0)
        assert acc > THRESHOLD
        
    def test_confusion_mat(self):
        kwargs = self.get_kwargs()
        trainer = Trainer(**kwargs)
        confusion_mat = trainer.confusion_mat('test', 1, False)
        assert confusion_mat.shape == (10, 10)
    
    def test_train_eval_epoches(self):
        kwargs = self.get_kwargs()
        trainer = Trainer(**kwargs)
        best_acc = trainer.train_eval_epoches(5, 5, 'test', True, 0)
        assert best_acc > THRESHOLD*3
        
    def test_hyper_train_eval_epoches(self):
        kwargs = self.get_kwargs()
        trainer = HyperTrainer(**kwargs)
        wrapped_obj = lambda trial: default_objective(
            3, trainer, default_trial, trial)
        study = start_tuning(wrapped_obj, 'maximize', 5)                
        best_acc = study.best_trial.value
        assert best_acc > THRESHOLD

del_dir_or_file('./tmp')
