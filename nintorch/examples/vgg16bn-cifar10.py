#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example of nintorch implemented model.
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ninstd.namer import Namer
from ninstd.path import del_dir_or_file, not_dir_mkdir
from nintorch import Trainer, HalfTrainer, init_logger, seed_torch, torch_cpu_or_gpu
from nintorch.dataset import (CIFAR10_MEAN, CIFAR10_STD,
                              crop_filp_normalize_transforms, load_dataset)
from nintorch.model_zoo import VGG16BN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Nintorch implemented model')
    parser.add_argument(
        '--epoch', '-e', type=int, default=10 + 1)
    parser.add_argument(
        '--lr', type=float, default=1e-2)
    parser.add_argument(
        '--name_log', type=str, default='log.txt')
    parser.add_argument(
        '--name_tensorboard', type=str, default='tensorboard')
    parser.add_argument(
        '--train_batch', type=int, default=32)
    parser.add_argument(
        '--test_batch', type=int, default=32)
    parser.add_argument(
        '--weight_decay', type=float, default=1e-3)
    parser.add_argument(
        '--seed', '-s', type=int, default=0)
    parser.add_argument(
        '--half', type=int, default=0)
    args = parser.parse_args()
    
    logger.info(args)
    TRAIN_CSV_NAME: str = 'train.csv'
    TEST_CSV_NAME: str = 'test.csv'
    STEP_DOWN_EPOCHS: list = [3, 6]
    DATASET_LOC: str = './dataset'
    VERBOSE: int = 1

    seed_torch(args.seed)
    namer = Namer.from_args(args)
    save_loc = namer.gen_name()
    save_train_csv_path = os.path.join(save_loc, TRAIN_CSV_NAME)
    save_test_csv_path = os.path.join(save_loc, TEST_CSV_NAME)

    save_log_path = os.path.join(save_loc, args.name_log)
    save_writer_path = os.path.join(save_loc, args.name_tensorboard)

    init_logger(save_log_path, rm_exist=True)
    del_dir_or_file(args.name_tensorboard)
    not_dir_mkdir(save_loc)
    writer = SummaryWriter(log_dir=save_writer_path)

    device = torch_cpu_or_gpu()
    train_loader, test_loader = load_dataset(
        num_train_batch=args.train_batch,
        num_test_batch=args.test_batch,
        num_extra_batch=0, num_worker=8, 
        dataset='cifar10', roof=DATASET_LOC,
        transforms_list=crop_filp_normalize_transforms(
            CIFAR10_MEAN, CIFAR10_STD, 32, 4))

    model = VGG16BN(in_chl=3).to(device)
    model.init_weight_bias()
    optim = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optim, STEP_DOWN_EPOCHS, gamma=0.1)
    loss_fuct = nn.CrossEntropyLoss()
    
    if args.half:
        trainer = HalfTrainer(
            model, optim=optim, loss_func=loss_fuct,
            train_loader=train_loader, test_loader=test_loader,
            scheduler=scheduler, writer=writer)
        trainer.to_half(opt_level='O2')
    else:
        trainer = Trainer(
        model, optim=optim, loss_func=loss_fuct,
        train_loader=train_loader, test_loader=test_loader,
        scheduler=scheduler, writer=writer)
    
    trainer.warm_up_lr(args.lr, verbose=VERBOSE)
    for i in range(1, args.epoch):
        trainer.train_an_epoch(verbose=VERBOSE)
        trainer.eval_an_epoch('test', verbose=VERBOSE)3c

    trainer.dfs['train'].to_csv(save_train_csv_path)
    trainer.dfs['test'].to_csv(save_test_csv_path)
    del_dir_or_file(DATASET_LOC)