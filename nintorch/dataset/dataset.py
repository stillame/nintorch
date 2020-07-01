#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# TODO: update with different dataset and other.
# TODO: do with albumentation library.
# TODO: update with DataSet for loading this one. Using with base and pandas?
# TODO: DataSet display the image and denormalize images. 
# TODO: DataSet 
"""
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import albumentations as albu
import torchvision
import torchvision.transforms as transforms
from path_tools import if_notdir_mkdir

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]
CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

normalize_transforms = lambda mean, std: [
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)]

crop_filp_normalize_transforms = lambda mean, std, size, pad: [
    transforms.RandomCrop(size, padding=pad),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)]

crop_filp_transforms = lambda size, pad: [
    transforms.RandomCrop(size, padding=pad),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()]


def load_dataset(
        num_train_batch: int, num_test_batch: int, num_extra_batch: int = 0,
        num_worker: int = 8, dataset: str = 'mnist', 
        roof: str = './dataset', transforms_list : list = None):
    """Using torchvision to load the provided dataset online.
    Can using with predefinded transform function with the predefind mean and std. 
    Using transform_list=normalize_transforms(CIFAR10_MEAN, CIFAR10_STD)
    """
    assert isinstance(num_train_batch, int)
    assert isinstance(num_test_batch, int)
    assert isinstance(num_extra_batch, int)
    assert isinstance(num_worker, int)
    assert isinstance(dataset, str)
    assert isinstance(roof, str)
    
    dataset = dataset.lower()
    if_notdir_mkdir(roof)
    if transforms_list is None:    
        transforms_list = [transforms.ToTensor()]
    transforms_list = transforms.Compose(transforms_list)

    if dataset == 'mnist':
        train_set = torchvision.datasets.MNIST(
            root=roof, train=True, download=True, transform=transforms_list)
        train_loader = DataLoader(
            train_set, batch_size=num_train_batch, shuffle=True, num_workers=num_worker)
        test_set = torchvision.datasets.MNIST(
            root=roof, train=False, download=True, transform=transforms_list)
        test_loader = DataLoader(
            test_set, batch_size=num_test_batch, shuffle=False, num_workers=num_worker)

    elif dataset == 'fmnist':
        train_set = torchvision.datasets.FashionMNIST(
            root=roof, train=True, download=True, transform=transforms_list)
        train_loader = DataLoader(
            train_set, batch_size=num_train_batch, shuffle=True, num_workers=num_worker)
        test_set = torchvision.datasets.FashionMNIST(
            root=roof, train=False, download=True, transform=transforms_list)
        test_loader = DataLoader(
            test_set, batch_size=num_test_batch, shuffle=False, num_workers=num_worker)

    elif dataset == 'kmnist':
        train_set = torchvision.datasets.KMNIST(
            root=roof, train=True, download=True, transform=transforms_list)
        train_loader = DataLoader(
            train_set, batch_size=num_train_batch, shuffle=True, num_workers=num_worker)
        test_set = torchvision.datasets.FashionMNIST(
            root=roof, train=False, download=True, transform=transforms_list)
        test_loader = DataLoader(
            test_set, batch_size=num_test_batch, shuffle=False, num_workers=num_worker)

    elif dataset == 'emnist':
        train_set = torchvision.datasets.EMNIST(
            root=roof, train=True, download=True, transform=transforms_list)
        train_loader = DataLoader(
            train_set, batch_size=num_train_batch, shuffle=True, num_workers=num_worker)
        test_set = torchvision.datasets.FashionMNIST(
            root=roof, train=False, download=True, transform=transforms_list)
        test_loader = DataLoader(
            test_set, batch_size=num_test_batch, shuffle=False, num_workers=num_worker)

    elif dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(
            root=roof, train=True, download=True, transform=transforms_list)
        train_loader = DataLoader(
            train_set, batch_size=num_train_batch, shuffle=True, num_workers=num_worker)
        test_set = torchvision.datasets.CIFAR10(
            root=roof, train=False, download=True, transform=transforms_list)
        test_loader = DataLoader(
            test_set, batch_size=num_test_batch, shuffle=False, num_workers=num_worker)

    elif dataset == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(
            root=roof, train=True, download=True, transform=transforms_list)
        train_loader = DataLoader(
            train_set, batch_size=num_train_batch, shuffle=True, num_workers=num_worker)
        test_set = torchvision.datasets.CIFAR100(
            root=roof, train=False, download=True, transform=transforms_list)
        test_loader = DataLoader(
            test_set, batch_size=num_test_batch, shuffle=False, num_workers=num_worker)

    elif dataset == 'svhn':
        # The extra-section or extra_set is exist in this dataset.
        train_set = torchvision.datasets.SVHN(
            root=roof, split='train', download=True, transform=transforms_list)
        train_loader = DataLoader(
            train_set, batch_size=num_train_batch, shuffle=True, num_workers=num_worker)
        test_set = torchvision.datasets.SVHN(
            root=roof, split='test', download=True, transform=transforms_list)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=num_test_batch, shuffle=False, num_workers=num_worker)
        extra_set = torchvision.datasets.SVHN(
            root=roof, split='extra', download=True, transform=transforms_list)
        extra_loader = DataLoader(
            extra_set, batch_size=num_extra_batch, shuffle=False, num_workers=num_worker)
        return train_loader, test_loader, extra_loader
    else:
        raise NotImplementedError(
            'dataset must be in [mnist, fmnist, kmnist, '
            f'emnist, cifar10, cifar100, svhn] only, your input: {dataset}')
    return train_loader, test_loader


class BaseDataset(Dataset):
    def __init__(self, roof: str, transforms: list, *args, **kwargs):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self):
        pass
    
    def get_mean(self):
        pass
    
    def get_std(self):
        pass
    
    def profile_load_time(self):
        pass
    
    def denormalize(self):
        pass
    
    def plot_examples(self):
        pass
    
    


class CsvDataset(BaseDataset):
    def __init__(self, csv_file: str, *args, **kwargs):
        pass
    
    
