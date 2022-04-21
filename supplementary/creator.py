import inspect
import importlib
from functools import partial
from typing import Dict, Tuple
from collections import OrderedDict
from omegaconf import OmegaConf, DictConfig, ListConfig

import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

PURPOSE_TRAIN = 'train'
PURPOSE_TEST = 'test'
TRANSFORM_PURPOSES = [PURPOSE_TRAIN, PURPOSE_TEST]
DATASET_PURPOSES = [PURPOSE_TRAIN, PURPOSE_TEST]

def get_attr_from_module(module, attr):
    try:
        mdl = importlib.import_module(module)
    except:
        raise RuntimeError(f'failed to import module {module}')

    try:
        result = getattr(mdl, attr)
    except:
        raise RuntimeError(f'attribute {attr} not found in {module}')

    return result


def create_universal(config: DictConfig):
    par_type = get_attr_from_module(config.module, config.type)
    if inspect.isclass(par_type):
        par = par_type(**config.config)
    else:
        par = partial(par_type, **config.config) if 'config' in config.keys() else par_type

    return par


def create_single_augment(config: DictConfig):
    transforms = []
    for one in config:
        transforms.append(create_universal(one))

    return A.Compose(transforms + [ToTensorV2()])

def create_augments(config: DictConfig) -> Tuple[callable, callable]:
    transforms = {k: [] for k in TRANSFORM_PURPOSES}

    for key, cfg in config.items():
        if key not in TRANSFORM_PURPOSES:
            raise RuntimeError(f'purpose of transform must be one of {TRANSFORM_PURPOSES}')

        transforms[key] = create_single_augment(cfg.augment)

    return transforms['train'], transforms['test']

def create_single_dataset(config: DictConfig, augment):
    ds_type = get_attr_from_module(config.module, config.type)
    ds = ds_type(config.config, augment=augment)
    return ds


def create_datasets(config: DictConfig, train_augment, test_augment):
    test_dataset = None
    train_dataset = None

    for purpose, cfg in config.items():
        if purpose not in DATASET_PURPOSES:
            raise RuntimeError(f'purpose must be one of {DATASET_PURPOSES}')

        if purpose == PURPOSE_TEST:
            test_dataset = create_single_dataset(cfg, augment=test_augment)

        elif purpose == PURPOSE_TRAIN:
            train_dataset = create_single_dataset(cfg, augment=train_augment)

    return train_dataset, test_dataset

def configure_sampler(config_sampler, train_dataset):
    sampler_var_type = config_sampler.config['balancing_var']
    if sampler_var_type == 'aspect_ratio':
        return train_dataset._proportion_classes

    elif sampler_var_type == 'classes':
        return train_dataset._labels

def create_dataloaders(config_loader: DictConfig, config_sampler: DictConfig, train_dataset, test_dataset):

    # sampler = get_attr_from_module(config_sampler.module, config_sampler.type)
    # sampler = sampler(labels=torch.tensor(train_dataset._proportion_classes, dtype=torch.int64))

    return DataLoader(train_dataset, **config_loader.train),\
           DataLoader(test_dataset, **config_loader.test)

def create_net(config: DictConfig, device):
    net_type = get_attr_from_module(config.module, config.type)
    net = net_type(device=device, **config.config.backbone)
    return net

def create_mobilenet(config: DictConfig):
    net_type = get_attr_from_module(config.module, config.type)
    net = net_type(**config.config.backbone)
    return net

def create_swin(config: DictConfig):
    net_type = get_attr_from_module(config.module, config.type)
    net = net_type(**config.config.backbone)
    return net

def create_loss(config: DictConfig):

    losses = {}

    for cfg in config:
        name, one = list(cfg.items())[0]
        loss_type = get_attr_from_module(one.module, one.type)
        losses[name] = (loss_type(**one.config), one.weight)
        # losses[name] = (loss_type(**one.config))

    return losses

def create_optimizer(config: DictConfig, net: nn.Module):
    opt_type = get_attr_from_module(config.module, config.type)
    opt = opt_type(params=net.parameters(), **config.config)
    return opt

def create_scheduler(config: DictConfig, optimizer: torch.optim.Optimizer):
    sch_type = get_attr_from_module(config.module, config.type)
    sch = sch_type(optimizer=optimizer, **config.config)
    return sch

def create_trainer(config: DictConfig, net: nn.Module, optimizer: torch.optim.Optimizer, train_loader,
                   test_loader, losses, device, mask):

    trainer_type = get_attr_from_module(config.module, config.type)
    trn = trainer_type(optimizer=optimizer, net=net, train_loader=train_loader, mask=mask,
                       test_loader=test_loader, losses=losses, device=device, **config.config)
    return trn








