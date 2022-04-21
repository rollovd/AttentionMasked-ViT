import os
import hydra
import wandb
import torch
import random
import inspect
import numpy as np
from colorama import Fore, Style
from supplementary.nets.ViT import VisionTransformer
from omegaconf import OmegaConf, DictConfig

from supplementary.seed import seed_everything
from supplementary.creator import create_augments, create_datasets, create_dataloaders, create_net, create_loss, \
            create_optimizer, create_trainer, create_mobilenet, create_swin

SEED = 42
seed_everything(SEED)

@hydra.main(config_path="./configs", config_name="config.yml")
def app(cfg: DictConfig):
    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda:0' if cuda else 'cpu')

    print(Fore.CYAN)

    print(f'--> Initializing augmentations...')
    train_augment, test_augment = create_augments(cfg.transform)
    print(f'--> Train augmentations:')
    print(Fore.LIGHTGREEN_EX, end='')
    for tr in train_augment:
        print(f'        {tr}')

    print(Fore.CYAN, end='')
    print(f'--> Test augmentations:')
    print(Fore.LIGHTGREEN_EX, end='')
    for tr in test_augment:
        print(f'        {tr}')
    print(Fore.CYAN, end='')

    print(f'--> Initializing datasets... ')
    train_dataset, test_dataset = create_datasets(config=cfg.data,
                                                  train_augment=train_augment, test_augment=test_augment)
    print(f'--> Train dataset:')
    print(Fore.LIGHTGREEN_EX, end='')
    print(f'        {train_dataset}')
    print(Fore.CYAN, end='')
    print(f'--> Test dataset:')
    print(Fore.LIGHTGREEN_EX, end='')
    print(f'        {test_dataset}')
    print(Fore.CYAN, end='')

    print(f'--> Initializing dataloaders... ')
    train_loader, test_loader = create_dataloaders(cfg.dataloaders, cfg.sampler, train_dataset, test_dataset)
    print(f'--> Train dataloader:')
    print(Fore.LIGHTGREEN_EX, end='')
    print(f'        {type(train_loader).__name__}({cfg.dataloaders.train, cfg.sampler})')
    print(Fore.CYAN, end='')
    print(f'--> Test dataloader:')
    print(Fore.LIGHTGREEN_EX, end='')
    print(f'        {type(test_loader).__name__}({cfg.dataloaders.test})')
    print(Fore.CYAN, end='')

    print(f'--> Initializing network... ')
    net = create_net(cfg.net, device=device)
    net.to(device)
    print(Fore.LIGHTGREEN_EX, end='')
    print(f'        {type(net).__name__}({cfg.net.config})')
    print(Fore.CYAN, end='')

    # print(f'--> Initializing network... ')
    # net = create_swin(cfg.net)
    # net.to(device)
    # print(Fore.LIGHTGREEN_EX, end='')
    # print(f'        {type(net).__name__}({cfg.net.config})')
    # print(Fore.CYAN, end='')

    print(f'--> Initializing losses... ')
    losses = create_loss(cfg.loss)
    print(Fore.LIGHTGREEN_EX, end='')
    for name, loss in losses.items():
        print(f'    {name}: {type(loss).__name__}')
    print(Fore.CYAN, end='')

    print(f'--> Initializing optimizer... ')
    optimizer = create_optimizer(cfg.optimizer, net)
    print(Fore.LIGHTGREEN_EX, end='')
    print(f'        {type(optimizer).__name__}({cfg.optimizer.config})')
    print(Fore.CYAN, end='')

    # print(f'--> Initializing optimizer... ')
    # from supplementary.optimizers.looksam import LookSAM
    #
    # optimizer = LookSAM(alpha=0.5, k=5, model=net,
    #                     base_optimizer=torch.optim.Adam, criterion=losses['cross_entropy'][0], rho=0.05, lr=1e-3,
    #                     weight_decay=1e-4)
    #
    # # optimizer = create_optimizer(cfg.optimizer, net)
    # print(Fore.LIGHTGREEN_EX, end='')
    # # print(f'        {type(optimizer).__name__}({cfg.optimizer.config})')
    # print(Fore.CYAN, end='')

    # print(f'--> Initializing optimizer... ')
    # from supplementary.optimizers.sam import SAM
    #
    # optimizer = SAM(params=net.parameters(), base_optimizer=torch.optim.Adam, lr=1e-3, weight_decay=1e-4)
    # print(Fore.LIGHTGREEN_EX, end='')
    # print(Fore.CYAN, end='')



    print(f'--> Initializing trainer... ')
    trainer = create_trainer(cfg.trainer, optimizer=optimizer, net=net, losses=losses,
                             train_loader=train_loader, test_loader=test_loader, device=device, mask=True)
    print(Fore.LIGHTGREEN_EX, end='')
    for name, value in trainer.__dict__.items():
        if name != 'net':
            print(f'{name}: {value}')
    print(Fore.CYAN, end='')

    print(f'--> Training... ')
    trainer.train()

    print(Style.RESET_ALL)


if __name__ == "__main__":
    app()