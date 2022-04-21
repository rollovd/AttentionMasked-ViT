import numpy as np

import os, glob, time
from tqdm import tqdm

import wandb
import torch
import pandas as pd

from supplementary.metrics import calculate_metrics
from transformers import get_cosine_schedule_with_warmup
from supplementary.helpers.wandb_helper import _wandb_images

class Trainer:

    def __init__(self, optimizer, net, train_loader, test_loader, losses, mask,
                 epochs, only_validate, save_images, times_to_validate_per_epoch, first_epoch_to_validate,
                 images_frequency_save, num_warmup_epochs, device, number_of_images_to_save, pretrained_model, logging):

        self.net = net
        if pretrained_model is not None:
            self.pretrained_model_name = pretrained_model
            self.net.load_state_dict(torch.load(pretrained_model))
            print('Prepare pretrained model...')

        self.optimizer = optimizer
        self.criterion = losses['cross_entropy'][0]
        # self.criterion = losses['wmv']
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.only_validate = only_validate
        self.save_images = save_images
        self.images_frequency_save = images_frequency_save
        self.first_epoch_to_validate = first_epoch_to_validate
        self.num_warmup_epochs = num_warmup_epochs
        self.scheduler = self._initializer_scheduler()
        self.device = device
        self.number_of_images_to_save = number_of_images_to_save
        self.logging = logging
        self.mask = mask
        self.min_accuracy = 0

        self.indices_to_validate = torch.linspace(0, len(self.train_loader), times_to_validate_per_epoch,
                                                  dtype=torch.int16)[:-1]

    def _initializer_scheduler(self):
        self._num_warmup_steps = len(self.train_loader) * self.num_warmup_epochs
        self._num_total_steps = len(self.train_loader) * (self.epochs - self.num_warmup_epochs)
        scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=self._num_warmup_steps,
                                                    num_training_steps=self._num_total_steps)

        return scheduler

    def _save_checkpoint(self, epoch, val_accuracy):
        if self.mask:
            model_name = f'index_{epoch}_accuracy_{round(val_accuracy, 3)}_{self.net.__class__.__name__}_embed_dim_{self.net.embed_dim}_pos_embed_{self.net.positional_emb_name}_epochs_{self.epochs}_depth_{self.net.depth}_heads_{self.net.num_heads}_head_name_{self.net.head_name}_query_bias_{self.net.query_bias}_key_bias_{self.net.key_bias}_value_bias_{self.net.value_bias}_3d_repeated'
            save_directory = f'/home/yandex/vit-final/saved_models_new/{self.net.__class__.__name__}_embed_dim_{self.net.embed_dim}_pos_embed_{self.net.positional_emb_name}_epochs_{self.epochs}_depth_{self.net.depth}_heads_{self.net.num_heads}_head_name_{self.net.head_name}_query_bias_{self.net.query_bias}_key_bias_{self.net.key_bias}_value_bias_{self.net.value_bias}_3d_repeated'
        else:
            model_name = f'index_{epoch}_accuracy_{round(val_accuracy, 3)}_{self.net.__class__.__name__}_epochs_{self.epochs}'
            save_directory = f'/home/yandex/vit-final/saved_models_new/{self.net.__class__.__name__}_epochs_{self.epochs}'

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        files_in_folder = glob.glob(save_directory + '/*')
        for file in files_in_folder:
            file_accuracy = float(file.split('/')[-1].split('_')[3])
            if file_accuracy < val_accuracy:
                os.remove(file)

        path_save = os.path.join(save_directory, model_name)
        torch.save(self.net.state_dict(), path_save)
        print(f'Model saved successfully with accuracy score {val_accuracy}!!!')

        return val_accuracy

    def train_one_epoch(self, epoch):

        self.net.train()
        self.net.to(self.device)
        train_scores, train_targets = [], []

        with torch.enable_grad():
            for train_index, train_batch in tqdm(enumerate(self.train_loader)):

                if self.mask:
                    samples, masks, targets, proportion_classes, _ = train_batch
                    samples, masks, targets = samples.to(self.device), masks.to(self.device), \
                                              targets.to(self.device).type(torch.float32)
                else:
                    samples, targets = train_batch
                    samples, targets = samples.to(self.device), targets.to(self.device).type(torch.float32)

                self.optimizer.zero_grad()
                logits = self.net(x=samples, attn_mask=masks).flatten()\
                    if self.mask else self.net(x=samples).flatten()

                ## Adam
                if self.optimizer.__class__.__name__ not in ['SAM', 'LookSAM']:
                    loss = self.criterion(logits, targets)
                    loss.backward()
                    self.optimizer.step()

                # ---------------------------------------------------------------------------------------------
                ## LookSAM
                elif self.optimizer.__class__.__name__  == 'LookSAM':
                    logits = self.net(samples, masks).flatten()
                    loss = self.criterion(logits, targets)
                    loss.backward()
                    self.optimizer.step(t=train_index, samples=samples, masks=masks, targets=targets, zero_grad=True)

                # ---------------------------------------------------------------------------------------------
                ## SAM
                elif self.optimizer.__class__.__name__ == 'SAM':
                    def closure():
                        loss = self.criterion(self.net(samples, masks), targets)
                        loss.backward()
                        return loss

                    logits = self.net(samples, masks).flatten()
                    loss = self.criterion(logits, targets)
                    loss.backward()
                    self.optimizer.step(closure)
                    self.optimizer.zero_grad()

                if self.scheduler is not None:
                    if self.logging:
                        wandb.log({'lr': self.optimizer.param_groups[0]["lr"]})
                    self.scheduler.step()

                train_scores += logits.cpu().detach().numpy().tolist()
                train_targets += targets.cpu().detach().numpy().tolist()

                if self.logging:
                    wandb.log({'train/loss': loss.item(), 'train/step': epoch * len(self.train_loader) + train_index})

                if self.save_images and (not train_index % self.images_frequency_save) and self.logging:
                    _wandb_images(self.number_of_images_to_save, samples)

                if (epoch >= self.first_epoch_to_validate) and train_index in self.indices_to_validate:
                    predicts, true_labels = self.predict(epoch)
                    val_accuracy, val_far, val_frr, val_thresh = calculate_metrics(
                        torch.tensor(true_labels).type(torch.int32), predicts)

                    if self.logging:
                        wandb.log({'val/accuracy': val_accuracy, 'val/far': val_far,
                                   'val/frr': val_frr, 'val/step': epoch})

                    if val_accuracy > self.min_accuracy:
                        self.min_accuracy = self._save_checkpoint(epoch, val_accuracy)

        train_accuracy, train_far, train_frr, train_thresh = calculate_metrics(train_targets, train_scores)

        wandb.log({'train/accuracy': train_accuracy, 'train/far': train_far,
                   'train/frr': train_frr})
        print(f'accuracy: {train_accuracy}, far: {train_far}, frr: {train_frr}')

    def predict(self, epoch):
        self.net.eval()

        losses, validation_scores, validation_targets = [], [], []
        if self.only_validate:
            validation_paths = []
        data = pd.DataFrame()

        with torch.no_grad():
            for val_index, val_batch in tqdm(enumerate(self.test_loader)):

                if self.mask:
                    samples, masks, targets, proportion_classes, image_paths = val_batch
                    samples, masks, targets = samples.to(self.device), masks.to(self.device),\
                                              targets.to(self.device).type(torch.float32)
                else:
                    samples, targets = val_batch
                    samples, targets = samples.to(self.device), targets.to(self.device).type(torch.float32)

                logits = self.net(x=samples, attn_mask=masks).flatten() if self.mask else self.net(x=samples).flatten()

                validation_loss = self.criterion(logits, targets)
                validation_scores += logits.cpu().detach().numpy().tolist()
                validation_targets += targets.cpu().detach().numpy().tolist()
                if self.only_validate:
                    validation_paths += image_paths

                losses.append(validation_loss.item())

            if self.logging:
                wandb.log({
                    'val/loss': np.mean(losses),
                    'val/step': epoch
                })

        if self.only_validate:
            data['path'] = validation_paths
            data['confidence'] = validation_scores
            data['target'] = validation_targets

            data.to_csv(f'{self.pretrained_model_name}.csv', index=False)

        return validation_scores, validation_targets


    def train(self):

        if self.only_validate:
            predicts, true_labels = self.predict(epoch=0)
            val_accuracy, val_far, val_frr, val_thresh = calculate_metrics(torch.tensor(true_labels).type(torch.int32), predicts)
            print(f'accuracy: {val_accuracy}, far: {val_far}, frr: {val_frr}')

        else:
            if self.logging:
                def create_wandb_config():
                    if self.mask:
                        return {
                            'net_name': self.net.__class__.__name__,
                            'optimizer': self.optimizer,
                            'epochs': self.epochs,
                            'num_warmup_epochs': self.num_warmup_epochs,
                            'patch_size': self.net.patch_size,
                            'embed_dim': self.net.embed_dim,
                            'depth': self.net.depth,
                            'num_heads': self.net.num_heads,
                            'pos_embed_name': self.net.positional_emb_name,
                            'head_name': self.net.head_name,
                            'loss': self.criterion.__class__.__name__,
                            'query_bias': self.net.query_bias,
                            'key_bias': self.net.key_bias,
                            'value_bias': self.net.value_bias
                        }

                    else:
                        return {
                            'net_name': self.net.__class__.__name__,
                            'optimier': self.optimizer,
                            'epochs': self.epochs
                        }

                wandb.init(project='ViT-Masked-Final-Diploma-Balanced', config=create_wandb_config())
                wandb.watch(self.net)

            for epoch in tqdm(range(self.epochs)):
                self.train_one_epoch(epoch)

if __name__ == "__main__":
    from supplementary.nets.ViT import VisionTransformer
    model = VisionTransformer(
        img_size=256,
        patch_size=16,
        in_chans=3,
        num_classes=1,
        embed_dim=192,
        depth=6,
        num_heads=8,
        attn_mask=True,
        pos_embed='harmonic',
        head_name='vanilla',
        sigma=96)


