import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.cuda import amp
from copy import deepcopy

from .base_trainer import BaseTrainer
from utils import (get_sr_metrics, sampler_set_epoch, log_config, de_parallel)


class SRTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.psnr = get_sr_metrics('psnr').to(self.device)
        self.ssim = get_sr_metrics('ssim').to(self.device)
        if config.metrics not in ['psnr', 'ssim']:
            raise ValueError(f'Unsupport metrics type: {config.metrics}')

    def train_one_epoch(self, config):
        self.model.train()

        sampler_set_epoch(config, self.train_loader, self.cur_epoch) 

        pbar = tqdm(self.train_loader) if self.main_rank else self.train_loader

        for cur_itrs, (images, labels) in enumerate(pbar):
            self.cur_itrs = cur_itrs
            self.train_itrs += 1

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward path
            with amp.autocast(enabled=config.amp_training):
                preds = self.model(images)
                loss = self.loss_fn(preds, labels)

            if config.use_tb and self.main_rank:
                self.writer.add_scalar('train/loss', loss.detach(), self.train_itrs)

            # Backward path
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            if self.cur_epoch >= config.ema_start_epoch:
                self.ema_model.update(self.model, self.train_itrs)
            else:
                self.ema_model.ema = deepcopy(de_parallel(self.model))

            if self.main_rank:
                pbar.set_description(('%s'*2) % 
                                (f'Epoch:{self.cur_epoch}/{config.total_epoch}{" "*4}|',
                                f'Loss:{loss.detach():4.4g}{" "*4}|',)
                                )

        return

    @torch.no_grad()
    def validate(self, config, val_best=False):
        pbar = tqdm(self.val_loader) if self.main_rank else self.val_loader
        for (images, labels) in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            preds = self.ema_model.ema(images).clamp(0.0, 1.0)
            self.psnr.update(preds.detach(), labels)
            self.ssim.update(preds.detach(), labels)

            if self.main_rank:
                pbar.set_description(('%s'*1) % (f'Validating:{" "*4}|',))

        psnr = self.psnr.compute()
        ssim = self.ssim.compute()
        if config.metrics == 'psnr':
            score = psnr
        elif config.metrics == 'ssim':
            score = ssim

        if self.main_rank:
            if val_best:
                self.logger.info(f'\n\nTrain {config.total_epoch} epochs finished.' + 
                                 f'\n\nBest {config.metrics.upper()} is: {psnr:.4f}\n')
            else:
                self.logger.info(f' Epoch{self.cur_epoch} PSNR: {psnr:.4f}  SSIM: {ssim:.4f}  | ' + 
                                 f'best {config.metrics.upper()} so far: {self.best_score:.4f}\n')

            if config.use_tb and self.cur_epoch < config.total_epoch:
                self.writer.add_scalar('val/PSNR', psnr.cpu(), self.cur_epoch+1)
                self.writer.add_scalar('val/SSIM', ssim.cpu(), self.cur_epoch+1)
        self.psnr.reset()
        self.ssim.reset()
        return score
        
    @torch.no_grad()
    def benchmark(self, config):
        if self.main_rank:
            log_config(config, self.logger)

        print(f'{"-"*25} Start benchmarking {"-"*25}\n')
        for i, val_loader in enumerate(self.val_loaders):
            print(f"\nStart validating dataset: {config.benchmark_datasets[i]}...")
            
            pbar = tqdm(val_loader) if self.main_rank else val_loader
            for (images, labels) in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                preds = self.model(images).clamp(0.0, 1.0)
                self.psnr.update(preds.detach(), labels)
                self.ssim.update(preds.detach(), labels)

                if self.main_rank:
                    pbar.set_description(('%s'*1) % (f'Validating:{" "*4}|',))

            psnr = self.psnr.compute()
            ssim = self.ssim.compute()

            if self.main_rank:
                self.logger.info(f' PSNR: {psnr:.4f}  SSIM: {ssim:.4f}\n')

            self.psnr.reset()
            self.ssim.reset()
        print(f'{"-"*25} Finish benchmarking {"-"*25}\n')

    @torch.no_grad()
    def predict(self, config):
        from datasets import SRBaseDataset
        if config.DDP:
            raise ValueError('Predict mode currently does not support DDP.')

        if config.test_bs != 1:
            self.logger.info('Warning: Predict mode only support batch size 1\n')

        self.logger.info('\nStart predicting...\n')

        if config.test_lr:
            test_score_path = f'{config.save_dir}/test_score_{config.model}_x{config.upscale}.txt'
            if os.path.isfile(test_score_path):
                os.remove(test_score_path)

        for (images, img_names) in tqdm(self.test_loader):
            hr = images[0]
            bicubic = images[1]
            img_name = img_names[0]

            hr = hr.to(self.device, dtype=torch.float32)
            if config.test_lr:
                lr = images[2].to(self.device, dtype=torch.float32)
                pred = self.model(lr).clamp(0.0, 1.0)

                # Compute PSNR and SSIM if test lr image
                self.psnr.update(pred.detach(), hr)
                self.ssim.update(pred.detach(), hr)

                psnr = self.psnr.compute()
                ssim = self.ssim.compute()

                self.psnr.reset()
                self.ssim.reset()

                with open(test_score_path, 'a+') as f:
                    f.write(f'{img_name}\tPSNR: {psnr:.2f}\tSSIM: {ssim:.4f}\n')
            else:
                pred = self.model(hr).clamp(0.0, 1.0)

            # BCHW --> HWC
            if config.train_y:
                # Need to concatenate [pred_y, bicubic_cb, bicubic_cr] to obtain predicted image
                ycbcr = images[-1].numpy().squeeze(0)
                pred = pred.mul(255.0).cpu().unsqueeze(-1).numpy().squeeze(0).squeeze(0)
                pred = np.concatenate([pred, ycbcr[...,1:]], axis=-1)
                pred = SRBaseDataset.ycbcr_to_rgb(pred)
            else:
                pred = pred.mul(255.0).cpu().numpy().squeeze(0).transpose([1, 2, 0])

            pred = np.clip(pred, 0., 255.).astype(np.uint8)

            # Saving results
            img_suffix = img_name.split('.')[-1]
            img_prefix = img_name[:-len(img_suffix)-1]
            pred_path = f'{config.save_dir}/{img_prefix}_{config.model}_x{config.upscale}.{img_suffix}'
            bicubic_path = f'{config.save_dir}/{img_prefix}_bicubic_x{config.upscale}.{img_suffix}'

            pred = Image.fromarray(pred.astype(np.uint8))
            pred.save(pred_path)

            bicubic = Image.fromarray(bicubic.numpy().squeeze().astype(np.uint8))
            bicubic.save(bicubic_path)
