#! /usr/bin/env python
import argparse
import os
import sys
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm
from shutil import copyfile
import yaml

from model.modules import MLP, CGBlock, MCGBlock, HistoryEncoder
from model.multipathpp import MultiPathPP, MultiPathPPPredictor
from model.data import get_dataloader, dict_to_cuda, normalize
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import subprocess
import glob
import random

parser = argparse.ArgumentParser(description='train intention network.')

parser.add_argument('-cfg',
                    '--config',
                    type=str,
                    default='',
                    required=True,
                    help='config file')

parser.add_argument('-ckpt',
                    '--checkpoint',
                    type=str,
                    default=None,
                    help='checkpoint file')

parser.add_argument('-g',
                    '--gpu',
                    type=int,
                    default=[],
                    nargs='+',
                    help='gpu list to train')

parser.add_argument('-l',
                    '--log_dir',
                    type=str,
                    default='./log',
                    help='log directory path')

if __name__ == '__main__':
    torch.manual_seed(1988)

    args = parser.parse_args()

    config_file = args.config
    config_name, _ = os.path.splitext(os.path.basename(config_file))
    print(f'Using config: {config_name}')
    with open(config_file, 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    checkpoint_file = args.checkpoint
    if checkpoint_file is not None:
        print(f'Using checkpoint: {checkpoint_file}')

    gpu_list = args.gpu

    # create logger
    logger = TensorBoardLogger(args.log_dir, name=config_name)

    predictor = MultiPathPPPredictor(config = config)

    # set checkpoint callback to save best val_error_rate and last epoch
    # checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
    #     monitor='val_error_rate', mode='min', save_last=True)

    save_every_ckpt = False
    if save_every_ckpt:
        checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(filename='{epoch}-{val_loss:.2f}', save_last=True, save_top_k=-1)
    else:
        checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(filename='{epoch}-{val_loss:.2f}', save_last=True)

    callbacks = [checkpoint_callback, EarlyStopping(monitor="val_loss", mode="min")]

    if len(gpu_list) == 0:
        print('Gpu not specified, exit normally')
        exit(0)

    trainer = pl.Trainer(
        callbacks=callbacks,
        gpus=gpu_list,
        max_epochs=config["train"]["n_epochs"],
        val_check_interval=1.0,  # How often to check the validation set.
        limit_val_batches=1.0,
        logger=logger,
        log_every_n_steps=50,  # H ow often to log within steps. Default: 50.
        flush_logs_every_n_steps=100,  # How often to flush logs to disk (defaults to every 100 steps).
        resume_from_checkpoint=None,
        sync_batchnorm=True,
        accelerator='ddp',
        deterministic=True,
    )

    trainer.fit(predictor)

    # cp config file to exp folder
    if os.path.exists(logger.log_dir):
        copyfile(config_file, os.path.join(logger.log_dir, os.path.basename(config_file)))
