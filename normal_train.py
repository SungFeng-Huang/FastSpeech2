import argparse
import os

import comet_ml
import pytorch_lightning as pl
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pytorch_lightning.callbacks.progress import ProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor, GPUStatsMonitor, ModelCheckpoint
from pytorch_lightning.profiler import AdvancedProfiler

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss, FastSpeech2
from dataset import Dataset
from lightning.system import System
from lightning.scheduler import get_scheduler
from lightning.optimizer import get_optimizer
from lightning.callbacks import GlobalProgressBar
from lightning.collate import get_single_collate

from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    train_dataset = Dataset(
        "train-clean-100-train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "train-clean-100-val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    group_size = 1

    # Prepare model
    model = FastSpeech2(preprocess_config, model_config)
    optimizer = get_optimizer(model, model_config, train_config)
    scheduler = get_scheduler(optimizer, train_config)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    system = System(
        model=model,
        optimizer=optimizer,
        loss_func=Loss,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        scheduler=scheduler,
        configs=configs,
        vocoder=vocoder,
        group_dataloader=(group_size>1),
    )

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_logger = pl.loggers.TensorBoardLogger(train_config["path"]["log_path"], "train")
    val_logger = pl.loggers.TensorBoardLogger(train_config["path"]["log_path"], "val")
    loggers = [train_logger, val_logger]
    profiler = AdvancedProfiler(train_config["path"]["log_path"], 'profile.log')

    # Training
    # step = args.restore_step + 1
    # epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    # synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    callbacks = []
    checkpoint = ModelCheckpoint(
        monitor="train_loss",
        mode="min",
        save_top_k=-1,
        every_n_train_steps=save_step,
        save_last=True,
    )
    callbacks.append(checkpoint)
    

    # outer_bar = tqdm(total=total_step, desc="Training", position=0)
    # outer_bar.n = args.restore_step
    # outer_bar.update()
    outer_bar = GlobalProgressBar()
    inner_bar = ProgressBar(process_position=1)
    lr_monitor = LearningRateMonitor()
    gpu_monitor = GPUStatsMonitor(memory_utilization=True, gpu_utilization=True, intra_step_time=True, inter_step_time=True)
    callbacks.append(outer_bar)
    callbacks.append(inner_bar)
    callbacks.append(lr_monitor)
    callbacks.append(gpu_monitor)

    gpus = -1 if torch.cuda.is_available() else None
    distributed_backend = "ddp" if torch.cuda.is_available() else None
    resume_ckpt = None #NOTE

    trainer = pl.Trainer(
        max_steps=total_step,
        weights_save_path=train_config["path"]["ckpt_path"],
        callbacks=callbacks,
        logger=loggers,
        gpus=gpus,
        distributed_backend=distributed_backend,
        limit_train_batches=1.0,  # Useful for fast experiment
        # fast_dev_run=True, # Useful for debugging
        # overfit_batches=0.001, # Useful for debugging
        gradient_clip_val=grad_clip_thresh,
        accumulate_grad_batches=grad_acc_step,
        resume_from_checkpoint=resume_ckpt,
        deterministic=True,
        log_every_n_steps=log_step,
        val_check_interval=val_step,
        profiler=profiler,
        reload_dataloaders_every_epoch=True,
        replace_sampler_ddp=(group_size==1),
    )
    trainer.fit(system)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
