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
from lightning.baseline import BaselineSystem
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
        f"{preprocess_config['meta']['train']}.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        f"{preprocess_config['meta']['val']}.txt", preprocess_config, train_config, sort=False, drop_last=False
    )

    # Prepare model
    model = FastSpeech2(preprocess_config, model_config)
    optimizer = get_optimizer(model, model_config, train_config)
    scheduler = get_scheduler(optimizer, train_config)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    system = BaselineSystem(
        model=model,
        optimizer=optimizer,
        loss_func=Loss,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        scheduler=scheduler,
        configs=configs,
        vocoder=vocoder,
    )

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    # train_logger = pl.loggers.TensorBoardLogger(train_config["path"]["log_path"], "baseline_train")
    # val_logger = pl.loggers.TensorBoardLogger(train_config["path"]["log_path"], "baseline_val")
    comet_logger = pl.loggers.CometLogger(
        save_dir=os.path.join(train_config["path"]["log_path"], "baseline"),
        experiment_key=args.exp_key,
        log_code=True,
        log_graph=True,
        parse_args=True,
        log_env_details=True,
        log_git_metadata=True,
        log_git_patch=True,
        log_env_gpu=True,
        log_env_cpu=True,
        log_env_host=True,
    )
    comet_logger.log_hyperparams({
        "preprocess_config": preprocess_config,
        "train_config": train_config,
        "model_config": model_config,
    })
    loggers = [comet_logger]
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
        monitor="val_Loss/total_loss",
        mode="min",
        save_top_k=-1,
        every_n_train_steps=save_step,
        save_last=True,
    )
    callbacks.append(checkpoint)
    

    # outer_bar = tqdm(total=total_step, desc="Training", position=0)
    # outer_bar.n = args.restore_step
    # outer_bar.update()
    outer_bar = GlobalProgressBar(process_position=0)
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
    if args.exp_key is not None:
        resume_ckpt = f'./output/ckpt/LibriTTS/meta-tts/{args.exp_key}/checkpoints/{args.ckpt_file}' #NOTE

    trainer = pl.Trainer(
        max_steps=total_step,
        weights_save_path=train_config["path"]["ckpt_path"],
        callbacks=callbacks,
        logger=loggers,
        gpus=gpus,
        auto_select_gpus=True,
        distributed_backend=distributed_backend,
        limit_train_batches=1.0,  # Useful for fast experiment
        # fast_dev_run=True, # Useful for debugging
        # overfit_batches=0.001, # Useful for debugging
        gradient_clip_val=grad_clip_thresh,
        accumulate_grad_batches=grad_acc_step,
        resume_from_checkpoint=resume_ckpt,
        deterministic=True,
        log_every_n_steps=log_step,
        # val_check_interval=val_step,
        profiler=profiler,
    )
    trainer.fit(system)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        help="path to preprocess.yaml",
        default='config/LibriTTS/preprocess.yaml',
    )
    parser.add_argument(
        "-m", "--model_config", type=str, help="path to model.yaml",
        default='config/LibriTTS/model.yaml',
    )
    parser.add_argument(
        "-t", "--train_config", type=str, help="path to train.yaml",
        default='config/LibriTTS/train.yaml',
    )
    parser.add_argument(
        "-s", "--meta_batch_size", type=int, help="meta batch size",
        default=torch.cuda.device_count(),
    )
    parser.add_argument(
        "-e", "--exp_key", type=str, help="experiment key",
        default=None,
    )
    parser.add_argument(
        "-c", "--ckpt_file", type=str, help="ckpt file name",
        default=None,
    )
    parser.add_argument(
        "-d", "--dev", action="store_true", help="dev mode"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    train_config["meta"]["meta_batch_size"] = args.meta_batch_size
    if args.dev:
        train_config["step"]["synth_step"] = 100
        train_config["step"]["val_step"] = 100
        train_config["step"]["save_step"] = 100
        model_config["transformer"]["encoder_layer"] = 2
        model_config["transformer"]["decoder_layer"] = 2
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
