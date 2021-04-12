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
from lightning.anil import ANILSystem
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
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    # group_size = 4  # Set this larger than 1 to enable sorting in Dataset

    # Prepare model
    model = FastSpeech2(preprocess_config, model_config)
    optimizer = get_optimizer(model, model_config, train_config)
    scheduler = get_scheduler(optimizer, train_config)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    system = ANILSystem(
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
    # train_logger = pl.loggers.TensorBoardLogger(train_config["path"]["log_path"], "meta_train")
    # val_logger = pl.loggers.TensorBoardLogger(train_config["path"]["log_path"], "meta_val")
    comet_kwargs = {
        "experiment_key": None,
        "log_code": True,
        "log_graph": True,
        "parse_args": True,
        "log_env_details": True,
        "log_git_metadata": True,
        "log_git_patch": True,
        "log_env_gpu": True,
        "log_env_cpu": True,
        "log_env_host": True,
    }
    train_logger = pl.loggers.CometLogger(
        save_dir=os.path.join(train_config["path"]["log_path"], "meta"),
        **comet_kwargs,
    )
    # val_logger = pl.loggers.CometLogger(
        # save_dir=os.path.join(train_config["path"]["log_path"], "meta_val"),
        # **comet_kwargs,
    # )
    loggers = [train_logger]
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
    meta_batch_size = train_config["meta"]["meta_batch_size"]

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
        # val_check_interval=val_step,
        profiler=profiler,
    )
    trainer.fit(system)

    exit()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                # Forward
                output = model(*(batch[2:]))

                # Cal Loss
                losses = Loss(batch, output)
                total_loss = losses[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                        *losses
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


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
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
