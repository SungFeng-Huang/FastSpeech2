# Reference:
# https://github.com/asteroid-team/asteroid/blob/master/asteroid/engine/system.py

import pytorch_lightning as pl
import torch
import numpy as np
from argparse import Namespace
from tqdm import tqdm
from pytorch_lightning.loggers.base import merge_dicts
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler

from model import FastSpeech2Loss, FastSpeech2
from lightning.optimizer import get_optimizer
from lightning.scheduler import get_scheduler
from lightning.collate import get_single_collate
from lightning.sampler import GroupBatchSampler, DistributedBatchSampler
from lightning.utils import LightningMelGAN


class System(pl.LightningModule):
    """Base class for deep learning systems.
    Contains a model, an optimizer, a loss function, training and validation
    dataloaders and learning rate scheduler.

    Note that by default, any PyTorch-Lightning hooks are *not* passed to the model.
    If you want to use Lightning hooks, add the hooks to a subclass::

        class MySystem(System):
            def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
                return self.model.on_train_batch_start(batch, batch_idx, dataloader_idx)

    Args:
        model (torch.nn.Module): Instance of model.
        optimizer (torch.optim.Optimizer): Instance or list of optimizers.
        loss_func (callable): Loss function with signature
            (est_targets, targets).
        train_loader (torch.utils.data.DataLoader): Training dataloader.
        val_loader (torch.utils.data.DataLoader): Validation dataloader.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Instance, or list
            of learning rate schedulers. Also supports dict or list of dict as
            ``{"interval": "step", "scheduler": sched}`` where ``interval=="step"``
            for step-wise schedulers and ``interval=="epoch"`` for classical ones.
        config: Anything to be saved with the checkpoints during training.
            The config dictionary to re-instantiate the run for example.

    .. note:: By default, ``training_step`` (used by ``pytorch-lightning`` in the
        training loop) and ``validation_step`` (used for the validation loop)
        share ``common_step``. If you want different behavior for the training
        loop and the validation loop, overwrite both ``training_step`` and
        ``validation_step`` instead.

    For more info on its methods, properties and hooks, have a look at lightning's docs:
    https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html#lightningmodule-api
    """

    default_monitor: str = "val_loss"

    def __init__(
        self,
        model=None,
        optimizer=None,
        loss_func=None,
        train_dataset=None,
        val_dataset=None,
        scheduler=None,
        configs=None,
        vocoder=None,
        group_dataloader=False,
    ):
        super().__init__()
        preprocess_config, model_config, train_config = configs
        self.preprocess_config = preprocess_config
        self.model_config = model_config
        self.train_config = train_config

        if model is None:
            model = FastSpeech2(preprocess_config, model_config)
        if loss_func is None:
            loss_func = FastSpeech2Loss(preprocess_config, model_config)
        self.model = model
        self.loss_func = loss_func

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.vocoder = LightningMelGAN(vocoder)
        self.vocoder.freeze()
        # self.config = {} if config is None else config
        # hparams will be logged to Tensorboard as text variables.
        # summary writer doesn't support None for now, convert to strings.
        # See https://github.com/pytorch/pytorch/issues/33140
        # self.hparams = Namespace(**self.config_to_hparams(self.config))
        self.group_dataloader = group_dataloader

    def forward(self, *args, **kwargs):
        """Applies forward pass of the model.

        Returns:
            :class:`torch.Tensor`
        """
        return self.model(*args, **kwargs)

    def common_step(self, batch, batch_nb, train=True):
        """Common forward step between training and validation.

        The function of this method is to unpack the data given by the loader,
        forward the batch through the model and compute the loss.
        Pytorch-lightning handles all the rest.

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.
            train (bool): Whether in training mode. Needed only if the training
                and validation steps are fundamentally different, otherwise,
                pytorch-lightning handles the usual differences.

        Returns:
            :class:`torch.Tensor` : The loss value on this batch.

        .. note::
            This is typically the method to overwrite when subclassing
            ``System``. If the training and validation steps are somehow
            different (except for ``loss.backward()`` and ``optimzer.step()``),
            the argument ``train`` can be used to switch behavior.
            Otherwise, ``training_step`` and ``validation_step`` can be overwriten.
        """
        output = self(*(batch[2:]))
        loss = self.loss_func(batch, output)
        return loss

    def loss2str(self, loss):
        return self.dict2str(self.loss2dict(loss))
        # message = f"Total Loss: {loss[0]:.4f}, "
        # message += f"Mel Loss: {loss[1]:.4f}, "
        # message += f"Mel PostNet Loss: {loss[2]:.4f}, "
        # message += f"Pitch Loss: {loss[3]:.4f}, "
        # message += f"Energy Loss: {loss[4]:.4f}, "
        # message += f"Duration Loss: {loss[5]:.4f}"
        # return message

    def loss2dict(self, loss):
        tblog_dict = {
            "Loss/total_loss"       : loss[0].item(),
            "Loss/mel_loss"         : loss[1].item(),
            "Loss/mel_postnet_loss" : loss[2].item(),
            "Loss/pitch_loss"       : loss[3].item(),
            "Loss/energy_loss"      : loss[4].item(),
            "Loss/duration_loss"    : loss[5].item(),
        }
        return tblog_dict
    
    def dict2loss(self, tblog_dict):
        loss = (
            tblog_dict["Loss/total_loss"],
            tblog_dict["Loss/mel_loss"],
            tblog_dict["Loss/mel_postnet_loss"],
            tblog_dict["Loss/pitch_loss"],
            tblog_dict["Loss/energy_loss"],
            tblog_dict["Loss/duration_loss"],
        )
        return loss

    def dict2str(self, tblog_dict):
        def convert_key(key):
            new_key = ' '.join([e.title() for e in key.split('/')[-1].split('_')])
            return new_key
        message = ", ".join([f"{convert_key(k)}: {v:.4f}" for k, v in tblog_dict.items()])
        return message

    def training_step(self, batch, batch_nb):
        """Pass data through the model and compute the loss.

        Backprop is **not** performed (meaning PL will do it for you).

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.

        Returns:
            torch.Tensor, the value of the loss.
        """
        loss = self.common_step(batch, batch_nb, train=True)

        if (self.trainer.global_step+1) % self.trainer.log_every_n_steps == 0:
            message = f"Step {self.trainer.global_step+1}/{self.trainer.max_steps}, "
            tqdm.write(message + self.loss2str(loss))

            self.logger[0].log_metrics(self.loss2dict(loss), self.trainer.global_step+1)

        total_loss = loss[0]
        self.log('train_loss', total_loss)
        return total_loss

    def validation_step(self, batch, batch_nb):
        """Need to overwrite PL validation_step to do validation.

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.
        """
        loss = self.common_step(batch, batch_nb, train=False)
        tblog_dict = self.loss2dict(loss)

        total_loss = loss[0]
        self.log('val_loss', total_loss)
        return tblog_dict

    def validation_epoch_end(self, val_outputs=None):
        """Log hp_metric to tensorboard for hparams selection."""
        if self.trainer.global_step > 0:
            tblog_dict = merge_dicts(val_outputs)
            loss = self.dict2loss(tblog_dict)

            message = f"Validation Step {self.trainer.global_step+1}, "
            tqdm.write(message + self.loss2str(loss))

            self.logger[1].log_metrics(tblog_dict, self.trainer.global_step+1)

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        if self.optimizer is None:
            self.optimizer = get_optimizer(self.model, self.model_config, self.train_config)

        if self.scheduler is None:
            self.scheduler = {
                "scheduler": get_scheduler(self.optimizer, self.train_config),
                'interval': 'step',
                'frequency': 1,
                'monitor': self.default_monitor,
            }
        else:
            if not isinstance(self.scheduler, dict):
                self.scheduler = {
                    "scheduler": self.scheduler,
                    'interval': 'step',
                    'frequency': 1,
                    "monitor": self.default_monitor
                }
            else:
                self.scheduler.setdefault("monitor", self.default_monitor)
                self.scheduler.setdefault("frequency", 1)
                assert self.scheduler["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"

        return [self.optimizer], [self.scheduler]

    def train_dataloader(self):
        """Training dataloader"""
        sampler = RandomSampler(self.train_dataset)
        batch_size = self.train_config["optimizer"]["batch_size"]
        if self.group_dataloader:
            group_size = 4
            batch_sampler = GroupBatchSampler(sampler, group_size, batch_size, drop_last=True, sort=True)
            ddp_batch_sampler = DistributedBatchSampler(batch_sampler)
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_sampler=ddp_batch_sampler, 
                collate_fn=get_single_collate(False),
                num_workers=8,
            )
        else:
            # batch_sampler = BatchSampler(sampler, batch_size, drop_last=True)
            self.train_loader = DataLoader(
                self.train_dataset,
                # batch_sampler=batch_sampler, 
                shuffle=True,
                batch_size=batch_size,
                drop_last=True,
                collate_fn=get_single_collate(False),
                num_workers=8,
            )
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        sampler = SequentialSampler(self.val_dataset)
        batch_size = self.train_config["optimizer"]["batch_size"]
        if self.group_dataloader:
            group_size = 4
            batch_sampler = GroupBatchSampler(sampler, group_size, batch_size, drop_last=False, sort=False)
            ddp_batch_sampler = DistributedBatchSampler(batch_sampler)
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_sampler=ddp_batch_sampler, 
                collate_fn=get_single_collate(False),
                num_workers=8,
            )
        else:
            # batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)
            self.val_loader = DataLoader(
                self.val_dataset,
                shuffle=False,
                batch_size=batch_size,
                drop_last=False,
                collate_fn=get_single_collate(False),
                num_workers=8,
            )
        return self.val_loader

    # def on_save_checkpoint(self, checkpoint):
        # """Overwrite if you want to save more things in the checkpoint."""
        # checkpoint["training_config"] = self.config
        # return checkpoint

