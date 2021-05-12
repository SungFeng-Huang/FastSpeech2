#!/usr/bin/env python3

import torch

from model import FastSpeech2Loss, FastSpeech2
from lightning.baseline import BaselineSystem


class BaselineEmb1System(BaselineSystem):
    """A PyTorch Lightning module for ANIL for FastSpeech2.
    1 embedding for all speakers.
    """

    def __init__(
        self,
        model=None,
        optimizer=None,
        loss_func=None,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        scheduler=None,
        configs=None,
        vocoder=None,
        log_dir=None,
        result_dir=None,
     ):
        preprocess_config, model_config, train_config = configs

        if model is None:
            model = FastSpeech2(preprocess_config, model_config)
        if loss_func is None:
            loss_func = FastSpeech2Loss(preprocess_config, model_config)

        if model_config["multi_speaker"]:
            del model.speaker_emb
            model.speaker_emb = torch.nn.Embedding(
                1,
                model_config["transformer"]["encoder_hidden"],
            )
        
        del optimizer
        del scheduler
        optimizer = None
        scheduler = None

        super().__init__(
            model, optimizer, loss_func,
            train_dataset, val_dataset, test_dataset,
            scheduler, configs, vocoder,
            log_dir, result_dir
        )

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        with torch.no_grad():
            batch[2].zero_()

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)
        with torch.no_grad():
            batch[0][0][0][2].zero_()
            batch[0][1][0][2].zero_()

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)
        with torch.no_grad():
            batch[0][0][0][2].zero_()
            batch[0][1][0][2].zero_()
