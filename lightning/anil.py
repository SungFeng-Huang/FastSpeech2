#!/usr/bin/env python3

import os
import json
import pytorch_lightning as pl
import torch
import numpy as np
import learn2learn as l2l
from tqdm import tqdm
from pytorch_lightning.loggers.base import merge_dicts
from torch.utils.data import DataLoader

from learn2learn.algorithms.lightning import LightningMAML
from learn2learn.utils.lightning import EpisodicBatcher

from model import FastSpeech2Loss, FastSpeech2
from utils.tools import get_mask_from_lengths
from lightning.system import System
from lightning.collate import get_meta_collate


class ANILSystem(System):
    """A PyTorch Lightning module for ANIL for FastSpeech2.
    """

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
    ):
        super().__init__(model, optimizer, loss_func, train_dataset, val_dataset, scheduler, configs, vocoder)

        self.train_ways     = self.train_config["meta"]["ways"]
        self.train_shots    = self.train_config["meta"]["shots"]
        self.train_queries  = self.train_config["meta"]["queries"]
        self.test_ways      = self.train_config["meta"]["ways"]
        self.test_shots     = self.train_config["meta"]["shots"]
        self.test_queries   = 1
        # self.test_queries   = self.train_config["meta"]["queries"]
        assert self.train_ways == self.test_ways, \
            "For ANIL, train_ways should be equal to test_ways."
        
        meta_config = self.train_config["meta"]
        self.adaptation_steps   = meta_config.get("adaptation_steps", LightningMAML.adaptation_steps)
        self.adaptation_lr      = meta_config.get("adaptation_lr", LightningMAML.adaptation_lr)
        self.data_parallel      = meta_config.get("data_parallel", False)

        self.encoder = self.model.encoder
        self.total_decoder = torch.nn.ModuleDict({
            'variance_adaptor'  : self.model.variance_adaptor,
            'decoder'           : self.model.decoder,
            'mel_linear'        : self.model.mel_linear,
            'postnet'           : self.model.postnet,
            'speaker_emb'       : self.model.speaker_emb,
        })

        if self.data_parallel and torch.cuda.device_count() > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)
        self.total_decoder = l2l.algorithms.MAML(self.total_decoder, lr=self.adaptation_lr)

    @torch.enable_grad()
    def meta_learn(self, batch, batch_idx, ways, shots, queries):
        learner = self.total_decoder.clone()
        self.encoder.train()
        learner.train()

        sup_batch = batch[0][0][0]
        qry_batch = batch[0][1][0]

        sup_enc_output, sup_src_masks = self.forward_encoder(*(sup_batch[2:]))
        qry_enc_output, qry_src_masks = self.forward_encoder(*(qry_batch[2:]))

        # Adapt the classifier
        for step in range(self.adaptation_steps):
            preds = self.forward_learner(
                learner, *(sup_batch[2:]), output=sup_enc_output, src_masks=sup_src_masks
            )
            train_error = self.loss_func(sup_batch, preds)
            learner.adapt(train_error[0], allow_unused=False, allow_nograd=True)

        # Evaluating the adapted model
        predictions = self.forward_learner(
            learner, *(qry_batch[2:]), output=qry_enc_output, src_masks=qry_src_masks
        )
        valid_error = self.loss_func(qry_batch, predictions)
        return valid_error, predictions

    def training_step(self, batch, batch_idx):
        assert len(batch) == 1, "meta_batch_per_gpu"
        assert len(batch[0]) == 2, "sup + qry"
        assert len(batch[0][0]) == 1, "n_batch == 1"
        assert len(batch[0][0][0]) == 12, "data with 12 elements"

        train_loss, predictions = self.meta_learn(
            batch, batch_idx, self.train_ways, self.train_shots, self.train_queries
        )

        sup_ids = batch[0][0][0][0]
        qry_ids = batch[0][1][0][0]

        # Synthesis one sample
        if (self.global_step+1) % self.train_config["step"]["synth_step"] == 0 and self.local_rank == 0:
            qry_batch = batch[0][1][0]
            metadata = {'stage': "Training", 'sup_ids': sup_ids}
            fig, wav_reconstruction, wav_prediction, basename = self.synth_one_sample(
                qry_batch, predictions
            )
            step = self.global_step+1
            self.log_figure(
                f"Training/step_{step}_{basename}",
                fig
            )
            self.log_audio(
                f"Training/step_{step}_{basename}_reconstructed",
                wav_reconstruction,
                metadata=metadata
            )
            self.log_audio(
                f"Training/step_{step}_{basename}_synthesized",
                wav_prediction,
                metadata=metadata
            )

        # Log message to log.txt and print to stdout
        if (self.global_step+1) % self.trainer.log_every_n_steps == 0:
            message = f"Step {self.global_step+1}/{self.trainer.max_steps}, "
            message += self.loss2str(train_loss)
            self.log_text(message)

        # Log metrics to CometLogger
        comet_log_dict = {f"train_{k}":v for k,v in self.loss2dict(train_loss).items()}
        self.log_dict(comet_log_dict, sync_dist=True)
        return train_loss[0]

    def validation_step(self, batch, batch_idx):
        assert len(batch) == 1, "meta_batch_per_gpu"
        assert len(batch[0]) == 2, "sup + qry"
        assert len(batch[0][0]) == 1, "n_batch == 1"
        assert len(batch[0][0][0]) == 12, "data with 12 elements"

        sup_ids = batch[0][0][0][0]
        qry_ids = batch[0][1][0][0]

        val_loss, predictions = self.meta_learn(
            batch, batch_idx, self.test_ways, self.test_shots, self.test_queries
        )

        # Sample one sample and log to CometLogger
        if batch_idx == 0 and self.local_rank == 0:
            metadata = {'stage': "Validation", 'sup_ids': sup_ids}
            qry_batch = batch[0][1][0]
            fig, wav_reconstruction, wav_prediction, basename = self.synth_one_sample(qry_batch, predictions)
            step = self.global_step+1
            self.log_figure(
                f"Validation/step_{step}_{basename}", fig
            )
            self.log_audio(
                f"Validation/step_{step}_{basename}_reconstructed", wav_reconstruction,
                metadata=metadata
            )
            self.log_audio(
                f"Validation/step_{step}_{basename}_synthesized", wav_prediction,
                metadata=metadata
            )

        # Log loss for each sample to csv files
        self.log_val_csv(sup_ids, qry_ids, val_loss)

        # Log metrics to CometLogger
        tblog_dict = self.loss2dict(val_loss)
        self.log_dict(
            {f"val_{k}":v for k, v in tblog_dict.items()}, sync_dist=True,
        )
        return tblog_dict

    def validation_epoch_end(self, val_outputs=None):
        """Log hp_metric to tensorboard for hparams selection."""
        if self.global_step > 0:
            tblog_dict = merge_dicts(val_outputs)
            loss = self.dict2loss(tblog_dict)

            # Log total loss to log.txt and print to stdout
            message = f"Validation Step {self.global_step+1}, "
            message += self.loss2str(loss)
            self.log_text(message)

    def train_dataloader(self):
        # Make meta-dataset, to apply 1-way-5-shots tasks
        id2lb = {k:v for k,v in enumerate(self.train_dataset.speaker)}
        meta_dataset = l2l.data.MetaDataset(self.train_dataset, indices_to_labels=id2lb)

        # 1-way-5-shots transforms
        transforms = [
            l2l.data.transforms.FusedNWaysKShots(
                meta_dataset, n=self.train_ways, k=self.train_shots+self.train_queries,
                replacement=True
            ),
            l2l.data.transforms.LoadData(meta_dataset),
        ]

        # 1-way-5-shot task dataset
        tasks = l2l.data.TaskDataset(
            meta_dataset,
            task_transforms=transforms,
            task_collate=get_meta_collate(self.train_shots, self.train_queries, False),
        )

        # Epochify task dataset, for periodic validation
        meta_batch_size = self.train_config["meta"]["meta_batch_size"]
        val_step = self.train_config["step"]["val_step"]
        episodic_tasks = EpisodicBatcher(
            tasks, epoch_length=meta_batch_size*val_step,
        )

        # DataLoader, batch_size = batch size on each gpu
        self.train_loader = DataLoader(
            episodic_tasks.train_dataloader(),
            batch_size=1,
            shuffle=True,
            collate_fn=lambda batch: batch,
            num_workers=8,
        )
        return self.train_loader

    def val_dataloader(self):
        # Fix random seed for resume training and comparison
        pl.seed_everything(42, True)

        # Make meta-dataset, to apply 1-way-5-shots tasks
        id2lb = {k:v for k,v in enumerate(self.val_dataset.speaker)}
        meta_dataset = l2l.data.MetaDataset(self.val_dataset, indices_to_labels=id2lb)

        # 1-way-5-shots tasks for each speaker
        val_tasks = []
        task_per_speaker = 8
        for label, indices in meta_dataset.labels_to_indices.items():
            if len(indices) >= self.test_shots+self.test_queries:
                transforms = [
                    l2l.data.transforms.FusedNWaysKShots(
                        meta_dataset, n=self.test_ways, k=self.test_shots+self.test_queries,
                        replacement=False, filter_labels=[label]
                    ),
                    l2l.data.transforms.LoadData(meta_dataset),
                ]
                tasks = l2l.data.TaskDataset(
                    meta_dataset,
                    task_transforms=transforms,
                    task_collate=get_meta_collate(self.test_shots, self.test_queries, False),
                    num_tasks=task_per_speaker,
                )
                val_tasks.append(tasks)
        val_concat_tasks = torch.utils.data.ConcatDataset(val_tasks)

        # Pre-run validation dataset to setup ids (with settled random seed)
        val_SQids = []
        self.val_SQids2vid = {}
        if not hasattr(self, 'log_dir'):
            self.log_dir = os.path.join(self.logger[0]._save_dir, self.logger[0].version)
            os.makedirs(self.log_dir, exist_ok=True)
        for i, task in enumerate(val_concat_tasks):
            sup_ids = task[0][0][0]
            qry_ids = task[1][0][0]
            val_SQids.append({'sup_id': sup_ids, 'qry_id': qry_ids})

            SQids = f"{'-'.join(sup_ids)}.{'-'.join(qry_ids)}"
            self.val_SQids2vid[SQids] = f"val_{i:03d}"
            os.makedirs(os.path.join(self.log_dir, "val_csv"), exist_ok=True)
            with open(os.path.join(self.log_dir, "val_csv", f"{self.val_SQids2vid[SQids]}.csv"), 'w') as f:
                f.write(
                    "step, total_loss, mel_loss, mel_postnet_loss, pitch_loss, energy_loss, duration_loss\n"
                )
        with open(os.path.join(self.log_dir, "val_SQids.json"), 'w') as f:
            json.dump(val_SQids, f, indent=4)

        # DataLoader
        self.val_loader = DataLoader(
            val_concat_tasks,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: batch,
            num_workers=8,
        )
        return self.val_loader

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["val_SQids2vid"] = self.val_SQids2vid
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        self.val_SQids2vid = checkpoint["val_SQids2vid"]

    def log_val_csv(self, sup_ids, qry_ids, losses):
        SQids = f"{'-'.join(sup_ids)}.{'-'.join(qry_ids)}"
        with open(os.path.join(self.log_dir, "val_csv", f"{self.val_SQids2vid[SQids]}.csv"), 'a') as f:
            f.write(f"{self.global_step}")
            for loss in losses:
                f.write(f", {loss.item()}")
            f.write("\n")

    def forward_encoder(self, *batch):
        texts = batch[1]
        src_lens = batch[2]
        max_src_len = batch[3]
        src_masks = get_mask_from_lengths(src_lens, max_src_len)

        output = self.encoder(texts, src_masks)

        return output, src_masks

    def forward_learner(
        self, learner, speakers, texts, src_lens, max_src_len,
        mels=None, mel_lens=None, max_mel_len=None,
        p_targets=None, e_targets=None, d_targets=None,
        p_control=1.0, e_control=1.0, d_control=1.0,
        output=None, src_masks=None,
    ):
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None
        )

        if learner.module['speaker_emb'] is not None:
            output = output + learner.module['speaker_emb'](speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output, p_predictions, e_predictions, log_d_predictions, d_rounded, mel_lens, mel_masks,
        ) = learner.module['variance_adaptor'](
            output, src_masks, mel_masks, max_mel_len,
            p_targets, e_targets, d_targets, p_control, e_control, d_control,
        )

        if learner.module['speaker_emb'] is not None:
            output = output + learner.module['speaker_emb'](speakers).unsqueeze(1).expand(
                -1, max(mel_lens), -1
            )

        output, mel_masks = learner.module['decoder'](output, mel_masks)
        output = learner.module['mel_linear'](output)

        postnet_output = learner.module['postnet'](output) + output

        return (
            output, postnet_output,
            p_predictions, e_predictions, log_d_predictions, d_rounded,
            src_masks, mel_masks, src_lens, mel_lens,
        )
