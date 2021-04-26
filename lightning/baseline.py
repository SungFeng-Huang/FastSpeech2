#!/usr/bin/env python3

import os
import json
import torch
import numpy as np
import pytorch_lightning as pl
import learn2learn as l2l
from scipy.io import wavfile
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
from pytorch_lightning.loggers.base import merge_dicts
from learn2learn.algorithms.lightning import LightningMAML
from learn2learn.utils.lightning import EpisodicBatcher

from model import FastSpeech2Loss, FastSpeech2
from utils.tools import get_mask_from_lengths, expand, plot_mel
from lightning.system import System
from lightning.collate import get_meta_collate, get_multi_collate, get_single_collate
from lightning.utils import seed_all, EpisodicInfiniteWrapper


class BaselineSystem(System):
    """A PyTorch Lightning module for ANIL for FastSpeech2.
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
    ):
        super().__init__(model, optimizer, loss_func, train_dataset, val_dataset, test_dataset, scheduler, configs, vocoder)

        # All of the settings below are for few-shot validation

        self.test_ways      = self.train_config["meta"]["ways"]
        self.test_shots     = self.train_config["meta"]["shots"]
        self.test_queries   = 1
        # self.test_queries   = self.train_config["meta"]["queries"]
        
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
    def adapt(self, batch, adaptation_steps=5, learner=None):
        if learner is None:
            learner = self.total_decoder.clone()
            learner.train()

        sup_batch = batch[0][0][0]
        sup_enc_output, sup_src_masks = self.forward_encoder(*(sup_batch[2:]))

        # Adapt the classifier
        for step in range(adaptation_steps):
            preds = self.forward_learner(
                learner, *(sup_batch[2:]), output=sup_enc_output, src_masks=sup_src_masks
            )
            train_error = self.loss_func(sup_batch, preds)
            learner.adapt(train_error[0], allow_unused=False, allow_nograd=True)
        return learner

    def meta_learn(self, batch, batch_idx, ways, shots, queries):
        # self.encoder.train()
        learner = self.adapt(batch, self.adaptation_steps)

        # Evaluating the adapted model
        qry_batch = batch[0][1][0]
        qry_enc_output, qry_src_masks = self.forward_encoder(*(qry_batch[2:]))
        predictions = self.forward_learner(
            learner, *(qry_batch[2:]), output=qry_enc_output, src_masks=qry_src_masks
        )
        valid_error = self.loss_func(qry_batch, predictions)
        return valid_error, predictions

    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)

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

        # Synthesis one sample and log to CometLogger
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
        self.log_val_csv(sup_ids, qry_ids, val_loss, self.log_dir)

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

    @torch.enable_grad()
    def test_step(self, batch, batch_idx):
        ways, shots, queries = self.test_ways, self.test_shots, self.test_queries
        assert len(batch) == 1, "meta_batch_per_gpu"
        assert len(batch[0]) == 2, "sup + qry"
        assert len(batch[0][0]) == 1, "n_batch == 1"
        assert len(batch[0][0][0]) == 12, "data with 12 elements"

        sup_batch = batch[0][0][0]
        qry_batch = batch[0][1][0]
        sup_ids = sup_batch[0]
        qry_ids = qry_batch[0]
        metadata = {'stage': "Testing", 'sup_ids': sup_ids}
        SQids = f"{'-'.join(sup_ids)}.{'-'.join(qry_ids)}"
        os.makedirs(os.path.join(self.result_dir, "test_csv", f"step_{self.test_global_step}"), exist_ok=True)
        with open(os.path.join(self.result_dir, "test_csv", f"step_{self.test_global_step}",
                               f"{self.test_SQids2vid[SQids]}.csv"), 'w') as f:
            f.write(
                "step, total_loss, mel_loss, mel_postnet_loss, pitch_loss, energy_loss, duration_loss\n"
            )

        # Evaluating the initial model
        qry_enc_output, qry_src_masks = self.forward_encoder(*(qry_batch[2:6]))
        predictions = self.forward_learner(
            self.total_decoder, *(qry_batch[2:]),
            output=qry_enc_output, src_masks=qry_src_masks
        )
        valid_error = self.loss_func(qry_batch, predictions)

        with open(os.path.join(self.result_dir, "test_csv", f"step_{self.test_global_step}",
                               f"{self.test_SQids2vid[SQids]}.csv"), 'a') as f:
            f.write(f"0")
            for loss in valid_error:
                f.write(f", {loss.item()}")
            f.write("\n")

        self.recon_samples(
            qry_batch, predictions,
            tag=f"Testing/{self.test_SQids2vid[SQids]}",
            log_dir=self.result_dir,
        )

        # synth_samples & save & log
        predictions = self.forward_learner(
            self.total_decoder, *(qry_batch[2:6]),
            output=qry_enc_output, src_masks=qry_src_masks
        )
        self.synth_samples(
            qry_batch, predictions,
            tag=f"Testing/{self.test_SQids2vid[SQids]}",
            name=f"step_{self.test_global_step}-FTstep_0",
            log_dir=self.result_dir,
        )

        # Adapt
        learner = self.adapt(batch, self.adaptation_steps)

        # Evaluating the adapted model
        predictions = self.forward_learner(
            learner, *(qry_batch[2:]),
            output=qry_enc_output, src_masks=qry_src_masks
        )
        valid_error = self.loss_func(qry_batch, predictions)

        with open(os.path.join(self.result_dir, "test_csv", f"step_{self.test_global_step}",
                               f"{self.test_SQids2vid[SQids]}.csv"), 'a') as f:
            f.write(f"{self.adaptation_steps}")
            for loss in valid_error:
                f.write(f", {loss.item()}")
            f.write("\n")

        # synth_samples & save & log
        predictions = self.forward_learner(
            learner, *(qry_batch[2:6]),
            output=qry_enc_output, src_masks=qry_src_masks
        )
        self.synth_samples(
            qry_batch, predictions,
            tag=f"Testing/{self.test_SQids2vid[SQids]}",
            name=f"step_{self.test_global_step}-FTstep_{self.adaptation_steps}",
            log_dir=self.result_dir,
        )

        # Log loss for each sample to csv files
        self.log_test_csv(sup_ids, qry_ids, valid_error, log_dir=self.result_dir)

        # Log metrics to CometLogger
        # tblog_dict = self.loss2dict(val_loss)
        # self.log_dict(
            # {f"val_{k}":v for k, v in tblog_dict.items()}, sync_dist=True,
        # )
        # return tblog_dict

    def train_dataloader(self):
        if not isinstance(self.train_dataset, EpisodicInfiniteWrapper):
            meta_batch_size = self.train_config["meta"]["meta_batch_size"]
            train_ways      = self.train_config["meta"]["ways"]
            train_shots     = self.train_config["meta"]["shots"]
            train_queries   = self.train_config["meta"]["queries"]
            batch_size = train_ways * (train_shots + train_queries) * meta_batch_size
            # batch_size = self.train_config["optimizer"]["batch_size"]
            val_step = self.train_config["step"]["val_step"]
            self.train_dataset = EpisodicInfiniteWrapper(self.train_dataset, val_step*batch_size)
        self.train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            drop_last=True,
            collate_fn=get_single_collate(False),
            num_workers=8,
        )
        return self.train_loader

    def val_dataloader(self):
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

        # Fix random seed for validation set. Don't use pl.seed_everything(43,
        # True) if don't want to affect training seed. Use my seed_all instead.
        with seed_all(43):
            self.val_SQids2vid = self.prefetch_tasks(val_concat_tasks, 'val', self.log_dir)

        # DataLoader
        self.val_loader = DataLoader(
            val_concat_tasks,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: batch,
            num_workers=8,
        )
        return self.val_loader

    def test_dataloader(self):
        # Make meta-dataset, to apply 1-way-5-shots tasks
        id2lb = {k:v for k,v in enumerate(self.test_dataset.speaker)}
        meta_dataset = l2l.data.MetaDataset(self.test_dataset, indices_to_labels=id2lb)

        # 1-way-5-shots tasks for each speaker
        test_tasks = []
        task_per_speaker = 40
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
                test_tasks.append(tasks)
        test_concat_tasks = torch.utils.data.ConcatDataset(test_tasks)

        # Fix random seed for testing set. Don't use pl.seed_everything(43,
        # True) if don't want to affect training seed. Use my seed_all instead.
        with seed_all(43):
            self.test_SQids2vid = self.prefetch_tasks(test_concat_tasks, 'test', self.result_dir)

        # DataLoader
        self.test_loader = DataLoader(
            test_concat_tasks,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: batch,
            num_workers=8,
        )
        return self.test_loader

    def prefetch_tasks(self, tasks, tag='val', log_dir=''):
        if tag == 'val' and not hasattr(self, 'log_dir'):
            self.log_dir = os.path.join(self.logger[0]._save_dir, self.logger[0].version)
        os.makedirs(os.path.join(log_dir, f"{tag}_csv"), exist_ok=True)

        SQids = []
        SQids2id = {}
        for i, task in enumerate(tasks):
            sup_ids = task[0][0][0]
            qry_ids = task[1][0][0]
            SQids.append({'sup_id': sup_ids, 'qry_id': qry_ids})

            SQid = f"{'-'.join(sup_ids)}.{'-'.join(qry_ids)}"
            SQids2id[SQid] = f"{tag}_{i:03d}"

        for SQid in SQids2id:
            with open(os.path.join(log_dir, f"{tag}_csv", f"{SQids2id[SQid]}.csv"), 'w') as f:
                f.write(
                    "step, total_loss, mel_loss, mel_postnet_loss, pitch_loss, energy_loss, duration_loss\n"
                )

        with open(os.path.join(log_dir, f"{tag}_SQids.json"), 'w') as f:
            json.dump(SQids, f, indent=4)

        return SQids2id

    def recon_samples(self, targets, predictions, tag='Testing', log_dir=''):
        """Synthesize the first sample of the batch."""
        for i in range(len(predictions[0])):
            basename    = targets[0][i]
            src_len     = predictions[8][i].item()
            mel_len     = predictions[9][i].item()
            mel_target  = targets[6][i, :mel_len].detach().transpose(0, 1)
            duration    = targets[11][i, :src_len].detach().cpu().numpy()
            pitch       = targets[9][i, :src_len].detach().cpu().numpy()
            energy      = targets[10][i, :src_len].detach().cpu().numpy()
            if self.preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
                pitch = expand(pitch, duration)
            if self.preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
                energy = expand(energy, duration)

            with open(
                os.path.join(self.preprocess_config["path"]["preprocessed_path"], "stats.json")
            ) as f:
                stats = json.load(f)
                stats = stats["pitch"] + stats["energy"][:2]

            fig = plot_mel(
                [
                    (mel_target.cpu().numpy(), pitch, energy),
                ],
                stats,
                ["Ground-Truth Spectrogram"],
            )
            os.makedirs(os.path.join(log_dir, "figure", f"{tag}"), exist_ok=True)
            plt.savefig(os.path.join(log_dir, "figure", f"{tag}/{basename}.target.png"))
            plt.close()

        mel_targets = targets[6].transpose(1, 2)
        lengths = predictions[9] * self.preprocess_config["preprocessing"]["stft"]["hop_length"]
        max_wav_value = self.preprocess_config["preprocessing"]["audio"]["max_wav_value"]
        wav_targets = self.vocoder.infer(mel_targets, max_wav_value, lengths=lengths)

        sampling_rate = self.preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        os.makedirs(os.path.join(log_dir, "audio", f"{tag}"), exist_ok=True)
        for wav, basename in zip(wav_targets, targets[0]):
            wavfile.write(os.path.join(log_dir, "audio", f"{tag}/{basename}.recon.wav"), sampling_rate, wav)

        # return fig, wav_prediction, basename

    def synth_samples(self, targets, predictions, tag='Testing', name='', log_dir=''):
        """Synthesize the first sample of the batch."""
        for i in range(len(predictions[0])):
            basename        = targets[0][i]
            src_len         = predictions[8][i].item()
            mel_len         = predictions[9][i].item()
            mel_prediction  = predictions[1][i, :mel_len].detach().transpose(0, 1)
            duration        = predictions[5][i, :src_len].detach().cpu().numpy()
            pitch           = predictions[2][i, :src_len].detach().cpu().numpy()
            energy          = predictions[3][i, :src_len].detach().cpu().numpy()
            if self.preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
                pitch = expand(pitch, duration)
            if self.preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
                energy = expand(energy, duration)

            with open(
                os.path.join(self.preprocess_config["path"]["preprocessed_path"], "stats.json")
            ) as f:
                stats = json.load(f)
                stats = stats["pitch"] + stats["energy"][:2]

            fig = plot_mel(
                [
                    (mel_prediction.cpu().numpy(), pitch, energy),
                ],
                stats,
                ["Synthetized Spectrogram"],
            )
            os.makedirs(os.path.join(log_dir, "figure", f"{tag}"), exist_ok=True)
            plt.savefig(os.path.join(log_dir, "figure", f"{tag}/{basename}.{name}.synth.png"))
            plt.close()

        mel_predictions = predictions[1].transpose(1, 2)
        lengths = predictions[9] * self.preprocess_config["preprocessing"]["stft"]["hop_length"]
        max_wav_value = self.preprocess_config["preprocessing"]["audio"]["max_wav_value"]
        wav_predictions = self.vocoder.infer(mel_predictions, max_wav_value, lengths=lengths)

        sampling_rate = self.preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        os.makedirs(os.path.join(log_dir, "audio", f"{tag}"), exist_ok=True)
        for wav, basename in zip(wav_predictions, targets[0]):
            wavfile.write(os.path.join(log_dir, "audio", f"{tag}/{basename}.{name}.synth.wav"), sampling_rate, wav)

        # return fig, wav_prediction, basename

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["val_SQids2vid"] = self.val_SQids2vid
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        self.val_SQids2vid = checkpoint["val_SQids2vid"]
        self.test_global_step = checkpoint["global_step"]

    def log_val_csv(self, sup_ids, qry_ids, losses, log_dir=''):
        SQids = f"{'-'.join(sup_ids)}.{'-'.join(qry_ids)}"
        with open(os.path.join(log_dir, "val_csv", f"{self.val_SQids2vid[SQids]}.csv"), 'a') as f:
            f.write(f"{self.global_step}")
            for loss in losses:
                f.write(f", {loss.item()}")
            f.write("\n")

    def log_test_csv(self, sup_ids, qry_ids, losses, log_dir=''):
        SQids = f"{'-'.join(sup_ids)}.{'-'.join(qry_ids)}"
        with open(os.path.join(log_dir, "test_csv", f"{self.test_SQids2vid[SQids]}.csv"), 'a') as f:
            f.write(f"{self.test_global_step}")
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
        self,
        learner,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        output=None,
        src_masks=None,
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
