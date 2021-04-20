import pytorch_lightning as pl
import numpy as np
import torch


class LightningMelGAN(pl.LightningModule):
    def __init__(self, vocoder):
        super().__init__()
        self.mel2wav = vocoder.mel2wav

    def inverse(self, mel):
        with torch.no_grad():
            return self.mel2wav(mel).squeeze(1)

    def infer(self, mels, max_wav_value, lengths=None):
        """preprocess_config["preprocessing"]["audio"]["max_wav_value"]
        """
        wavs = self.inverse(mels / np.log(10))
        wavs = (wavs.cpu().numpy() * max_wav_value).astype("int16")
        wavs = [wav for wav in wavs]

        for i in range(len(mels)):
            if lengths is not None:
                wavs[i] = wavs[i][: lengths[i]]
        return wavs


