import pytorch_lightning as pl

class LightningMelGAN(pl.LightningModule):
    def __init__(self, vocoder):
        super().__init__()
        self.mel2wav = vocoder.mel2wav

    def forward(self, mel):
        with torch.no_grad():
            return self.mel2wav(mel).unsqueeze(1)

def split_data(batch, support_indices):
    (
        ids,
        raw_texts,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        max_mel_len,
        pitches,
        energies,
        durations,
        enc_output,
        src_masks,
    ) = batch

    support_idx = []
    query_idx = []
    support_raw_texts = []
    query_raw_texts = []
    for i in support_indices:
        if i:
            support_idx.append(ids[i])
            support_raw_texts.append(raw_texts[i])
        else:
            query_idx.append(ids[i])
            query_raw_texts.append(raw_texts[i])

    support_set = (
        support_idx,
        support_raw_texts,
        speakers[support_indices],
        texts[support_indices],
        src_lens[support_indices],
        max(src_lens[support_indices]),
        mels[support_indices],
        mel_lens[support_indices],
        max(mel_lens[support_indices]),
        pitches[support_indices],
        energies[support_indices],
        durations[support_indices],
        enc_output[support_indices],
        src_masks[support_indices],
    )
    query_set = (
        query_idx,
        query_raw_texts,
        speakers[query_indices],
        texts[query_indices],
        src_lens[query_indices],
        max(src_lens[query_indices]),
        mels[query_indices],
        mel_lens[query_indices],
        max(mel_lens[query_indices]),
        pitches[query_indices],
        energies[query_indices],
        durations[query_indices],
        enc_output[query_indices],
        src_masks[query_indices],
    )
    return support_set, query_set

