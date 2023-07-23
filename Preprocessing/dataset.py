import datasets
import numpy as np
import torch
import torchaudio.transforms as T
from datasets import load_dataset
from huggingface_hub.hf_api import HfFolder
from itypes import List
from transformers import BartTokenizer


class DatasetProcessor:
    def __init__(self, num_proc):
        self.num_proc = num_proc
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    def download(self, dataset_name, language_code, token, split="train") -> datasets.Dataset:
        HfFolder.save_token(token)
        return load_dataset(dataset_name, language_code, split=split, use_auth_token=True)

    def find_max_audio(self, audios: list) -> int:
        return np.max([audio["array"].shape[0] for audio in audios])

    def pad_audio(self, audio: np.ndarray, max_len: int) -> np.ndarray:
        return np.pad(audio, (0, max_len - audio.shape[0]), mode="constant", constant_values=0)

    def make_spectrogram(self, audio, sample_rate=32000, n_fft=1024, win_length=None, hop_length=512,
                         n_mels=64) -> torch.Tensor:
        spect = T.MelSpectrogram(
            sample_rate=audio["sampling_rate"],
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=n_mels,
            mel_scale="htk",
        )
        spect = spect.to(torch.float32)
        return spect(torch.from_numpy(audio["array"]).to(torch.float32))

    def pre_process(self, datasets_: List[datasets.Dataset]) -> List[datasets.Dataset]:
        max_len = max([self.find_max_audio(dataset["audio"]) for dataset in datasets_])
        print(f"Max audio length: {max_len}")

        def process(batch):
            batch["audio"] = self.pad_audio(batch["audio"], max_len)
            batch["audio"] = self.make_spectrogram(batch["audio"])
            batch["sentence"] = self.tokenizer(batch["sentence"], truncation=True, padding=True, return_tensors="pt")[
                "input_ids"]

            # to tensors
            batch["sentence"] = torch.tensor(batch["sentence"])
            batch["audio"] = torch.tensor(batch["audio"])
            return batch

        datasets_ = [dataset.map(process, num_proc=self.num_proc, batched=True) for dataset in datasets_]

        return datasets_
