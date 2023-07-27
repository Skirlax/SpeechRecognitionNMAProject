import datasets
import librosa
import torch
import torchaudio.transforms as T
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from huggingface_hub.hf_api import HfFolder
from transformers import BartTokenizer, AutoTokenizer
from functools import partial

class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        sample = self.data[index]["audio"]["array"]
        label = self.data[index]["sentence"]

        # Apply transformation
        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def __len__(self):
        return self.data.num_rows


def collate_fn(batch,max_length, sample_rate,dataset_processor):
    inputs,labels = zip(*batch)
    inputs = dataset_processor.preprocess_sample(inputs, max_length, sample_rate).to("cuda")
    labels = torch.stack([torch.tensor(label) for label in labels]).to("cuda")
    return inputs,labels



class DataProcessor:
    def __init__(self, num_proc):
        self.num_proc = num_proc
        self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-base",use_legacy=False)

    def download(self, dataset_name, language_code, token, split="train") -> datasets.Dataset:
        HfFolder.save_token(token)
        return load_dataset(dataset_name, language_code, split=split, use_auth_token=True)

    def make_spectrogram(self, audio, sample_rate=32000, n_fft=1024, win_length=None, hop_length=512,
                         n_mels=64) -> torch.Tensor:
        new_sample_rate = 16000
        audio = librosa.resample(audio, target_sr=new_sample_rate, orig_sr=sample_rate)
        spect = T.MelSpectrogram(
            sample_rate=new_sample_rate,
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
        return spect(torch.from_numpy(audio).to(torch.float32)).reshape(1, 64, -1)

    def pre_process_text(self, datasets_: list[datasets.Dataset]) -> list[datasets.Dataset]:
        max_sentence_length = 193 # max([max([len(sample["sentence"]) for sample in dataset]) for dataset in datasets_])
        print("Max sentence length: ", max_sentence_length)

        def tokenize(batch):
            batch["sentence"] = self.tokenizer(batch["sentence"], truncation=True, padding="max_length",
                                               max_length=max_sentence_length, return_tensors="pt")["input_ids"]

            return batch

        datasets_ = [dataset.map(tokenize, num_proc=self.num_proc) for dataset in
                     datasets_]

        return datasets_

    def preprocess_sample(self, audio: list, max_length: int, sample_rate: int):
        audio = [librosa.util.fix_length(sample, size=max_length) for sample in audio]
        res = [self.make_spectrogram(sample, sample_rate=sample_rate) for sample in audio]
        return torch.cat(res)

    def batch_data(self, x, y, batch_size):
        for i in range(0, len(x), batch_size):
            yield x[i:i + batch_size], y[i:i + batch_size]


    def get_tokenizer(self):
        return self.tokenizer
