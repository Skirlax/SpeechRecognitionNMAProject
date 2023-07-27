import json
import os
from torch.utils.data import Dataset, DataLoader
import optuna
import torch
import torchaudio.transforms as T
from evaluate import load
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForSeq2SeqLM
from tqdm import tqdm
from SpeechRecognitionNMAProject.Preprocessing.dataset import DataProcessor


# import shift tokens right from modeling_bart.py


class CustomReshape(torch.nn.Module):
    def forward(self, x):
        batch_size, channels, frequency, time = x.shape
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, frequency)
        x = x.reshape(batch_size, time, -1)  # (batch, time, channels * frequency)
        # x = x.permute(0, 2, 1)

        return x


class SpeechRecognitionNet(torch.nn.Module):
    def __init__(self, input_channels, bart_hidden, dp: DataProcessor, sample_rate: int,
                 max_text_length: int,
                 params: dict | None = None):  # hidden for mt5-base is 768
        super(SpeechRecognitionNet, self).__init__()
        # TODO: !!! Gotta clean this code and make it more readable !!!
        self.params = params
        self.sample_rate = sample_rate
        self.bart_hidden = bart_hidden
        self.dp = dp
        self.max_text_length = max_text_length
        self.conv_net = self.build_net(params, input_channels, bart_hidden)
        self.transformer = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
        # self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-base",use_legacy=False)
        self.linear_1 = torch.nn.Linear(64, 256)
        self.linear_2 = torch.nn.Linear(256, bart_hidden)
        self.conv1d = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.loss_fn = torch.nn.CrossEntropyLoss()  # TODO: Replace with CTC loss if crossentropy doesn't work
        self.writer = SummaryWriter("Logs")
        self.wer = load("wer")
        self.optim = self.init_optim(params)

    def forward(self, x):
        last_two_dims = x.shape[-2:]
        x = x.reshape(-1,1,last_two_dims[0],last_two_dims[1])
        self._complete_net_if_needed(x)
        x = self.conv_net(x)
        x = x.transpose(1, 2)
        x = self.linear_1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear_2(x)
        x, mask = self.pad_to_max_sequence_and_make_mask(x, self.max_text_length)
        bart_out = self.transformer(inputs_embeds=x, attention_mask=mask, decoder_inputs_embeds=x)
        return bart_out

    def pad_to_max_sequence_and_make_mask(self, x: Tensor, max_length: int):
        batch_size, seq_length, feature_dim = x.shape

        if seq_length > max_length:
            x = x[:, :max_length, :]
            seq_length = max_length

        if seq_length < max_length:
            padding = torch.zeros(batch_size, max_length - seq_length, feature_dim, device=x.device)
            x = torch.cat([x, padding], dim=1)

        mask = torch.ones((batch_size, max_length), device=x.device)
        if seq_length < max_length:
            mask[:, seq_length:] = 0
        return x, mask

    def make_spectrogram(self, audio, n_fft=1024, win_length=None, hop_length=512,
                         n_mels=64) -> torch.Tensor:
        spect = T.MelSpectrogram(
            sample_rate=self.sample_rate,
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
        res = spect(torch.from_numpy(audio).to(torch.float32)).reshape(1, 64, -1)
        return res

    def init_optim(self, params: dict | None):
        if params is None:
            return torch.optim.Adam([
                {'params': self.conv_net.parameters(), 'lr': 1e-4},
                {'params': self.transformer.parameters(), 'lr': 1e-4},
            ])
        return torch.optim.Adam([
            {'params': self.conv_net.parameters(), 'lr': params["lr1"], 'weight_decay': params["decay1"]},
            {'params': self.transformer.parameters(), 'lr': params["lr2"], 'weight_decay': params["decay2"]},
        ])

    def build_net(self, params: dict | None, input_channels: int, bart_hidden: int):
        # if params is None:
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            CustomReshape(),
            torch.nn.Conv1d(in_channels=156, out_channels=64, kernel_size=3, stride=1, padding=1)

        )
        # return self.build_cnn_model_part(params)

    def train_(self, epochs: int, train_loader: DataLoader, max_length: int, batch_size=64):

        for epoch in range(epochs):
            for i, (inputs, labels) in enumerate(tqdm(train_loader,desc="Training")):

                # inputs = self.dp.preprocess_sample([sample["array"] for sample in inputs], max_length, self.sample_rate).unsqueeze(1).to("cuda")
                # labels = torch.stack([torch.tensor(label) for label in labels]).to("cuda")
                self.optim.zero_grad()
                outputs = self.forward(inputs)
                logits_ = outputs.logits
                logits_ = logits_.to("cuda")
                logits_ = logits_.view(-1, logits_.shape[-1])
                loss = self.loss_fn(logits_, labels.view(-1))
                self.writer.add_scalar("Loss/train", loss, epoch)
                loss.backward()
                self.optim.step()

    def build_cnn_model_part(self, params: dict) -> torch.nn.Sequential:
        model = torch.nn.Sequential()
        for inpt, output, kernel, padding, stride in zip(params["cnn_inputs"], params["cnn_outputs"],
                                                         params["cnn_kernels"], params["cnn_paddings"],
                                                         params["cnn_strides"]):
            model.append(torch.nn.Conv2d(in_channels=inpt, out_channels=output, kernel_size=kernel, stride=stride,
                                         padding=padding))
            model.append(torch.nn.ReLU())

        model.insert(params["max_pool_pos"],
                     torch.nn.MaxPool2d(kernel_size=params["max_pool_kernel_size"], stride=params["max_pool_stride"],
                                        padding=params["max_pool_padding"]))

        model.append(CustomReshape())

        return model

    def _complete_net_if_needed(self, x: Tensor):
        if self.linear_1 is not None:
            return

        cnn_output = self.conv_net(x)
        conv_input_size = cnn_output.shape[-2]

        self.conv_net.append(
            torch.nn.Conv1d(in_channels=conv_input_size, out_channels=64,
                            kernel_size=3, stride=1, padding=1))
        self.conv_net.append(torch.nn.ReLU())
        self.conv_net.append(torch.nn.MaxPool1d(kernel_size=1, stride=1))
        self.linear_1 = torch.nn.Linear(64, 256).to("cuda")
        self.linear_2 = torch.nn.Linear(256, self.bart_hidden).to("cuda")
        self.conv_net = self.conv_net.to("cuda")

    def predict(self, x):
        x = self.conv_net(x).unsqueeze(1)
        bert_out = self.transformer(inputs_embeds=x)
        return self.dp.get_tokenizer().decode(bert_out.logits.argmax(dim=-1))

    def evaluate(self, test_loader, max_length: int, batch_size=64):
        wers = []
        wer_metric = load("wer")
        # batched_data = self.dp.batch_data(eval_x, eval_y, batch_size)
        with torch.no_grad():
            self.conv_net.eval()
            self.transformer.eval()
            iteration = 0
            for i, (inputs, labels) in enumerate(tqdm(test_loader,desc="Evaluating")):

                # inputs = self.dp.preprocess_sample([sample["array"] for sample in inputs], max_length, self.sample_rate).unsqueeze(1).to("cuda")
                # labels = torch.stack([torch.tensor(label) for label in labels]).to("cuda")
                labels = [x[0] for x in labels]
                outputs = self.forward(inputs)
                pred = self.dp.get_tokenizer().batch_decode(outputs.logits.argmax(dim=-1).tolist())
                # print(labels)
                labels = self.dp.get_tokenizer().batch_decode(labels)
                wer = wer_metric.compute(predictions=pred, references=labels)
                random_sentence_idx = torch.randint(0, len(pred), (1,)).item()
                random_sentence_pred = pred[random_sentence_idx]
                random_sentence_label = labels[random_sentence_idx]
                self.writer.add_text("Random Batch Predicted Sentence sentence", random_sentence_pred,
                                     iteration)
                self.writer.add_text("Random Batch Label Sentence", random_sentence_label, iteration)
                self.writer.add_scalar("WER/eval", wer, iteration)
                wers.append(wer)
                iteration += 1

        return sum(wers) / len(wers) if len(wers) > 0 else 0

    def save_all(self, path):
        lr1 = self.optim.param_groups[0]["lr"]
        lr2 = self.optim.param_groups[1]["lr"]
        decay1 = self.optim.param_groups[0]["weight_decay"]
        decay2 = self.optim.param_groups[1]["weight_decay"]
        to_save = {
            "conv_net": self.conv_net.state_dict(),
            "transformer": self.transformer.state_dict(),
            "linear_1": self.linear_1.state_dict(),
            "linear_2": self.linear_2.state_dict(),
            "optim": self.optim.state_dict(),
            "lr1": lr1,
            "lr2": lr2,
            "decay1": decay1,
            "decay2": decay2
        }
        torch.save(to_save, path)

    def load_all(self, path):
        saved = torch.load(path)
        self.conv_net.load_state_dict(saved["conv_net"])
        self.transformer.load_state_dict(saved["transformer"])
        self.linear_1.load_state_dict(saved["linear_1"])
        self.linear_2.load_state_dict(saved["linear_2"])
        self.optim.load_state_dict(saved["optim"])
        self.optim.param_groups[0]["lr"] = saved["lr1"]
        self.optim.param_groups[1]["lr"] = saved["lr2"]
        self.optim.param_groups[0]["weight_decay"] = saved["decay1"]
        self.optim.param_groups[1]["weight_decay"] = saved["decay2"]


def wipe_logs_folder():
    for file in os.listdir("Logs"):
        os.remove(os.path.join("Logs", file))


def search_model(n_trials, trial_epochs, train_loader, test_loader, sample_rate, max_length, dp_, sequnce_max_length,
                 batch_size=32):
    def objective(trial, epochs=trial_epochs):
        lr1 = trial.suggest_float("lr1", 1e-6, 1e-1, log=True)
        lr2 = trial.suggest_float("lr2", 1e-6, 1e-1, log=True)
        decay1 = trial.suggest_float("decay1", 0.001, 0.1, log=True)
        decay2 = trial.suggest_float("decay2", 0.001, 0.1, log=True)
        # n_layers = 3
        # cnn_outputs = [trial.suggest_int(f"cnn_output_{i}", 16, 64, log=True) for i in range(n_layers)]
        # cnn_inputs = cnn_outputs[:-1]
        # cnn_inputs.insert(0, 1)
        # cnn_kernels = [trial.suggest_int(f"cnn_kernel_{i}", 1, 4) for i in range(n_layers)]
        # cnn_strides = [trial.suggest_int(f"cnn_stride_{i}", 1, 2) for i in range(n_layers)]
        # cnn_paddings = [trial.suggest_int(f"cnn_padding_{i}", 0, 3) for i in range(n_layers)]
        # max_pool_pos = trial.suggest_int("max_pool_pos", 0, n_layers - 1)
        # max_pool_kernel_size = trial.suggest_int("max_pool_kernel_size", 2, 3)
        # max_pool_stride = trial.suggest_int("max_pool_stride", 2, 3)
        # max_pool_padding = trial.suggest_int("max_pool_padding", 0, 1)
        # conv_1d_output_channels = trial.suggest_int("conv_1d_output_channels", 16, 128, log=True)

        architecture_params = {
            "lr1": lr1,
            "lr2": lr2,
            "decay1": decay1,
            "decay2": decay2,
            # "n_layers": n_layers,
            # "cnn_outputs": cnn_outputs,
            # "cnn_inputs": cnn_inputs,
            # "cnn_kernels": cnn_kernels,
            # "cnn_strides": cnn_strides,
            # "cnn_paddings": cnn_paddings,
            # "max_pool_pos": max_pool_pos,
            # "max_pool_kernel_size": max_pool_kernel_size,
            # "max_pool_stride": max_pool_stride,
            # "max_pool_padding": max_pool_padding,
            # "conv_1d_output_channels": conv_1d_output_channels

        }

        model = SpeechRecognitionNet(1, 768, dp_, params=architecture_params, sample_rate=sample_rate,
                                     max_text_length=sequnce_max_length).to("cuda")
        model.train()
        model.train_(epochs, train_loader, max_length, batch_size=batch_size)
        model.eval()
        mean_wer = model.evaluate(test_loader, max_length, batch_size=batch_size)

        trial.report(mean_wer, epochs)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        del model
        return mean_wer

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Best WER: ", study.best_value)
    print("Best params: ", study.best_params)
    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f)

    wipe_logs_folder()
