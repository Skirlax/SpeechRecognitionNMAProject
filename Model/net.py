import json

import optuna
import torch
from evaluate import load
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from transformers import MBart50Tokenizer
from transformers import MBartForConditionalGeneration


class SpeechRecognitionNet(torch.nn.Module):
    def __init__(self, input_channels, bart_hidden, params: dict | None = None):  # hidden for bart large is 1024
        super(SpeechRecognitionNet, self).__init__()
        self.params = params
        self.conv_net = self.build_net(params, input_channels, bart_hidden)
        self.transformer = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50')
        self.tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50")
        self.loss_fn = torch.nn.CrossEntropyLoss()  # TODO: Replace with CTC loss if crossentropy doesn't work
        self.writer = SummaryWriter("Logs")
        self.wer = load("wer")
        self.optim = self.init_optim(params)

    def forward(self, x):
        linear_input_size = 0
        linear_input_size = self._add_linears_if_not_present(x, linear_input_size)
        x = self.conv_net(x).unsqueeze(1)
        bart_out = self.transformer(inputs_embeds=x)
        return bart_out, linear_input_size  # TODO: Remove if not necessary.

    def init_optim(self, params: dict | None):
        if params is None:
            return torch.optim.Adam([
                {'params': self.conv_net.parameters(), 'lr': 1e-4},
                {'params': self.transformer.parameters(), 'lr': 1e-4},
            ])
        return torch.optim.Adam([
            {'params': self.conv_net.parameters(), 'lr': params["lr1"]},
            {'params': self.transformer.parameters(), 'lr': params["lr2"]},
        ])

    def build_net(self, params: dict | None, input_channels: int, bart_hidden: int):
        if params is None:
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=4, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=32 * 4 * 4, out_features=32),  # change later
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=32, out_features=bart_hidden)

            )
        return self.build_cnn_model_part(params)

    def train_(self, epochs: int, x: Tensor, y: Tensor):
        for epoch in range(epochs):
            for inputs, labels in zip(x, y):
                self.optim.zero_grad()
                outputs, _ = self.forward(inputs)
                loss = self.loss_fn(outputs.logits, labels)
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
        return model

    def _add_linears_if_not_present(self, x: Tensor, linear_input_size: int):
        if self.conv_net[-1].__class__.__name__ == "Linear":
            return linear_input_size

        cnn_output = self.conv_net(x)

        num_el = torch.numel(cnn_output)
        linear_input_size = num_el
        self.conv_net.append(torch.nn.Linear(num_el, 256))
        self.conv_net.append(torch.nn.ReLU())
        self.conv_net.append(torch.nn.Linear(256, self.bart_hidden))
        return linear_input_size

    def predict(self, x):
        x = self.conv_net(x).unsqueeze(1)
        bert_out = self.transformer(inputs_embeds=x)
        return self.tokenizer.decode(bert_out.logits.argmax(dim=-1))

    def evaluate(self, eval_x, eval_y):
        wers = []
        wer_metric = load("wer")
        with torch.no_grad():
            if eval_x is not None and eval_y is not None:
                self.conv_net.eval()
                self.transformer.eval()
                for epoch, (inputs, labels) in enumerate(zip(eval_x, eval_y)):
                    outputs, _ = self.forward(inputs)
                    pred = self.tokenizer.decode(outputs.logits.argmax(dim=-1))
                    labels = self.tokenizer.decode(labels)
                    wer = wer_metric.compute(predictions=pred, references=labels)
                    wers.append(wer)
                    self.writer.add_scalar("WER/eval", wer, epoch)

        return sum(wers) / len(wers) if len(wers) > 0 else 0


def search_model(n_trials, trial_epochs, x: Tensor, y: Tensor, eval_x: Tensor, eval_y: Tensor):
    def objective(trial, epochs=trial_epochs):
        lr1 = trial.suggest_float("lr1", 1e-6, 1e-1, log=True)
        lr2 = trial.suggest_float("lr2", 1e-6, 1e-1, log=True)
        decay = trial.suggest_float("decay", 0.001, 0.1, log=True)
        n_layers = 4
        cnn_outputs = [trial.suggest_int(f"cnn_output_{i}", 1, 64, log=True) for i in range(n_layers)]
        cnn_inputs = cnn_outputs[:-1]
        cnn_inputs.insert(0, 1)
        cnn_kernels = [trial.suggest_int(f"cnn_kernel_{i}", 1, 4, log=True) for i in range(n_layers)]
        cnn_strides = [trial.suggest_int(f"cnn_stride_{i}", 1, 3, log=True) for i in range(n_layers)]
        cnn_paddings = [trial.suggest_int(f"cnn_padding_{i}", 1, 4, log=True) for i in range(n_layers)]
        max_pool_pos = trial.suggest_int("max_pool_pos", 1, n_layers - 1, log=True)
        max_pool_kernel_size = trial.suggest_int("max_pool_kernel_size", 1, 4, log=True)
        max_pool_stride = trial.suggest_int("max_pool_stride", 1, 3, log=True)
        max_pool_padding = trial.suggest_int("max_pool_padding", 1, 4, log=True)
        architecture_params = {
            "n_layers": n_layers,
            "cnn_outputs": cnn_outputs,
            "cnn_inputs": cnn_inputs,
            "cnn_kernels": cnn_kernels,
            "cnn_strides": cnn_strides,
            "cnn_paddings": cnn_paddings,
            "max_pool_pos": max_pool_pos,
            "max_pool_kernel_size": max_pool_kernel_size,
            "max_pool_stride": max_pool_stride,
            "max_pool_padding": max_pool_padding,
        }

        model = SpeechRecognitionNet(1, 1024, params=architecture_params)
        model.train()
        model.train_(epochs, x, y)
        model.eval()
        mean_wer = model.evaluate(eval_x, eval_y)

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
