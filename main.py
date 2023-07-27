import os
from functools import partial

from SpeechRecognitionNMAProject.Model.net import SpeechRecognitionNet
from SpeechRecognitionNMAProject.Preprocessing.dataset import DataProcessor
from SpeechRecognitionNMAProject.Preprocessing.dataset import MyDataset, collate_fn
from torch.utils.data import DataLoader


def main():
    dp = DataProcessor(num_proc=4 if os.cpu_count() > 4 else os.cpu_count())
    train_size = 10_000

    train = dp.download(dataset_name="mozilla-foundation/common_voice_13_0", language_code="be", split="train",
                        token="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").select(range(train_size))

    test = dp.download(dataset_name="mozilla-foundation/common_voice_13_0", language_code="be", split="test",
                       token="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").select(range(round(train_size * 0.2)))

    validation = dp.download(dataset_name="mozilla-foundation/common_voice_13_0", language_code="be",
                             split="validation", token="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").select(
        range(round(train_size * 0.1)))
    print(train.num_rows)

    # train = train
    # test = test # 20% of train
    # validation = validation # 10% of train

    train, test, validation = dp.pre_process_text([train, test, validation])

    max_audio_length = 962496

    sampling_rate = train[0]["audio"]["sampling_rate"]

    batch_size = 16

    custom_collate_fn = partial(collate_fn, max_length=max_audio_length, sample_rate=sampling_rate,
                                dataset_processor=dp)

    train_loader = DataLoader(MyDataset(train), batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True)
    test_loader = DataLoader(MyDataset(test), batch_size=batch_size, collate_fn=custom_collate_fn)
    val_loader = DataLoader(MyDataset(validation), batch_size=batch_size, collate_fn=custom_collate_fn)

    del train
    del test
    del validation

    model = SpeechRecognitionNet(1, 768, dp, sample_rate=sampling_rate,
                                 max_text_length=193).to("cuda")

    model.train_(10, train_loader, None)

    # search_model(70, 5, train_loader,test_loader, sampling_rate,
    #                  max_audio_length,dp,193,batch_size=batch_size)


if __name__ == "__main__":
    main()
# 160 lin output
