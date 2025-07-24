
from transformers import Trainer, TrainingArguments
from dataclasses import dataclass
import torch


@dataclass
class DummyCollator:
    tokenizer: any

    def __call__(self, features):
        print(f"🟢 Called! Batch: {features}")
        return {"dummy": torch.tensor([1])}

class DummyDataset:
    def __getitem__(self, idx):
        return {"input_ids": [1, 2, 3], "labels": [1, 2, 3]}
    def __len__(self):
        return 2

dummy_dataset = DummyDataset()

trainer = Trainer(
    model=None,
    args=TrainingArguments(
        output_dir="./out",
        per_device_train_batch_size=2,
        num_train_epochs=1,
    ),
    train_dataset=dummy_dataset,
    data_collator=DummyCollator(tokenizer=None),
)

trainer.train()
