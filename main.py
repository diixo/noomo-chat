
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict, load_dataset
import re

IGNORE_INDEX = -100

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

tokenizer.pad_token = tokenizer.eos_token

def check_special_tokens():
    print(f"eos_token: {tokenizer.eos_token}={tokenizer.eos_token_id}, decode={tokenizer.decode([tokenizer.eos_token_id])}")
    print(f"pad_token_id: {tokenizer.pad_token}={tokenizer.pad_token_id}, decode={tokenizer.decode([tokenizer.pad_token_id])}")
#check_special_tokens()

model = GPT2LMHeadModel.from_pretrained('gpt2')

num_added_toks = tokenizer.add_special_tokens({
    'additional_special_tokens': ["<|im_start|>", "<|im_end|>"],
    'pad_token': "<|pad|>"
    })
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

dataset = load_dataset("json",
        data_files="./data/test_debug.json",
        split="train")

print(f"dataset.sz={len(dataset)}")


"""
def build_prompt(system_message, conversation_history, user_message):
    prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n"

    for role, content in conversation_history:
        if role == "user":
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"

    prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n"
    return prompt
"""

def preprocess_fn(example):
    """
    Prompt for model:
    <|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    {user_message}<|im_end|>
    <|im_start|>assistant
    """
    input_ids = []
    labels = []

    # System block
    system_block = f"<|im_start|>system\n{example['system']}\n<|im_end|>\n"
    system_tokens = tokenizer(system_block, add_special_tokens=False).input_ids
    input_ids.extend(system_tokens)
    labels.extend([IGNORE_INDEX] * len(system_tokens))

    # Conversation blocks
    for turn in example["conversation"]:
        role = turn["role"]
        content = turn["content"]
        block = f"<|im_start|>{role}\n{content}\n<|im_end|>\n<|endoftext|>"

        block_tokens = tokenizer(block, add_special_tokens=False).input_ids
        input_ids.extend(block_tokens)

        if role == "assistant":
            labels.extend(block_tokens)
        else:
            labels.extend([IGNORE_INDEX] * len(block_tokens))

    return { "input_ids": input_ids, "labels": labels }


processed_dataset = dataset.map(preprocess_fn, remove_columns=dataset.column_names)


from dataclasses import dataclass
from typing import Any, Dict, List
import torch

@dataclass
class DataCollatorForCausalLMwithIgnorePad:
    tokenizer: Any
    ignore_index: int = IGNORE_INDEX

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.ignore_index
        )

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        #for ids in input_ids:
        #    print([tokenizer.decode([tid]) for tid in ids])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

collator = DataCollatorForCausalLMwithIgnorePad(tokenizer=tokenizer, ignore_index=IGNORE_INDEX)

training_args = TrainingArguments(
    output_dir="./train_products",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    learning_rate=3e-5,
    logging_steps=1,
    save_strategy="no",
    save_total_limit=1,
    lr_scheduler_type="constant",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    data_collator=collator,
)

trainer.train()
