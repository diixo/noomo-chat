

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict, load_dataset
from dataclasses import dataclass
from typing import Any, Dict, List
import torch


MODEL_PATH = "./gpt2-dialog"
IGNORE_INDEX = -100

##################################################################

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def check_special_tokens(tokenizer: GPT2Tokenizer):
    print(f"eos_token: {tokenizer.eos_token}={tokenizer.eos_token_id}, decode={tokenizer.decode([tokenizer.eos_token_id])}")
    print(f"pad_token_id: {tokenizer.pad_token}={tokenizer.pad_token_id}, decode={tokenizer.decode([tokenizer.pad_token_id])}")
#check_special_tokens(tokenizer)


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


##################################################################

if __name__ == "__main__":

    num_added_toks = tokenizer.add_special_tokens({
        'additional_special_tokens': ["<|im_start|>", "<|im_end|>"],
        'pad_token': "<|pad|>"
        })
    tokenizer.pad_token = "<|pad|>"
    tokenizer.save_pretrained(MODEL_PATH)


    model = GPT2LMHeadModel.from_pretrained("gpt2")

    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    data_files=["./data/driver_mood_focus_100.json",
                "./data/driver_supermarket_100_dialogs.json",
                "./data/driver_work_100_varied_dialogs.json",
                "./data/intent_training_dialogues.json",
                "./data/psych_state_100_dialogs.json",
                "./data/test_debug.json",
                "./data/uncertainty_dialogs_100.json",
                "./data/uncertainty_film_final_100.json",
                "./data/uncertainty_film_more_100.json",
                ]
    dataset = load_dataset(path="json", data_files=data_files, split="train")

    print(f"dataset.sz={len(dataset)}")

    processed_dataset = dataset.map(preprocess_fn, remove_columns=dataset.column_names)

    collator = DataCollatorForCausalLMwithIgnorePad(tokenizer=tokenizer, ignore_index=IGNORE_INDEX)

    training_args = TrainingArguments(
        output_dir="./train_products",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        learning_rate=3e-5,
        logging_steps=5,
        save_strategy="no",
        save_total_limit=1,
        lr_scheduler_type="constant",
        #prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=collator,
    )

    trainer.train()

    #trainer.model.save_pretrained(MODEL_PATH)
