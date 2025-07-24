
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict, load_dataset
from dataclasses import dataclass
from typing import Any, Dict, List
import torch
from main_train import check_special_tokens


IGNORE_INDEX = -100

##################################################################

class Chatbot_gpt2:

    def __init__(self):
        llm_model_name = "./gpt2-dialog"
        self.tokenizer = GPT2Tokenizer.from_pretrained(llm_model_name)
        self.model = GPT2LMHeadModel.from_pretrained(llm_model_name)
        check_special_tokens(self.tokenizer)


if __name__ == "__main__":

    system_prompt = "You are helpful car driver assistant."

    chat = Chatbot_gpt2()

    while True:
        user_message = input("user: ")

        if user_message.strip() == "exit":
            break

        assistant_reply, mood, conversation_history = chat.handle_user_message(
            system_prompt, conversation_history, user_message
        )
        print(f"Assistant: {assistant_reply}")
