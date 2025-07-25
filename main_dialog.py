
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from main_train import check_special_tokens


class Chatbot_gpt2:

    def __init__(self):
        llm_model_name = "./gpt2-dialog"
        self.tokenizer = GPT2Tokenizer.from_pretrained(llm_model_name)
        self.model = GPT2LMHeadModel.from_pretrained(llm_model_name)
        check_special_tokens(self.tokenizer)


    def build_prompt(self, system_message, conversation_history, user_message):
        # Create prompt for model in format:
        """
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        """
        prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n"

        for role, content in conversation_history:
            if role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"

        prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n"
        return prompt


    def generate_llm_response(self, prompt, max_new_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        if "<|im_start|>assistant" in text:
            text = text.split("<|im_start|>assistant")[-1].strip()
            text = text.split("<|im_end|>")[0].strip()
        return text


    def handle_user_message(self, system_prompt, conversation_history, user_message):
        # Build prompt
        prompt = self.build_prompt(system_prompt, conversation_history, user_message)

        # LLM response
        assistant_reply = self.generate_llm_response(prompt)

        # Update history
        conversation_history.append(("user", user_message))
        conversation_history.append(("assistant", assistant_reply))
        # TODO: return real mood-level
        return assistant_reply, conversation_history, "mood-level"


if __name__ == "__main__":

    assistant_prompt = "You are helpful car driver assistant."

    conversation_history = []

    chat = Chatbot_gpt2()

    while True:
        user_message = input("user: ")

        if user_message.strip().lower() == "exit":
            break

        assistant_reply, conversation_history = chat.handle_user_message(assistant_prompt, conversation_history, user_message)

        print(f"Assistant: {assistant_reply}")
