
### input file-format:
"""
User: Hello.
Bot: Hello-hello.
#####
User: Hello.
Bot: Hello-hello.
"""

import json


def dialogs_txt_to_json(input_file):

    output_file = input_file.split(".txt")[0] + ".json"

    all_conversations = []
    current_conversation = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("###"):
                # New dialog — save previous, if it was
                if current_conversation:
                    all_conversations.append({
                        "system": "You are helpful car driver assistant.",
                        "conversation": current_conversation
                    })
                    current_conversation = []
                continue

            if line.startswith("User:"):
                role = "user"
                content = line[len("User:"):].strip()
            elif line.startswith("Bot:"):
                role = "assistant"
                content = line[len("Bot:"):].strip()
            else:
                continue

            current_conversation.append({
                "role": role,
                "content": content
            })

    if current_conversation:
        all_conversations.append({
            "system": "You are helpful car driver assistant.",
            "conversation": current_conversation
        })

    # save final
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)

    print(f"JSON was saved into: {output_file}")


dialogs_txt_to_json(input_file="data/dialogs.txt")
