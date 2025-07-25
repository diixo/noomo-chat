
import json


finput = "./data/intent_training_dialogues.jsonl"


def convert_to_json(finput: str):

    fout = finput.split(".jsonl")[0] + ".json"

    all_items = []

    with open(finput, "r", encoding="utf-8") as fin:

        for line in fin:
            item = json.loads(line)

            prompt = item["prompt"]
            completion = item["completion"]

            conversation = []
            for turn in prompt.split("\n"):
                turn = turn.strip()
                if not turn:
                    continue
                if turn.startswith("Bot:"):
                    role = "assistant"
                    content = turn[len("Bot:"):].strip()
                elif turn.startswith("User:"):
                    role = "user"
                    content = turn[len("User:"):].strip()
                else:
                    raise ValueError(f"Unknown line: {turn}")

                conversation.append({"role": role, "content": content})

            # Append final completion as assistant
            final = completion.strip()
            if final.startswith("Bot:"):
                final = final[len("Bot:"):].strip()
            conversation.append({"role": "assistant", "content": final})

            new_item = {
                "system": "You are helpful car driver assistant.",
                "conversation": conversation
            }

            all_items.append(new_item)


    with open(fout, "w", encoding="utf-8") as fout:
        json.dump(all_items, fout, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    
    #convert_to_json(finput)

    exit(0)

    with open(finput, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    with open("./data/intent_training_dialogues.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
