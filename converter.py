
import json

finput = "./data/driver_mood_focus_100.jsonl"

fout = finput.split(".jsonl")[0] + ".json"

# Открываем старый файл
with open(finput, "r", encoding="utf-8") as fin:

    with open(fout, "w", encoding="utf-8") as fout:

        for line in fin:
            item = json.loads(line)

            prompt = item["prompt"]
            completion = item["completion"]

            # Преобразуем prompt в список пар ролей
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

                conversation.append({
                    "role": role,
                    "content": content
                })

            # Добавляем финальный completion как ещё одну реплику assistant
            # Убираем префикс Bot: если он есть
            final = completion.strip()
            if final.startswith("Bot:"):
                final = final[len("Bot:"):].strip()

            conversation.append({
                "role": "assistant",
                "content": final
            })

            # Формируем новый объект
            new_item = {
                "system": "You are a helpful assistant.",
                "conversation": conversation
            }

            # Записываем как новую строку JSONL
            fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")
