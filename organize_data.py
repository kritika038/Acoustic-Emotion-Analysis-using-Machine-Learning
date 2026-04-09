import os
import shutil

source = "archive"
target = "data"

mapping = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry"
}

count = 0

for actor in os.listdir(source):
    actor_path = os.path.join(source, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            parts = file.split("-")

            if len(parts) < 3:
                continue

            emotion_code = parts[2]

            if emotion_code in mapping:
                emotion = mapping[emotion_code]

                os.makedirs(os.path.join(target, emotion), exist_ok=True)

                shutil.copy(
                    os.path.join(actor_path, file),
                    os.path.join(target, emotion, file)
                )

                count += 1

print(f"✅ Copied {count} files")