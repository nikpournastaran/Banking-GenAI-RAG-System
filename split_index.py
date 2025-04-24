import os
import shutil

# Пути
INDEX_FILE = "index/index.faiss"
PARTS_DIR = "index/parts"

# Создаем директорию для частей, если не существует
os.makedirs(PARTS_DIR, exist_ok=True)

# Разделяем файл на части по 45 МБ
print(f"Разделение файла {INDEX_FILE} на части...")
with open(INDEX_FILE, "rb") as f:
    chunk_size = 45 * 1024 * 1024  # 45 МБ
    i = 0
    while True:
        chunk = f.read(chunk_size)
        if not chunk:
            break
        part_file = f"{PARTS_DIR}/part_{i:03d}.faiss"
        with open(part_file, "wb") as part:
            part.write(chunk)
        print(f"Создана часть {i+1}: {part_file} ({len(chunk) / (1024 * 1024):.2f} МБ)")
        i += 1

print(f"Индекс успешно разделен на {i} частей")