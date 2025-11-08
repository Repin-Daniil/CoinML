#!/usr/bin/env python3

import zipfile
import sys
from pathlib import Path

ZIP_FILE_NAME = "coin_dataset.zip" 
EXTRACT_DIR = Path("data")

def main():
    """
    Ищет ZIP_FILE_NAME и распаковывает его в EXTRACT_DIR.
    """
    zip_path = Path(ZIP_FILE_NAME)

    if not zip_path.exists():
        print(f"Ошибка: Архив '{ZIP_FILE_NAME}' не найден.")
        print("Пожалуйста, скачай его с Яндекс Диска и положи в корень проекта.")
        sys.exit(1) # Выход с кодом ошибки

    # 2. Проверяем, что это действительно ZIP (на всякий случай)
    if not zipfile.is_zipfile(zip_path):
        print(f"Ошибка: Файл '{ZIP_FILE_NAME}' не является корректным ZIP-архивом.")
        sys.exit(1)

    # 3. Создаем папку 'data', если ее нет
    try:
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Ошибка: Не удалось создать директорию '{EXTRACT_DIR}'. {e}")
        sys.exit(1)

    # 4. Распаковываем
    print(f"Начинаю распаковку '{ZIP_FILE_NAME}' в папку '{EXTRACT_DIR}'...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        
        print("="*30)
        print(f"✅ Успешно распаковано!")
        print(f"Структура данных должна быть готова в '{EXTRACT_DIR}'")
        
        print(f"Удаляю {zip_path}...")
        zip_path.unlink()

    except zipfile.BadZipFile:
        print(f"Ошибка: Архив '{ZIP_FILE_NAME}' поврежден.")
        sys.exit(1)
    except Exception as e:
        print(f"Произошла неизвестная ошибка при распаковке: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()