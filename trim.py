import os
import json

from pathlib import Path
import cv2
import numpy as np

from coin_processor import PipelineBuilder

CONFIG_PATH = "coin_processor/config.json"
INPUT_DIR = "data/input"
OUTPUT_DIR = "data/cropped"

def main():
    """Главная функция запуска."""
    
    if not os.path.exists(CONFIG_PATH):
        print(f"Ошибка: не найден файл конфигурации {CONFIG_PATH}")
        return

    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        builder = PipelineBuilder()
        pipeline = builder.build_from_config(config_data)
        
    except Exception as e:
        print(f"Произошла критическая ошибка при загрузке конфига/пайплайна: {e}")
        raise

    input_dir_path = Path(INPUT_DIR)
    output_dir_path = Path(OUTPUT_DIR)

    image_extensions = [".jpg", ".jpeg", ".png"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(input_dir_path.rglob(f"*{ext}"))
        
    if not image_paths:
        print(f"Не найдено изображений (jpg, jpeg, png) в {INPUT_DIR}")
        return

    print(f"--- Найдено {len(image_paths)} изображений. Начинаю пакетную обработку... ---")

    processed_count = 0
    failed_count = 0

    for input_path in image_paths:
        try:

            input_path_str = str(input_path) 
            print(f"\nОбработка: {input_path_str}")


            result = pipeline.process_image(input_path_str)
            
            if result is None or result.processed_image is None:
                print("  Результат обработки: None (файл пропущен)")
                failed_count += 1
                continue
                
            final_array = result.processed_image


            relative_path = input_path.relative_to(input_dir_path)
            
            stem = relative_path.stem 
            suffix = relative_path.suffix 
            new_filename = f"{stem}_cropped{suffix}"
            
            output_parent_dir = output_dir_path / relative_path.parent
            
            output_path = output_parent_dir / new_filename
            output_parent_dir.mkdir(parents=True, exist_ok=True)

            image_to_save = final_array
            if 'float' in str(image_to_save.dtype) and image_to_save.max() <= 1.0 + 1e-6:
                image_to_save = (image_to_save * 255.0).astype(np.uint8)
            elif 'float' in str(image_to_save.dtype):
                 image_to_save = image_to_save.astype(np.uint8)
            
            cv2.imwrite(str(output_path), image_to_save)
            print(f"  Сохранено: {output_path}")
            processed_count += 1

        except Exception as e:
            print(f"  !!! Ошибка при обработке файла {input_path_str}: {e}")
            failed_count += 1


    print("\n--- Пакетная обработка завершена ---")
    print(f"  Успешно обработано: {processed_count}")
    print(f"  Ошибки / Пропущено: {failed_count}")


if __name__ == "__main__":
    main()