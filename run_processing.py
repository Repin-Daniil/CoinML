import os
import json
import glob
import cv2  # <--- Добавлено для сохранения изображения
import numpy as np # <--- Добавлено для работы с типами (конвертация)

from coin_processor import PipelineBuilder

CONFIG_PATH = "config.json"
SAMPLE_IMAGE_PATH = "data/input/broken.jpg"
OUTPUT_DIR = "data/output" # <--- Добавлено

def main():
    """Главная функция запуска."""
    
    # Убедимся, что выходная директория существует
    os.makedirs(OUTPUT_DIR, exist_ok=True) # <--- Добавлено
    
    if not os.path.exists(CONFIG_PATH):
        print(f"Ошибка: не найден файл конфигурации {CONFIG_PATH}")
        return
    if not os.path.exists(SAMPLE_IMAGE_PATH):
        print(f"Ошибка: не найден файл изображения {SAMPLE_IMAGE_PATH}")
        return

    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        builder = PipelineBuilder()
        pipeline = builder.build_from_config(config_data)
        
        print(f"--- Запуск обработки для: {SAMPLE_IMAGE_PATH} ---")
        final_array = pipeline.process_image(SAMPLE_IMAGE_PATH).processed_image
        
        if final_array is not None:
            print("\n--- Результат (один файл) ---")
            print(f"  Тип данных: {final_array.dtype}")
            print(f"  Форма массива: {final_array.shape}")
            print(f"  Мин/Макс значения: {final_array.min()} / {final_array.max()}")
            
            # --- Начало: Логика сохранения изображения ---
            
            # Формируем путь для сохранения
            base_name = os.path.basename(SAMPLE_IMAGE_PATH)
            file_name, file_ext = os.path.splitext(base_name)
            OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, f"{file_name}_processed{file_ext}")

            try:
                image_to_save = final_array
                
                # cv2.imwrite ожидает данные в формате uint8 (0-255).
                # Если массив типа float (обычно 0.0-1.0), его нужно конвертировать.
                if 'float' in str(image_to_save.dtype) and image_to_save.max() <= 1.0 + 1e-6: # 1e-6 - допуск
                    print("  (Конвертация float 0-1.0 в uint8 0-255 для сохранения)")
                    image_to_save = (image_to_save * 255.0).astype(np.uint8)
                elif 'float' in str(image_to_save.dtype):
                     # Если это float, но не 0-1, просто приведем тип
                     print(f"  (ВНИМАНИЕ: Конвертация float с макс={image_to_save.max()} в uint8)")
                     image_to_save = image_to_save.astype(np.uint8)
                
                # Сохраняем массив как изображение
                cv2.imwrite(OUTPUT_IMAGE_PATH, image_to_save)
                print(f"\n--- Изображение сохранено ---")
                print(f"  Путь: {OUTPUT_IMAGE_PATH}")
                
            except Exception as e:
                print(f"\n--- Ошибка при сохранении изображения ---")
                print(f"  Ошибка: {e}")
            # --- Конец: Логика сохранения изображения ---
        
        # # --- 5. Пример: обработка всех .jpg в папке ---
        # print("\n--- Запуск пакетной обработки ---")
        # all_images = []
        # all_labels = [] # Тут будут твои метки (например, из имени файла)
        
        # image_paths = glob.glob("data/input/*.jpg")
        
        # for img_path in image_paths:
        #     result = pipeline.process_image(img_path)
        #     if result is not None:
        #         all_images.append(result)
        #         # (здесь ты должен как-то получить label для img_path)
        #         # all_labels.append(get_label_from_path(img_path)) 

        # # Теперь у тебя есть готовый датасет для TensorFlow
        # # X_train = np.array(all_images)
        # # y_train = np.array(all_labels)
        # # print(f"\nСобран датасет для TF: {X_train.shape}")


    except Exception as e:
        print(f"Произошла критическая ошибка в main: {e}")
        raise

if __name__ == "__main__":
    main()