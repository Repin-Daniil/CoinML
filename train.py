import os
import json
import glob
import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models


from coin_processor import PipelineBuilder

CONFIG_PATH = "config_train.json"
INPUT_DATA_PATH = "data/input/"
MODEL_SAVE_PATH = "models/coin_model_v1.h5"

def get_label_from_filename(filepath):
    """Примитивная функция для получения метки из имени файла."""
    filename = os.path.basename(filepath).lower()
    if "broken" in filename:
        return 0 # "сломанная"
    elif "good" in filename:
        return 1 # "хорошая"
    # ... и т.д.
    return -1 # неизвестный

def main():
    print("--- Запуск процесса обучения ---")
    
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    # 1. Собираем пайплайн предобработки
    # Он точно такой же, как мы уже сделали
    builder = PipelineBuilder()
    preprocessing_pipeline = builder.build_from_config(config_data)

    # 2. Готовим датасет
    print("Подготовка датасета...")
    X_train_list = []
    y_train_list = []

    image_paths = glob.glob(os.path.join(INPUT_DATA_PATH, "*.jpg"))
    for img_path in image_paths:
        # Используем наш ОБЩИЙ пайплайн
        ctx = preprocessing_pipeline.process_image(img_path)
        
        if ctx.status == "success":
            label = get_label_from_filename(img_path)
            if label != -1:
                X_train_list.append(ctx.processed_image) # Нормализованный массив
                y_train_list.append(label)

    if not X_train_list:
        print("Ошибка: не удалось подготовить ни одного изображения. Проверь пути и конфиг.")
        return

    # Превращаем списки в NumPy-массивы, готовые для TF
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    
    print(f"Датасет готов. Форма X: {X_train.shape}, Форма Y: {y_train.shape}")

    # 3. Создаем и обучаем модель (простая CNN как пример)
    print("Сборка и обучение модели...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid') # Бинарная классификация (0 или 1)
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=10, validation_split=0.2)
    
    # 4. Сохраняем модель
    if not os.path.exists("models"):
        os.makedirs("models")
    model.save(MODEL_SAVE_PATH)
    print(f"Модель обучена и сохранена в: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()