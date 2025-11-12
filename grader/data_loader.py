"""
Модуль для загрузки и предобработки данных монет
"""
import os
from pathlib import Path
import tensorflow as tf
import numpy as np


from pathlib import Path

def scan_dataset(data_path, banned_file="banned.txt"):
    """
    Сканирует папку с данными и создает список пар изображений с метками.
    Игнорирует монеты, ID которых указаны в banned.txt.

    Args:
        data_path (str): Путь к корневой папке с данными
        banned_file (str): Путь к файлу со списком забаненных ID (по одному в строке)

    Returns:
        list: Список кортежей (obverse_path, reverse_path, label_index)
    """
    data_path = Path(data_path)
    samples = []

    # --- Читаем список забаненных coin_id ---
    banned_path = Path(banned_file)
    banned_ids = set()
    if banned_path.exists():
        with open(banned_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    banned_ids.add(line)
        print(f"🛑 Загружено {len(banned_ids)} забаненных ID из {banned_file}")
    else:
        print(f"⚠️ Файл {banned_file} не найден, фильтрация не выполняется")

    # --- Обход классов ---
    for class_folder in sorted(data_path.iterdir()):
        if not class_folder.is_dir():
            continue

        try:
            label = int(class_folder.name)
            if label < 1 or label > 5:
                continue
        except ValueError:
            continue

        label_index = label - 1
        all_files = {f.stem: f for f in class_folder.glob("*.jpg")}

        coin_ids = set()
        for filename in all_files.keys():
            if '_obverse' in filename:
                coin_id = filename.replace('_obverse', '')
                coin_ids.add(coin_id)
            elif '_reverse' in filename:
                coin_id = filename.replace('_reverse', '')
                coin_ids.add(coin_id)

        for coin_id in coin_ids:
            # --- Проверка на бан ---
            if coin_id in banned_ids:
                print(f"🚫 Пропущен забаненный coin_id: {coin_id}")
                continue

            obverse_name = f"{coin_id}_obverse"
            reverse_name = f"{coin_id}_reverse"

            if obverse_name in all_files and reverse_name in all_files:
                obverse_path = str(all_files[obverse_name])
                reverse_path = str(all_files[reverse_name])
                samples.append((obverse_path, reverse_path, label_index))

    print(f"✓ Найдено {len(samples)} пар изображений монет после фильтрации")

    # --- Статистика по классам ---
    class_counts = {}
    for _, _, label in samples:
        class_counts[label] = class_counts.get(label, 0) + 1

    print("Распределение по классам:")
    for label in sorted(class_counts.keys()):
        print(f"  Класс {label+1}: {class_counts[label]} образцов")

    return samples



def create_augmentation_layer(config):
    """
    Создает Sequential слой с аугментациями на основе конфига
    
    Args:
        config (dict): Конфигурация с параметрами аугментации
        
    Returns:
        tf.keras.Sequential: Слой с аугментациями
    """
    aug_config = config['augmentation']
    
    layers = []
    
    # Флипы
    if aug_config['flip_mode'] == 'horizontal':
        layers.append(tf.keras.layers.RandomFlip("horizontal"))
    elif aug_config['flip_mode'] == 'vertical':
        layers.append(tf.keras.layers.RandomFlip("vertical"))
    elif aug_config['flip_mode'] == 'horizontal_and_vertical':
        layers.append(tf.keras.layers.RandomFlip("horizontal_and_vertical"))
    
    # Поворот
    if aug_config['rotation_factor'] > 0:
        layers.append(tf.keras.layers.RandomRotation(aug_config['rotation_factor']))
    
    # Zoom
    if aug_config.get('zoom_factor', 0) > 0:
        layers.append(tf.keras.layers.RandomZoom(aug_config['zoom_factor']))
    
    # Яркость
    if aug_config['brightness_factor'] > 0:
        layers.append(tf.keras.layers.RandomBrightness(aug_config['brightness_factor']))
    
    return tf.keras.Sequential(layers, name='augmentation')


def load_and_preprocess_image(path, image_size):
    """
    Загружает и нормализует одно изображение
    
    Args:
        path: Путь к изображению
        image_size (int): Размер изображения
        
    Returns:
        tf.Tensor: Нормализованное изображение [H, W, 3]
    """
    # Читаем файл
    img = tf.io.read_file(path)
    # Декодируем JPEG
    img = tf.image.decode_jpeg(img, channels=3)
    # Изменяем размер
    img = tf.image.resize(img, [image_size, image_size])
    # Нормализуем к [0, 1]
    img = img / 255.0
    
    return img


def create_dataset_pipeline(samples, config, is_training=True):
    """
    Создает tf.data.Dataset пайплайн
    
    Args:
        samples (list): Список кортежей (obverse_path, reverse_path, label)
        config (dict): Конфигурация
        is_training (bool): Флаг обучающей выборки (для аугментации)
        
    Returns:
        tf.data.Dataset: Готовый датасет
    """
    # Разделяем на списки
    obverse_paths = [s[0] for s in samples]
    reverse_paths = [s[1] for s in samples]
    labels = [s[2] for s in samples]
    
    # Создаем датасет из путей
    dataset = tf.data.Dataset.from_tensor_slices(
        (obverse_paths, reverse_paths, labels)
    )
    
    image_size = config['data']['image_size']
    
    # Функция загрузки и предобработки пары изображений
    def load_pair(obverse_path, reverse_path, label):
        img_a = load_and_preprocess_image(obverse_path, image_size)
        img_b = load_and_preprocess_image(reverse_path, image_size)
        return (img_a, img_b), label
    
    # Применяем загрузку
    dataset = dataset.map(
        load_pair,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Кэшируем (т.к. данные поместятся в RAM)
    dataset = dataset.cache()
    
    # Shuffle только для обучающей выборки
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(samples))
    
    # Батчирование
    batch_size = config['training']['batch_size']
    dataset = dataset.batch(batch_size)
    
    # Аугментация (применяется к батчу, только для обучения)
    if is_training:
        augmentation = create_augmentation_layer(config)
        
        def apply_augmentation(images, labels):
            img_a, img_b = images
            img_a = augmentation(img_a, training=True)
            img_b = augmentation(img_b, training=True)
            return (img_a, img_b), labels
        
        dataset = dataset.map(
            apply_augmentation,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Prefetch для оптимизации
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def get_datasets(config):
    """
    Главная функция для получения обучающей и валидационной выборок
    
    Args:
        config (dict): Конфигурация проекта
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    print("\n" + "="*50)
    print("ЗАГРУЗКА ДАННЫХ")
    print("="*50)
    
    # Сканируем датасет
    samples = scan_dataset(config['data']['path'])
    
    if len(samples) == 0:
        raise ValueError("❌ Не найдено ни одной пары изображений!")
    
    # Перемешиваем
    np.random.shuffle(samples)
    
    # Разделяем на train/val
    val_split = config['data']['val_split']
    split_idx = int(len(samples) * (1 - val_split))
    
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    print(f"\nРазделение данных:")
    print(f"  Обучающая выборка: {len(train_samples)} образцов")
    print(f"  Валидационная выборка: {len(val_samples)} образцов")
    
    # Создаем пайплайны
    train_dataset = create_dataset_pipeline(train_samples, config, is_training=True)
    val_dataset = create_dataset_pipeline(val_samples, config, is_training=False)
    
    print(f"\n✓ Датасеты успешно созданы")
    print("="*50 + "\n")
    
    return train_dataset, val_dataset