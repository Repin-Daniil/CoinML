"""
Скрипт для предсказания класса сохранности монеты
"""
import argparse
import yaml
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_and_preprocess_image(image_path, image_size):
    """
    Загружает и предобрабатывает одно изображение
    
    Args:
        image_path (str): Путь к изображению
        image_size (int): Размер для ресайза
        
    Returns:
        np.ndarray: Предобработанное изображение
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [image_size, image_size])
    img = img / 255.0
    return img.numpy()


def predict_coin(model, obverse_path, reverse_path, config):
    """
    Делает предсказание для одной монеты
    
    Args:
        model: Загруженная Keras модель
        obverse_path (str): Путь к изображению аверса
        reverse_path (str): Путь к изображению реверса
        config (dict): Конфигурация
        
    Returns:
        tuple: (predicted_class, confidence, probabilities)
    """
    image_size = config['data']['image_size']
    
    # Загружаем изображения
    img_obverse = load_and_preprocess_image(obverse_path, image_size)
    img_reverse = load_and_preprocess_image(reverse_path, image_size)
    
    # Добавляем batch dimension
    img_obverse = np.expand_dims(img_obverse, axis=0)
    img_reverse = np.expand_dims(img_reverse, axis=0)
    
    # Предсказание
    predictions = model.predict([img_obverse, img_reverse], verbose=0)
    
    # Результаты
    predicted_class = np.argmax(predictions[0]) + 1  # +1 т.к. классы 1-5
    confidence = np.max(predictions[0])
    probabilities = predictions[0]
    
    return predicted_class, confidence, probabilities


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='Предсказание класса сохранности монеты'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Путь к обученной модели (.h5)'
    )
    parser.add_argument(
        '--obverse',
        type=str,
        required=True,
        help='Путь к изображению аверса'
    )
    parser.add_argument(
        '--reverse',
        type=str,
        required=True,
        help='Путь к изображению реверса'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Путь к файлу конфигурации'
    )
    
    return parser.parse_args()


def main():
    """Главная функция для предсказания"""
    args = parse_args()
    
    print("\n" + "="*50)
    print("ПРЕДСКАЗАНИЕ КЛАССА СОХРАННОСТИ МОНЕТЫ")
    print("="*50 + "\n")
    
    # Проверка существования файлов
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Модель не найдена: {args.model}")
    if not Path(args.obverse).exists():
        raise FileNotFoundError(f"Изображение не найдено: {args.obverse}")
    if not Path(args.reverse).exists():
        raise FileNotFoundError(f"Изображение не найдено: {args.reverse}")
    
    # Загрузка конфигурации
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Загрузка модели: {args.model}")
    model = keras.models.load_model(args.model)
    print(f"✓ Модель загружена\n")
    
    # Предсказание
    print("Обработка изображений...")
    predicted_class, confidence, probabilities = predict_coin(
        model, args.obverse, args.reverse, config
    )
    
    # Вывод результатов
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ")
    print("="*50)
    print(f"\nПредсказанный класс: {predicted_class}")
    print(f"Уверенность: {confidence:.2%}\n")
    
    print("Вероятности по классам:")
    for i, prob in enumerate(probabilities):
        class_num = i + 1
        bar = "█" * int(prob * 30)
        print(f"  Класс {class_num}: {prob:.2%} {bar}")
    
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()