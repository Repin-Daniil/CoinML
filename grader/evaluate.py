"""
Скрипт для оценки модели на тестовой выборке
"""
import argparse
import yaml
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import data_loader


def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """
    Рисует и сохраняет confusion matrix
    
    Args:
        cm: Confusion matrix
        save_path (str): Путь для сохранения
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(1, 6), yticklabels=range(1, 6))
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✓ Confusion matrix сохранена: {save_path}")


def evaluate_model(model, dataset, num_samples):
    """
    Оценивает модель на датасете
    
    Args:
        model: Keras модель
        dataset: tf.data.Dataset
        num_samples (int): Количество образцов
        
    Returns:
        tuple: (y_true, y_pred, y_pred_proba)
    """
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    print("\nВыполнение предсказаний...")
    
    for (images, labels) in dataset:
        predictions = model.predict(images, verbose=0)
        
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
        y_pred_proba.extend(predictions)
    
    return np.array(y_true), np.array(y_pred), np.array(y_pred_proba)


def parse_args():
    """Парсинг аргументов"""
    parser = argparse.ArgumentParser(
        description='Оценка модели на тестовой выборке'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Путь к обученной модели (.h5)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Путь к конфигурации'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./evaluation_results',
        help='Папка для сохранения результатов'
    )
    
    return parser.parse_args()


def main():
    """Главная функция оценки"""
    args = parse_args()
    
    print("\n" + "="*50)
    print("ОЦЕНКА МОДЕЛИ НА ВАЛИДАЦИОННОЙ ВЫБОРКЕ")
    print("="*50 + "\n")
    
    # Загрузка конфигурации
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Создание папки для результатов
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Загрузка модели
    print(f"✓ Загрузка модели: {args.model}")
    model = keras.models.load_model(args.model)
    
    # Загрузка данных
    print("\n" + "="*50)
    print("ЗАГРУЗКА ДАННЫХ")
    print("="*50)
    
    # Сканируем датасет
    samples = data_loader.scan_dataset(config['data']['path'])
    np.random.shuffle(samples)
    
    # Разделяем на train/val
    val_split = config['data']['val_split']
    split_idx = int(len(samples) * (1 - val_split))
    val_samples = samples[split_idx:]
    
    print(f"\nВалидационная выборка: {len(val_samples)} образцов")
    
    # Создаем датасет
    val_dataset = data_loader.create_dataset_pipeline(
        val_samples, config, is_training=False
    )
    
    # Оценка модели
    print("\n" + "="*50)
    print("ОЦЕНКА МОДЕЛИ")
    print("="*50)
    
    y_true, y_pred, y_pred_proba = evaluate_model(
        model, val_dataset, len(val_samples)
    )
    
    # Метрики
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ")
    print("="*50 + "\n")
    
    # Общая точность
    accuracy = np.mean(y_true == y_pred)
    print(f"Общая точность: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    
    # Classification Report
    print("Classification Report:")
    print("-" * 50)
    class_names = [f"Class {i}" for i in range(1, 6)]
    print(classification_report(
        y_true, y_pred, 
        target_names=class_names,
        digits=4
    ))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Сохранение confusion matrix
    cm_path = save_dir / 'confusion_matrix.png'
    plot_confusion_matrix(cm, str(cm_path))
    
    # Сохранение детальных результатов
    results_path = save_dir / 'evaluation_results.txt'
    with open(results_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ\n")
        f.write("="*50 + "\n\n")
        f.write(f"Модель: {args.model}\n")
        f.write(f"Общая точность: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names))
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
    
    print(f"\n✓ Результаты сохранены в: {save_dir}")
    
    # Статистика по уверенности предсказаний
    confidences = np.max(y_pred_proba, axis=1)
    print(f"\nСтатистика уверенности предсказаний:")
    print(f"  Средняя уверенность: {np.mean(confidences):.4f}")
    print(f"  Медианная уверенность: {np.median(confidences):.4f}")
    print(f"  Мин уверенность: {np.min(confidences):.4f}")
    print(f"  Макс уверенность: {np.max(confidences):.4f}")
    
    print("\n" + "="*50)
    print("ОЦЕНКА ЗАВЕРШЕНА")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()