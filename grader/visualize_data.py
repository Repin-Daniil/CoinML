"""
Скрипт для визуализации примеров из датасета
"""
import argparse
import yaml
import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import data_loader


def visualize_samples(samples, num_samples=12, save_path='dataset_samples.png'):
    """
    Визуализирует случайные образцы из датасета
    
    Args:
        samples (list): Список образцов (obverse_path, reverse_path, label)
        num_samples (int): Количество образцов для визуализации
        save_path (str): Путь для сохранения визуализации
    """
    # Выбираем случайные образцы
    random_samples = random.sample(samples, min(num_samples, len(samples)))
    
    # Создаем фигуру
    rows = num_samples // 2
    fig, axes = plt.subplots(rows, 4, figsize=(16, rows * 4))
    fig.suptitle('Примеры из датасета (Аверс | Реверс)', fontsize=16, y=0.995)
    
    for idx, (obverse_path, reverse_path, label) in enumerate(random_samples):
        row = idx // 2
        col = (idx % 2) * 2
        
        # Загружаем изображения
        img_obverse = Image.open(obverse_path)
        img_reverse = Image.open(reverse_path)
        
        # Отображаем аверс
        axes[row, col].imshow(img_obverse)
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Класс {label+1} - Аверс', fontsize=10)
        
        # Отображаем реверс
        axes[row, col+1].imshow(img_reverse)
        axes[row, col+1].axis('off')
        axes[row, col+1].set_title(f'Класс {label+1} - Реверс', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Визуализация сохранена: {save_path}")
    plt.close()


def plot_class_distribution(samples, save_path='class_distribution.png'):
    """
    Рисует распределение классов в датасете
    
    Args:
        samples (list): Список образцов
        save_path (str): Путь для сохранения
    """
    # Подсчитываем образцы по классам
    class_counts = {}
    for _, _, label in samples:
        class_counts[label+1] = class_counts.get(label+1, 0) + 1
    
    # Создаем график
    plt.figure(figsize=(10, 6))
    
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    colors = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#27ae60']
    
    bars = plt.bar(classes, counts, color=colors, edgecolor='black', linewidth=1.5)
    
    # Добавляем значения на столбцы
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.xlabel('Класс сохранности', fontsize=12)
    plt.ylabel('Количество образцов', fontsize=12)
    plt.title('Распределение образцов по классам', fontsize=14, fontweight='bold')
    plt.xticks(classes)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Легенда
    legend_labels = [
        'Класс 1: Плохое',
        'Класс 2: Удовлетворительное',
        'Класс 3: Среднее',
        'Класс 4: Хорошее',
        'Класс 5: Отличное'
    ]
    plt.legend(bars, legend_labels, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Распределение классов сохранено: {save_path}")
    plt.close()


def analyze_image_sizes(samples, num_samples=100):
    """
    Анализирует размеры изображений в датасете
    
    Args:
        samples (list): Список образцов
        num_samples (int): Сколько образцов проанализировать
    """
    print("\n" + "="*50)
    print("АНАЛИЗ РАЗМЕРОВ ИЗОБРАЖЕНИЙ")
    print("="*50)
    
    widths = []
    heights = []
    
    # Берем случайную выборку
    random_samples = random.sample(samples, min(num_samples, len(samples)))
    
    for obverse_path, reverse_path, _ in random_samples:
        # Аверс
        img = Image.open(obverse_path)
        widths.append(img.width)
        heights.append(img.height)
        
        # Реверс
        img = Image.open(reverse_path)
        widths.append(img.width)
        heights.append(img.height)
    
    widths = np.array(widths)
    heights = np.array(heights)
    
    print(f"\nПроанализировано {len(widths)} изображений:")
    print(f"\nШирина:")
    print(f"  Мин: {widths.min()} px")
    print(f"  Макс: {widths.max()} px")
    print(f"  Средняя: {widths.mean():.1f} px")
    print(f"  Медиана: {np.median(widths):.1f} px")
    
    print(f"\nВысота:")
    print(f"  Мин: {heights.min()} px")
    print(f"  Макс: {heights.max()} px")
    print(f"  Средняя: {heights.mean():.1f} px")
    print(f"  Медиана: {np.median(heights):.1f} px")
    
    # Соотношение сторон
    aspects = widths / heights
    print(f"\nСоотношение сторон:")
    print(f"  Мин: {aspects.min():.2f}")
    print(f"  Макс: {aspects.max():.2f}")
    print(f"  Среднее: {aspects.mean():.2f}")
    
    print("="*50 + "\n")


def parse_args():
    """Парсинг аргументов"""
    parser = argparse.ArgumentParser(
        description='Визуализация и анализ датасета'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Путь к конфигурации'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=12,
        help='Количество образцов для визуализации'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./visualizations',
        help='Папка для сохранения визуализаций'
    )
    
    return parser.parse_args()


def main():
    """Главная функция"""
    args = parse_args()
    
    print("\n" + "="*50)
    print("ВИЗУАЛИЗАЦИЯ И АНАЛИЗ ДАТАСЕТА")
    print("="*50 + "\n")
    
    # Загрузка конфигурации
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Создание папки для сохранения
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Сканируем датасет
    print("Сканирование датасета...")
    samples = data_loader.scan_dataset(config['data']['path'])
    
    if len(samples) == 0:
        print("❌ Датасет пуст!")
        return
    
    print(f"✓ Найдено {len(samples)} пар изображений\n")
    
    # Анализ размеров
    analyze_image_sizes(samples, num_samples=100)
    
    # Распределение по классам
    print("Создание графика распределения классов...")
    plot_class_distribution(
        samples, 
        save_path=str(save_dir / 'class_distribution.png')
    )
    
    # Визуализация образцов
    print(f"Создание визуализации {args.num_samples} образцов...")
    visualize_samples(
        samples,
        num_samples=args.num_samples,
        save_path=str(save_dir / 'dataset_samples.png')
    )
    
    # Статистика по каждому классу
    print("\n" + "="*50)
    print("ДЕТАЛЬНАЯ СТАТИСТИКА ПО КЛАССАМ")
    print("="*50)
    
    for class_label in range(5):
        class_samples = [s for s in samples if s[2] == class_label]
        percentage = (len(class_samples) / len(samples)) * 100
        print(f"\nКласс {class_label+1}:")
        print(f"  Образцов: {len(class_samples)}")
        print(f"  Процент: {percentage:.1f}%")
    
    print("\n" + "="*50)
    print("ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА")
    print("="*50)
    print(f"\nРезультаты сохранены в: {save_dir}")
    print(f"  - class_distribution.png")
    print(f"  - dataset_samples.png")
    print("\n")


if __name__ == "__main__":
    main()