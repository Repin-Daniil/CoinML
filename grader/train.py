"""
Главный скрипт для запуска обучения модели классификации монет
"""
import argparse
import yaml
import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

# WandB для логирования
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbEvalCallback

# Наши модули
import utils
import data_loader
import model


def parse_args():
    """
    Парсинг аргументов командной строки
    """
    parser = argparse.ArgumentParser(
        description='Обучение CNN для классификации сохранности монет'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Путь к файлу конфигурации (default: config.yaml)'
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        help='Использовать предобученные веса ImageNet'
    )

    return parser.parse_args()


def load_config(config_path):
    """
    Загружает конфигурацию из YAML файла

    Args:
        config_path (str): Путь к config.yaml

    Returns:
        dict: Словарь с конфигурацией
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Файл конфигурации не найден: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"✓ Конфигурация загружена из: {config_path}")
    return config


from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

def setup_callbacks(config, model_save_path='best_model.h5'):
    """
    Создает список callbacks для обучения
    """
    callbacks_list = []

    # 1. WandB Metrics Logger — безопасный заменитель WandbCallback
    wandb_metrics = WandbMetricsLogger(log_freq='epoch')
    callbacks_list.append(wandb_metrics)

    # 2. WandB Model Checkpoint — сохранение весов на W&B
    wandb_checkpoint = WandbModelCheckpoint(filepath=model_save_path)
    callbacks_list.append(wandb_checkpoint)

    # 3. Локальный ModelCheckpoint — сохранение на диск
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    callbacks_list.append(checkpoint_callback)

    # 4. ReduceLROnPlateau — уменьшение learning rate
    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=config['training']['lr_scheduler_factor'],
        patience=config['training']['lr_scheduler_patience'],
        min_lr=config['training']['lr_scheduler_min_lr'],
        verbose=1
    )
    callbacks_list.append(reduce_lr_callback)

    # 5. EarlyStopping
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1
    )
    callbacks_list.append(early_stopping_callback)

    # 6. TensorBoard (по желанию)
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1,
        write_graph=False
    )
    callbacks_list.append(tensorboard_callback)

    print(f"✓ Настроено {len(callbacks_list)} callbacks")

    return callbacks_list



def main():
    """
    Главная функция обучения
    """
    # ========================================
    # 1. ПАРСИНГ АРГУМЕНТОВ
    # ========================================
    args = parse_args()

    print("\n" + "=" * 70)
    print(" " * 15 + "🪙 COIN GRADER - ОБУЧЕНИЕ МОДЕЛИ 🪙")
    print("=" * 70 + "\n")

    # ========================================
    # 2. ЗАГРУЗКА КОНФИГУРАЦИИ
    # ========================================
    config = load_config(args.config)

    # ========================================
    # 3. ИНИЦИАЛИЗАЦИЯ
    # ========================================
    # Фиксируем seed для воспроизводимости
    utils.set_random_seed(config['seed'])

    # Проверяем GPU
    utils.get_gpu_info()

    # Инициализируем WandB
    wandb.init(
        project=config['wandb']['project_name'],
        name=config['wandb']['run_name'],
        config=config,
        reinit=True
    )

    print(f"✓ WandB инициализирован: {config['wandb']['project_name']}")
    print(f"  Run name: {config['wandb']['run_name']}\n")

    # ========================================
    # 4. ЗАГРУЗКА ДАННЫХ
    # ========================================
    train_dataset, val_dataset = data_loader.get_datasets(config)

    # ========================================
    # 5. СОЗДАНИЕ МОДЕЛИ
    # ========================================
    if args.pretrained:
        coin_model = model.build_model_with_pretrained(config, weights='imagenet')
    else:
        coin_model = model.build_model(config)

    # Вывод архитектуры
    print("\n" + "=" * 50)
    print("АРХИТЕКТУРА МОДЕЛИ")
    print("=" * 50)
    coin_model.summary()

    # Подсчет параметров
    utils.print_model_info(coin_model)

    # ========================================
    # 6. CALLBACKS
    # ========================================
    # Создаем папку для сохранения моделей
    save_dir = Path("./saved_models")
    save_dir.mkdir(exist_ok=True)

    model_save_path = save_dir / f"{config['wandb']['run_name']}_best.h5"

    callbacks = setup_callbacks(config, str(model_save_path))

    # ========================================
    # 7. ОБУЧЕНИЕ
    # ========================================
    print("\n" + "=" * 50)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 50 + "\n")

    history = coin_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config['training']['epochs'],
        callbacks=callbacks,
        verbose=1
    )

    # ========================================
    # 8. ЗАВЕРШЕНИЕ
    # ========================================
    print("\n" + "=" * 50)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 50)

    # Финальные метрики
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    print(f"\nФинальные метрики:")
    print(f"  Train Loss:     {final_train_loss:.4f}")
    print(f"  Train Accuracy: {final_train_acc:.4f}")
    print(f"  Val Loss:       {final_val_loss:.4f}")
    print(f"  Val Accuracy:   {final_val_acc:.4f}")

    # Лучшие метрики
    best_val_acc = max(history.history['val_accuracy'])
    best_val_acc_epoch = history.history['val_accuracy'].index(best_val_acc) + 1

    print(f"\nЛучшая валидационная точность:")
    print(f"  Accuracy: {best_val_acc:.4f}")
    print(f"  Epoch: {best_val_acc_epoch}")

    print(f"\n✓ Лучшая модель сохранена: {model_save_path}")

    # Логируем финальные метрики в WandB
    wandb.log({
        "final_train_acc": final_train_acc,
        "final_val_acc": final_val_acc,
        "best_val_acc": best_val_acc,
        "best_val_acc_epoch": best_val_acc_epoch
    })

    # Сохраняем финальную модель
    final_model_path = save_dir / f"{config['wandb']['run_name']}_final.h5"
    coin_model.save(final_model_path)
    print(f"✓ Финальная модель сохранена: {final_model_path}")

    # Завершаем WandB
    wandb.finish()
    print("\n✓ WandB сессия завершена")

    print("\n" + "=" * 70)
    print(" " * 20 + "🎉 УСПЕШНОЕ ЗАВЕРШЕНИЕ 🎉")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
