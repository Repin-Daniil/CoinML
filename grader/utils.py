"""
Вспомогательные функции для воспроизводимости и утилит
"""
import os
import random
import numpy as np
import tensorflow as tf


def set_random_seed(seed=42):
    """
    Фиксирует seed для воспроизводимости результатов.
    
    Args:
        seed (int): Значение seed
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Для детерминированного поведения на GPU (может замедлить обучение)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    print(f"✓ Random seed установлен: {seed}")


def get_gpu_info():
    """
    Выводит информацию о доступных GPU
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n{'='*50}")
        print(f"Найдено GPU: {len(gpus)}")
        for gpu in gpus:
            print(f"  - {gpu.name}")
        print(f"{'='*50}\n")
        
        # Включаем memory growth чтобы TF не забирал всю память сразу
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Ошибка при настройке memory growth: {e}")
    else:
        print("\n⚠ GPU не найдены, используется CPU")


def count_parameters(model):
    """
    Подсчитывает количество параметров в модели
    
    Args:
        model: Keras модель
        
    Returns:
        dict: Словарь с информацией о параметрах
    """
    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_count = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    
    return {
        'trainable': trainable_count,
        'non_trainable': non_trainable_count,
        'total': trainable_count + non_trainable_count
    }


def print_model_info(model):
    """
    Выводит детальную информацию о модели
    
    Args:
        model: Keras модель
    """
    params = count_parameters(model)
    
    print(f"\n{'='*50}")
    print("ИНФОРМАЦИЯ О МОДЕЛИ")
    print(f"{'='*50}")
    print(f"Обучаемых параметров:     {params['trainable']:,}")
    print(f"Необучаемых параметров:   {params['non_trainable']:,}")
    print(f"Всего параметров:         {params['total']:,}")
    print(f"{'='*50}\n")