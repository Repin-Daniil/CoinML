#!/bin/bash

# Скрипт для быстрого старта проекта Coin Grader

echo "=================================="
echo "🪙 COIN GRADER - БЫСТРЫЙ СТАРТ 🪙"
echo "=================================="
echo ""

# Проверка Python
echo "Проверка Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден. Установите Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✓ $PYTHON_VERSION найден"
echo ""

# Создание виртуального окружения
echo "Создание виртуального окружения..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Виртуальное окружение создано"
else
    echo "✓ Виртуальное окружение уже существует"
fi
echo ""

# Активация виртуального окружения
echo "Активация виртуального окружения..."
source venv/bin/activate
echo "✓ Виртуальное окружение активировано"
echo ""

# Установка зависимостей
echo "Установка зависимостей..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
echo "✓ Зависимости установлены"
echo ""

# Проверка TensorFlow и GPU
echo "Проверка TensorFlow и GPU..."
python3 -c "import tensorflow as tf; print(f'TensorFlow версия: {tf.__version__}'); gpus = tf.config.list_physical_devices('GPU'); print(f'Найдено GPU: {len(gpus)}') if gpus else print('⚠ GPU не найдены, используется CPU')"
echo ""

# Проверка структуры датасета
echo "Проверка структуры датасета..."
if [ -d "dataset" ]; then
    echo "✓ Папка dataset найдена"
    
    # Подсчет файлов в каждом классе
    for class in 1 2 3 4 5; do
        if [ -d "dataset/$class" ]; then
            count=$(ls -1 dataset/$class/*.jpg 2>/dev/null | wc -l)
            echo "  Класс $class: $count изображений"
        else
            echo "  ⚠ Папка dataset/$class не найдена"
        fi
    done
else
    echo "⚠ Папка dataset не найдена"
    echo "Создайте папку dataset со структурой:"
    echo "  dataset/"
    echo "    1/"
    echo "    2/"
    echo "    3/"
    echo "    4/"
    echo "    5/"
fi
echo ""

# Создание необходимых папок
echo "Создание рабочих папок..."
mkdir -p saved_models
mkdir -p logs
mkdir -p evaluation_results
echo "✓ Папки созданы"
echo ""

# Инициализация WandB (опционально)
echo "=================================="
echo "ИНИЦИАЛИЗАЦИЯ WANDB (опционально)"
echo "=================================="
echo ""
echo "Хотите настроить WandB для логирования? (y/n)"
read -r setup_wandb

if [ "$setup_wandb" = "y" ] || [ "$setup_wandb" = "Y" ]; then
    echo "Введите ваш WandB API ключ:"
    read -r wandb_key
    wandb login "$wandb_key"
    echo "✓ WandB настроен"
else
    echo "⊘ WandB пропущен (можно настроить позже командой: wandb login)"
fi
echo ""

# Итоговая информация
echo "=================================="
echo "✅ УСТАНОВКА ЗАВЕРШЕНА"
echo "=================================="
echo ""
echo "Доступные команды:"
echo ""
echo "1. Обучение модели:"
echo "   python train.py"
echo ""
echo "2. Обучение с кастомной конфигурацией:"
echo "   python train.py --config my_config.yaml"
echo ""
echo "3. Обучение с предобученными весами:"
echo "   python train.py --pretrained"
echo ""
echo "4. Предсказание для одной монеты:"
echo "   python predict.py --model saved_models/best_model.h5 \\"
echo "                     --obverse path/to/obverse.jpg \\"
echo "                     --reverse path/to/reverse.jpg"
echo ""
echo "5. Оценка модели на тестовой выборке:"
echo "   python evaluate.py --model saved_models/best_model.h5"
echo ""
echo "=================================="
echo "Удачи в обучении! 🚀"
echo "=================================="