# 📚 Примеры использования Coin Grader

## 🎓 Базовые примеры

### 1. Обучение с настройками по умолчанию
```bash
python train.py
```

### 2. Обучение с кастомной конфигурацией
```bash
python train.py --config experiments/config_mobilenet.yaml
```

### 3. Обучение с предобученными весами
```bash
python train.py --pretrained --config experiments/config_finetune.yaml
```

---

## 🔬 Эксперименты с разными архитектурами

### ResNet50 (глубокая архитектура)
```yaml
# config_resnet.yaml
model:
  backbone_name: "ResNet50"
  dropout_rate: 0.5
  dense_units: 256

training:
  batch_size: 32
  learning_rate: 0.001
```

```bash
python train.py --config config_resnet.yaml
```

### MobileNetV2 (легковесная)
```yaml
# config_mobilenet.yaml
model:
  backbone_name: "MobileNetV2"
  dropout_rate: 0.4
  dense_units: 128

training:
  batch_size: 64
  learning_rate: 0.0005
```

```bash
python train.py --config config_mobilenet.yaml
```

### SimpleCNN (с нуля)
```yaml
# config_simple.yaml
model:
  backbone_name: "SimpleCNN"
  dropout_rate: 0.3
  dense_units: 256

training:
  batch_size: 32
  learning_rate: 0.001
```

```bash
python train.py --config config_simple.yaml
```

---

## 🎨 Эксперименты с аугментацией

### Слабая аугментация
```yaml
# config_weak_aug.yaml
augmentation:
  flip_mode: "horizontal"
  rotation_factor: 0.05
  brightness_factor: 0.1
  zoom_factor: 0.05
```

### Сильная аугментация
```yaml
# config_strong_aug.yaml
augmentation:
  flip_mode: "horizontal_and_vertical"
  rotation_factor: 0.2
  brightness_factor: 0.3
  zoom_factor: 0.15
```

---

## 📊 Оптимизация гиперпараметров

### Высокий Learning Rate (для быстрой сходимости)
```yaml
training:
  learning_rate: 0.01
  lr_scheduler_factor: 0.5
  lr_scheduler_patience: 2
```

### Низкий Learning Rate (для fine-tuning)
```yaml
training:
  learning_rate: 0.0001
  lr_scheduler_factor: 0.1
  lr_scheduler_patience: 5
```

### Большой Batch Size (для A100)
```yaml
training:
  batch_size: 128  # Требует ~40GB VRAM
```

### Маленький Batch Size (для CPU/малого GPU)
```yaml
training:
  batch_size: 8
```

---

## 🔍 Предсказания

### Одна монета
```bash
python predict.py \
    --model saved_models/run_resnet50_best.h5 \
    --obverse dataset/test/coin001_obverse.jpg \
    --reverse dataset/test/coin001_reverse.jpg
```

### Batch предсказания (используя evaluate.py)
```bash
python evaluate.py \
    --model saved_models/run_resnet50_best.h5 \
    --config config.yaml
```

---

## 📈 WandB интеграция

### Запуск с кастомным именем в WandB
```yaml
# config.yaml
wandb:
  project_name: "CoinGrader-Experiments"
  run_name: "resnet50_strong_aug_lr001"
```

### Сравнение нескольких экспериментов
```bash
# Эксперимент 1: ResNet50
python train.py --config exp1_resnet.yaml

# Эксперимент 2: MobileNet
python train.py --config exp2_mobilenet.yaml

# Эксперимент 3: SimpleCNN
python train.py --config exp3_simple.yaml

# Результаты автоматически появятся в WandB для сравнения
```

---

## 🚀 Production примеры

### Обучение для production
```yaml
# config_production.yaml
data:
  val_split: 0.15  # Больше данных для обучения

training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.0001

model:
  backbone_name: "ResNet50"
  dropout_rate: 0.5

early_stopping_patience: 15  # Больше терпения
```

```bash
python train.py --config config_production.yaml
```

### Fine-tuning с предобученными весами
```yaml
# config_finetune.yaml
training:
  learning_rate: 0.00001  # Очень маленький LR
  epochs: 50

model:
  backbone_name: "ResNet50"
```

```bash
python train.py --pretrained --config config_finetune.yaml
```

---

## 🧪 Debugging и тестирование

### Быстрый тест на малом датасете
```yaml
# config_test.yaml
data:
  val_split: 0.5  # 50/50 split для быстрого теста

training:
  epochs: 5
  batch_size: 16
```

```bash
python train.py --config config_test.yaml
```

### Проверка overfitting
```yaml
# config_overfit_test.yaml
model:
  dropout_rate: 0.0  # Без dropout

training:
  epochs: 100
  
augmentation:
  flip_mode: "none"  # Без аугментации
```

Цель: Train accuracy должен стремиться к 100%, val может быть ниже

---

## 💡 Best Practices

### 1. Начните с базового эксперимента
```bash
# Используйте SimpleCNN для быстрого теста
python train.py --config config_simple.yaml
```

### 2. Градиентно увеличивайте сложность
```bash
# После SimpleCNN -> MobileNet -> ResNet50
python train.py --config config_mobilenet.yaml
python train.py --config config_resnet.yaml
```

### 3. Экспериментируйте с аугментацией
```bash
# Если overfitting - усильте аугментацию
# Если underfitting - ослабьте аугментацию
```

### 4. Используйте transfer learning если мало данных
```bash
python train.py --pretrained
```

### 5. Мониторьте через WandB
- Следите за разницей train_loss и val_loss
- Большая разница = overfitting
- Обе высокие = underfitting

---

## 🎯 Целевые метрики

### Минимальные требования
- Val Accuracy > 70%
- Train-Val gap < 15%

### Хорошие результаты
- Val Accuracy > 85%
- Train-Val gap < 10%

### Отличные результаты
- Val Accuracy > 90%
- Train-Val gap < 5%

---

## 📞 Troubleshooting

### Out of Memory
```yaml
training:
  batch_size: 8  # Уменьшите batch size
```

### Модель не обучается
```yaml
training:
  learning_rate: 0.01  # Увеличьте LR
```

### Overfitting
```yaml
model:
  dropout_rate: 0.7  # Увеличьте dropout

augmentation:
  rotation_factor: 0.2  # Усильте аугментацию
```

### Underfitting
```yaml
model:
  backbone_name: "ResNet50"  # Более мощная модель
  dense_units: 512  # Больше параметров

training:
  epochs: 100  # Больше эпох
```