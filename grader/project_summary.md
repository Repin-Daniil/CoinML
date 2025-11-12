# 🪙 Coin Grader - Краткое описание проекта

## 📋 Общая информация

**Название:** Coin Grader - CNN для классификации сохранности монет  
**Задача:** Классификация монет по 5 классам сохранности (1-плохое, 5-отличное)  
**Подход:** Двухвходовая CNN с shared weights для аверса и реверса  
**Фреймворк:** TensorFlow 2.x + Keras  
**Мониторинг:** Weights & Biases (WandB)

---

## 🎯 Ключевые особенности

1. **Двухвходовая архитектура**
   - Отдельные входы для аверса и реверса монеты
   - Один общий backbone (shared weights) для извлечения признаков
   - Объединение признаков через concatenation

2. **Гибкая конфигурация**
   - Все гиперпараметры в `config.yaml`
   - Легкое переключение между архитектурами
   - Настройка аугментации без изменения кода

3. **Production-ready**
   - Полное логирование в WandB
   - Автоматическое сохранение лучшей модели
   - Callbacks для оптимизации обучения
   - Скрипты для inference и evaluation

---

## 📁 Структура файлов

### Основные модули

| Файл | Описание |
|------|----------|
| `config.yaml` | Конфигурация всех гиперпараметров |
| `train.py` | Главный скрипт обучения |
| `data_loader.py` | Загрузка и preprocessing данных |
| `model.py` | Архитектура нейронной сети |
| `utils.py` | Вспомогательные функции |

### Инструменты

| Файл | Описание |
|------|----------|
| `predict.py` | Предсказание для одной монеты |
| `evaluate.py` | Оценка модели на тестовой выборке |
| `visualize_data.py` | Визуализация и анализ датасета |
| `quickstart.sh` | Автоматическая установка |

### Документация

| Файл | Описание |
|------|----------|
| `README.md` | Полная документация |
| `EXAMPLES.md` | Примеры использования |
| `PROJECT_SUMMARY.md` | Этот файл |

---

## 🏗️ Архитектура модели

```
┌─────────────────┐
│  Input A        │  (Аверс)
│  (384x384x3)    │
└────────┬────────┘
         │
         ├────────────────┐
         │                │
         ▼                ▼
    ┌────────────────────────┐
    │   Shared Backbone      │  (ResNet50/MobileNetV2/SimpleCNN)
    │   (Извлечение признаков)│
    └────────┬───────────────┘
             │
             ▼
    ┌──────────────────┐
    │ GlobalAvgPool2D  │
    └────────┬─────────┘
             │
             ├─────────────┐
             │             │
             ▼             ▼
    ┌──────────────────────────┐
    │     Concatenate          │  (Объединение признаков)
    └────────┬─────────────────┘
             │
             ▼
    ┌──────────────────┐
    │  Dense(256)      │  + ReLU + Dropout(0.5)
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Dense(128)      │  + ReLU + Dropout(0.25)
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Dense(5)        │  + Softmax
    │  (Выход)         │
    └──────────────────┘

┌─────────────────┐
│  Input B        │  (Реверс)
│  (384x384x3)    │
└─────────────────┘
```

---

## ⚙️ Основные гиперпараметры

### Данные
- **Image Size:** 384x384
- **Classes:** 5
- **Train/Val Split:** 80/20

### Обучение
- **Batch Size:** 32 (можно увеличить до 64-128 на A100)
- **Epochs:** 50
- **Optimizer:** Adam
- **Learning Rate:** 0.001 → автоматическое снижение
- **Loss:** Sparse Categorical Crossentropy

### Модель
- **Backbone:** ResNet50 / MobileNetV2 / SimpleCNN
- **Dropout:** 0.5
- **Dense Units:** 256 → 128

### Аугментация
- Horizontal & Vertical Flip
- Rotation (36°)
- Brightness (±20%)
- Zoom (±10%)

---

## 📊 Pipeline обучения

1. **Загрузка данных**
   - Сканирование папок с классами
   - Создание пар (obverse, reverse, label)
   - Split на train/val

2. **Preprocessing**
   - Resize до 384x384
   - Нормализация (0-1)
   - Аугментация (только для train)

3. **Обучение**
   - tf.data pipeline с prefetch
   - Callbacks: ModelCheckpoint, ReduceLR, EarlyStopping
   - Логирование в WandB

4. **Сохранение**
   - Лучшая модель по val_accuracy
   - Финальная модель после всех эпох

---

## 🚀 Быстрый старт (3 команды)

```bash
# 1. Установка
./quickstart.sh

# 2. Обучение
python train.py

# 3. Оценка
python evaluate.py --model saved_models/best_model.h5
```

---

## 📈 Метрики и мониторинг

### WandB Dashboard
- Train/Val Loss & Accuracy (каждая эпоха)
- Learning Rate schedule
- Confusion Matrix
- Примеры предсказаний
- Сравнение экспериментов

### TensorBoard
- Локальная альтернатива WandB
- Histograms слоев
- Graph visualization

### Console
- Real-time прогресс
- Метрики каждой эпохи
- Warnings о переобучении

---

## 🎛️ Возможности кастомизации

### Легко изменить:
- ✅ Архитектура бэкбона (1 строка в config)
- ✅ Гиперпараметры обучения (весь блок в config)
- ✅ Аугментация (блок augmentation в config)
- ✅ Размер изображений (image_size в config)
- ✅ Batch size (для разных GPU)

### Требует изменения кода:
- 🔧 Добавление новых бэкбонов (model.py)
- 🔧 Новые виды аугментации (data_loader.py)
- 🔧 Изменение loss функции (model.py)
- 🔧 Фильтрация данных по БД (data_loader.py - секция помечена)

---

## 💡 Рекомендации по обучению

### Если мало данных (<1000 образцов):
1. Используйте предобученные веса: `--pretrained`
2. Увеличьте аугментацию
3. Используйте высокий dropout (0.6-0.7)

### Если много данных (>5000 образцов):
1. Обучайте с нуля
2. Используйте сложные архитектуры (ResNet50)
3. Можно уменьшить dropout (0.3-0.4)

### Если переобучение (train_acc >> val_acc):
1. ↑ Увеличьте dropout
2. ↑ Усильте аугментацию
3. ↓ Уменьшите модель (меньше параметров)
4. Добавьте L2 регуляризацию

### Если недообучение (обе accuracy низкие):
1. ↑ Используйте более мощную модель
2. ↑ Увеличьте learning rate
3. ↓ Уменьшите dropout
4. ↓ Ослабьте аугментацию
5. ↑ Увеличьте epochs

---

## 🎓 Эксперименты для начала

### Эксперимент 1: Baseline
```yaml
model:
  backbone_name: "SimpleCNN"
training:
  epochs: 30
  learning_rate: 0.001
```

### Эксперимент 2: Transfer Learning
```bash
python train.py --pretrained
```

### Эксперимент 3: Strong Augmentation
```yaml
augmentation:
  rotation_factor: 0.2
  brightness_factor: 0.3
```

### Эксперимент 4: Production Model
```yaml
model:
  backbone_name: "ResNet50"
training:
  epochs: 100
  batch_size: 64
```

---

## 🔧 Системные требования

### Минимальные (CPU)
- Python 3.8+
- 8GB RAM
- 10GB HDD
- Batch size: 8-16

### Рекомендуемые (GPU)
- Python 3.8+
- GPU 8GB+ VRAM (RTX 3070+)
- 16GB RAM
- 20GB SSD
- Batch size: 32-64

### Оптимальные (A100)
- Python 3.8+
- NVidia A100 (40GB)
- 32GB RAM
- 50GB SSD
- Batch size: 128+

---

## 📞 Поддержка

### Частые проблемы

**Q: Out of Memory**  
A: Уменьшите `batch_size` в config.yaml

**Q: Модель не обучается**  
A: Увеличьте `learning_rate` или проверьте данные

**Q: Val accuracy не растет**  
A: Overfitting - увеличьте `dropout_rate` и аугментацию

**Q: WandB не работает**  
A: Выполните `wandb login` с вашим API ключом

---

## 📚 Полезные ресурсы

- **TensorFlow Docs:** https://www.tensorflow.org/
- **WandB Docs:** https://docs.wandb.ai/
- **Keras Applications:** https://keras.io/api/applications/

---

## 🎉 Успехов в обучении!

Этот проект создан для максимального удобства экспериментов с CNN для классификации монет. Все основные паттерны современного ML включены:
- ✅ Модульная архитектура
- ✅ Конфигурация через YAML
- ✅ Логирование в WandB
- ✅ Production-ready callbacks
- ✅ Полная документация
- ✅ Примеры использования

**Happy Training! 🚀**