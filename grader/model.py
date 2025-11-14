"""
Модуль для создания архитектуры модели с shared weights
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

def create_simple_cnn(input_shape):
    """
    Создает простую CNN архитектуру для обучения с нуля.
    Альтернатива предобученным моделям для учебных целей.
    
    Args:
        input_shape (tuple): Форма входного изображения (H, W, C)
        
    Returns:
        tf.keras.Model: CNN модель без fully-connected слоев
    """
    model = keras.Sequential([
        # Block 1
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Block 5
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
    ], name='SimpleCNN')
    
    return model


def create_backbone(backbone_name, input_shape, pretrained=False):
    """
    Создает backbone для извлечения признаков
    
    Args:
        backbone_name (str): Название архитектуры 
            ('ResNet50', 'ResNet50V2', 'MobileNetV2', 'EfficientNetB0', 
             'EfficientNetB3', 'DenseNet121', 'ConvNeXtTiny', 'SimpleCNN')
        input_shape (tuple): Форма входного изображения (H, W, C)
        pretrained (bool): Использовать ли предобученные веса (imagenet)
        
    Returns:
        tf.keras.Model: Модель-бэкбон
    """
    
    weights = 'imagenet' if pretrained else None
    
    if backbone_name == "ResNet50":
        backbone = keras.applications.ResNet50(
            weights=weights,
            include_top=False,
            input_shape=input_shape,
            pooling=None
        )
        
    elif backbone_name == "ResNet50V2":
        backbone = keras.applications.ResNet50V2(
            weights=weights,
            include_top=False,
            input_shape=input_shape,
            pooling=None
        )
        
    elif backbone_name == "MobileNetV2":
        backbone = keras.applications.MobileNetV2(
            weights=weights,
            include_top=False,
            input_shape=input_shape,
            pooling=None
        )
        
    elif backbone_name == "EfficientNetB0":
        backbone = keras.applications.EfficientNetB0(
            weights=weights,
            include_top=False,
            input_shape=input_shape,
            pooling=None
        )
        
    elif backbone_name == "EfficientNetB3":
        backbone = keras.applications.EfficientNetB3(
            weights=weights,
            include_top=False,
            input_shape=input_shape,
            pooling=None
        )
        
    elif backbone_name == "DenseNet121":
        backbone = keras.applications.DenseNet121(
            weights=weights,
            include_top=False,
            input_shape=input_shape,
            pooling=None
        )
        
    elif backbone_name == "ConvNeXtTiny":
        backbone = keras.applications.ConvNeXtTiny(
            weights=weights,
            include_top=False,
            input_shape=input_shape,
            pooling=None
        )

    elif backbone_name == "SimpleCNN":
        backbone = create_simple_cnn(input_shape)
        weights = None  # Для кастомной сети
    else:
        raise ValueError(f"Неизвестный backbone: {backbone_name}")
    
    print(f"✓ Создан backbone: {backbone_name} "
          f"({'предобученные веса' if pretrained else 'обучение с нуля'})")
    
    return backbone

def create_augmentation_layer(config):
    aug_config = config['augmentation']
    layers = []

    if aug_config['flip_mode'] == 'horizontal':
        layers.append(tf.keras.layers.RandomFlip("horizontal"))
    elif aug_config['flip_mode'] == 'vertical':
        layers.append(tf.keras.layers.RandomFlip("vertical"))
    elif aug_config['flip_mode'] == 'horizontal_and_vertical':
        layers.append(tf.keras.layers.RandomFlip("horizontal_and_vertical"))
    
    if aug_config['rotation_factor'] > 0:
        layers.append(tf.keras.layers.RandomRotation(aug_config['rotation_factor']))
    
    if aug_config.get('zoom_factor', 0) > 0:
        layers.append(tf.keras.layers.RandomZoom(aug_config['zoom_factor']))
    
    if aug_config['brightness_factor'] > 0:
        layers.append(tf.keras.layers.RandomBrightness(aug_config['brightness_factor']))

    return tf.keras.Sequential(layers, name='augmentation')


def build_model(config):
    """
    Создает полную модель с двумя входами и shared weights
    
    Архитектура:
    - Два входа (аверс и реверс монеты)
    - Один общий бэкбон (shared weights) для извлечения признаков
    - GlobalAveragePooling2D для каждого выхода бэкбона
    - Concatenate объединяет признаки
    - Dense слои для классификации
    
    Args:
        config (dict): Конфигурация модели
        
    Returns:
        tf.keras.Model: Скомпилированная модель
    """
    print("\n" + "="*50)
    print("СОЗДАНИЕ МОДЕЛИ")
    print("="*50)

    image_size = config['data']['image_size']
    num_classes = config['data']['num_classes']
    backbone_name = config['model']['backbone_name']
    dropout_rate = config['model']['dropout_rate']
    dense_units = config['model']['dense_units']

    mixed_precision.set_global_policy('mixed_float16')

    input_shape = (image_size, image_size, 3)
    
    # Создаем два входа
    input_obverse = keras.Input(shape=input_shape, name='input_obverse')
    input_reverse = keras.Input(shape=input_shape, name='input_reverse')
    
    # 🔹 Создаём слой аугментации (он сам знает про training=True/False)
    augmentation = create_augmentation_layer(config)
    aug_obverse = augmentation(input_obverse)
    aug_reverse = augmentation(input_reverse)
    
    # Создаем ОДИН бэкбон (shared weights)
    backbone = create_backbone(backbone_name, input_shape) # try
    
    # Применяем бэкбон к обоим входам
    features_obverse = backbone(aug_obverse)
    features_reverse = backbone(aug_reverse)
    
    # Global Average Pooling для каждого выхода
    pooled_obverse = layers.GlobalAveragePooling2D(name='gap_obverse')(features_obverse)
    pooled_reverse = layers.GlobalAveragePooling2D(name='gap_reverse')(features_reverse)
    
    # Объединяем признаки
    combined = layers.Concatenate(name='concatenate')([pooled_obverse, pooled_reverse])
    
    # Classification head
    x = layers.Dense(dense_units, activation='relu', name='dense1')(combined)
    x = layers.Dropout(dropout_rate, name='dropout1')(x)
    
    # Можно добавить еще один Dense слой для большей выразительности
    x = layers.Dense(dense_units // 2, activation='relu', name='dense2')(x)
    x = layers.Dropout(dropout_rate / 2, name='dropout2')(x)
    
    # Выходной слой
    outputs = layers.Dense(num_classes, activation='softmax', name='output', dtype='float32')(x)
    
    # Создаем модель
    model = keras.Model(
        inputs=[input_obverse, input_reverse],
        outputs=outputs,
        name='CoinGrader'
    )
    
    # Компиляция
    optimizer_name = config['training']['optimizer']
    learning_rate = config['training']['learning_rate']
    
    if optimizer_name == "Adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Неизвестный оптимизатор: {optimizer_name}")
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n✓ Модель создана и скомпилирована")
    print(f"  Оптимизатор: {optimizer_name}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Dropout: {dropout_rate}")
    print("="*50 + "\n")
    
    return model


def build_model_with_pretrained(config, weights='imagenet'):
    """
    Вариант модели с предобученными весами (для экспериментов)
    
    Args:
        config (dict): Конфигурация модели
        weights (str): 'imagenet' или None
        
    Returns:
        tf.keras.Model: Скомпилированная модель
    """
    print("\n" + "="*50)
    print("СОЗДАНИЕ МОДЕЛИ С ПРЕДОБУЧЕННЫМИ ВЕСАМИ")
    print("="*50)
    
    image_size = config['data']['image_size']
    num_classes = config['data']['num_classes']
    backbone_name = config['model']['backbone_name']
    dropout_rate = config['model']['dropout_rate']
    dense_units = config['model']['dense_units']
    
    input_shape = (image_size, image_size, 3)
    
    # Создаем два входа
    input_obverse = keras.Input(shape=input_shape, name='input_obverse')
    input_reverse = keras.Input(shape=input_shape, name='input_reverse')
    
    # Создаем бэкбон с предобученными весами
    if backbone_name == "ResNet50":
        backbone = keras.applications.ResNet50(
            weights=weights,
            include_top=False,
            input_shape=input_shape,
            pooling=None
        )
    elif backbone_name == "MobileNetV2":
        backbone = keras.applications.MobileNetV2(
            weights=weights,
            include_top=False,
            input_shape=input_shape,
            pooling=None
        )
    else:
        raise ValueError(f"Предобученные веса доступны только для ResNet50 и MobileNetV2")
    
    print(f"✓ Создан backbone: {backbone_name} с весами {weights}")
    
    # Заморозка начальных слоев (опционально)
    backbone.trainable = False  # Раскомментировать для fine-tuning
    
    # Применяем бэкбон к обоим входам
    features_obverse = backbone(input_obverse)
    features_reverse = backbone(input_reverse)
    
    # Global Average Pooling
    pooled_obverse = layers.GlobalAveragePooling2D()(features_obverse)
    pooled_reverse = layers.GlobalAveragePooling2D()(features_reverse)
    
    # Объединяем
    combined = layers.Concatenate()([pooled_obverse, pooled_reverse])
    
    # Classification head
    x = layers.Dense(dense_units, activation='relu')(combined)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(
        inputs=[input_obverse, input_reverse],
        outputs=outputs,
        name='CoinGrader_Pretrained'
    )
    
    # Компиляция
    optimizer = keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✓ Модель с предобученными весами создана")
    print("="*50 + "\n")
    
    return model