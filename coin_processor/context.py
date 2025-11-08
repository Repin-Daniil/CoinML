import numpy as np
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class ProcessingContext:
    """
    Объект для хранения состояния обработки на всех этапах конвейера.
    """
    image_path: str
    
    # Статус обработки
    status: str = "pending"
    error_message: str = None
    
    # Промежуточные результаты
    original_image: np.ndarray = None
    detected_circle: np.ndarray = None  # (x, y, r)
    
    # Изображения для отладки и сохранения
    debug_images: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Конечный результат (то, что пойдет в TensorFlow)
    processed_image: np.ndarray = None

    def fail(self, message: str):
        """Хелпер для пометки контекста как неудавшегося."""
        self.status = "error"
        self.error_message = message
        print(f"Ошибка: {message} [Файл: {self.image_path}]")