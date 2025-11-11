import logging

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)

class CoinExtractor:
    def __init__(
            self,
            output_size: Tuple[int, int] = (384, 384),
            blur_ksize: Tuple[int, int] = (5, 5),
            hough_params: Optional[dict] = None,
            detection_max_size: int = 1024,  # НОВОЕ: макс. размер для детекции
    ):
        """
        Args:
            output_size: Размер выходного изображения (ширина, высота)
            blur_ksize: Размер ядра для размытия
            hough_params: Параметры для HoughCircles
            detection_max_size: Макс. размер большей стороны для детекции (для ускорения)
        """
        self.output_size = output_size
        self.blur_ksize = blur_ksize
        self.detection_max_size = detection_max_size

        default_params = {
            'dp': 1.2,
            'minDist_ratio': 0.25,
            'param1': 100,
            'param2': 35,
            'minRadius_ratio': 0.1,
            'maxRadius_ratio': 0.5
        }
        self.hough_params = {**default_params, **(hough_params or {})}

    def extract(
            self,
            image_path: str,
            normalize: bool = True,
            return_debug: bool = False
    ) -> Optional[np.ndarray]:
        """
        Извлекает монету из изображения.

        Args:
            image_path: Путь к изображению
            normalize: Нормализовать значения пикселей в [0, 1] (для TensorFlow)
            return_debug: Если True, возвращает tuple (coin, debug_image)

        Returns:
            BGR изображение монеты (3 канала) или None при ошибке
            Если return_debug=True: (coin, debug_image)
        """
        debug_image = None

        # Загрузка
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Файл не найден: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось прочитать изображение: {image_path}")

        # Детекция (на уменьшенном изображении для скорости)
        circle = self._detect_coin(image)
        if circle is None:
            return (None, debug_image) if return_debug else None

        # Debug изображение с найденной окружностью
        if return_debug:
            debug_image = self._create_debug_image(image, circle)

        # Вырезание и маскирование
        coin = self._crop_and_mask(image, circle)

        # Нормализация для нейросети
        if normalize:
            coin = coin.astype(np.float32) / 255.0

        return (coin, debug_image) if return_debug else coin

    def _detect_coin(self, image: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Детектирует монету на изображении, возвращает (x, y, radius)."""
        h, w = image.shape[:2]
        max_dim = max(h, w)

        # Если изображение большое, уменьшаем для детекции
        if max_dim > self.detection_max_size:
            scale = self.detection_max_size / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            detection_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            detection_img = image

        gray = cv2.cvtColor(detection_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_ksize, 2)

        dh, dw = detection_img.shape[:2]
        min_dim = min(dh, dw)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.hough_params['dp'],
            minDist=int(min_dim * self.hough_params['minDist_ratio']),
            param1=self.hough_params['param1'],
            param2=self.hough_params['param2'],
            minRadius=int(min_dim * self.hough_params['minRadius_ratio']),
            maxRadius=int(min_dim * self.hough_params['maxRadius_ratio'])
        )

        if circles is None:
            return None

        # Берем первую найденную окружность и масштабируем обратно к оригиналу
        x, y, r = np.round(circles[0, 0]).astype(int)
        x = int(x / scale)
        y = int(y / scale)
        r = int(r / scale)

        return (x, y, r)

    def _crop_and_mask(self, image: np.ndarray, circle: Tuple[int, int, int]) -> np.ndarray:
        """Вырезает и маскирует монету."""
        x, y, r = circle

        # Обрезка квадратной области вокруг монеты
        start_x = max(0, x - r)
        end_x = min(image.shape[1], x + r)
        start_y = max(0, y - r)
        end_y = min(image.shape[0], y + r)

        cropped = image[start_y:end_y, start_x:end_x]

        # Изменение размера
        resized = cv2.resize(cropped, self.output_size, interpolation=cv2.INTER_AREA)

        # ИЗМЕНЕНО: Белая маска вместо чёрной
        # Создаём белый фон
        masked = np.ones((self.output_size[1], self.output_size[0], 3), dtype=resized.dtype) * 255

        # Круговая маска
        mask = np.zeros(self.output_size, dtype=np.uint8)
        center = (self.output_size[0] // 2, self.output_size[1] // 2)
        cv2.circle(mask, center, self.output_size[0] // 2, 255, -1)

        # Накладываем монету на белый фон через маску
        masked = cv2.bitwise_and(resized, resized, mask=mask)
        masked[mask == 0] = 255  # Белый фон за пределами круга

        return masked

    def _create_debug_image(self, image: np.ndarray, circle: Tuple[int, int, int]):
        """Создаёт debug изображение с отмеченной окружностью и центром."""
        x, y, r = circle
        debug_img = image.copy()

        # Зелёный круг по контуру монеты
        cv2.circle(debug_img, (x, y), r, (0, 255, 0), 4)

        # Оранжевый квадрат в центре
        cv2.rectangle(debug_img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        # Красный крестик в центре
        cv2.line(debug_img, (x - 10, y), (x + 10, y), (0, 0, 255), 2)
        cv2.line(debug_img, (x, y - 10), (x, y + 10), (0, 0, 255), 2)

        return debug_img

    def extract_and_save(self, image_path, output_path):
        coin = self.extract(image_path, normalize=True)

        if coin is None:
            logger.warning(f"Не удалось извлечь монету из {image_path}")
            return False

        coin_uint8 = (coin * 255).astype(np.uint8)
        cv2.imwrite(str(output_path), coin_uint8)

        return True


# Примеры использования
if __name__ == "__main__":
    # 1. Для обучения TensorFlow (рекомендуется)
    # detection_max_size=1024 означает, что детекция будет на изображении не больше 1024px
    extractor = CoinExtractor(output_size=(384, 384), detection_max_size=1024)
    coin = extractor.extract("obverse.jpg", normalize=True)

    if coin is not None:
        print(f"Монета извлечена: {coin.shape}, диапазон: [{coin.min():.3f}, {coin.max():.3f}]")
        coin_uint8 = (coin * 255).astype(np.uint8)
        cv2.imwrite("dataset/coin_001.jpg", coin_uint8)

    # 2. С отладкой
    coin, debug = extractor.extract("obverse.jpg", normalize=False, return_debug=True)
    if coin is not None:
        cv2.imwrite("debug_detected.jpeg", debug)
        cv2.imwrite("coin_result.jpg", coin)