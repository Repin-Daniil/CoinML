import cv2
import numpy as np
import os
from typing import Tuple

from .base import ProcessingStep
from ..context import ProcessingContext

class LoadImage(ProcessingStep):
    """Шаг 1: Загрузка изображения из файла."""
    def process(self, context: ProcessingContext) -> ProcessingContext:
        if not os.path.exists(context.image_path):
            context.fail(f"Файл не найден: {context.image_path}")
            return context
            
        context.original_image = cv2.imread(context.image_path)
        
        if context.original_image is None:
            context.fail(f"Не удалось прочитать изображение: {context.image_path}")
            return context
            
        context.status = "success"
        return context

class DetectCoin(ProcessingStep):
    """Шаг 2: Детекция монеты с помощью HoughCircles."""
    def __init__(self, blur_ksize: Tuple[int, int], dp: float, minDist_ratio: float, 
                 param1: int, param2: int, minRadius_ratio: float, maxRadius_ratio: float):
        self.blur_ksize = tuple(blur_ksize)
        self.dp = dp
        self.minDist_ratio = minDist_ratio
        self.param1 = param1
        self.param2 = param2
        self.minRadius_ratio = minRadius_ratio
        self.maxRadius_ratio = maxRadius_ratio

    def process(self, context: ProcessingContext) -> ProcessingContext:
        image = context.original_image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_ksize, 2)

        h, w = image.shape[:2]
        min_dim = min(h, w)
        
        minDist = int(min_dim * self.minDist_ratio)
        minRadius = int(min_dim * self.minRadius_ratio)
        maxRadius = int(min_dim * self.maxRadius_ratio)

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=self.dp, minDist=minDist,
            param1=self.param1, param2=self.param2,
            minRadius=minRadius, maxRadius=maxRadius
        )

        if circles is not None:
            circle = np.round(circles[0, 0]).astype("int")
            context.detected_circle = circle
            
            (x, y, r) = circle
            debug_img = image.copy()
            cv2.circle(debug_img, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(debug_img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            context.debug_images["detected"] = debug_img
        else:
            context.fail("Монета не найдена (HoughCircles)")

        return context

class CropAndMask(ProcessingStep):
    """Шаг 3: Обрезка, изменение размера и маскирование монеты."""
    def __init__(self, output_size: Tuple[int, int]):
        self.size = tuple(output_size)

    def process(self, context: ProcessingContext) -> ProcessingContext:
        (x, y, r) = context.detected_circle
        image = context.original_image

        start_x, end_x = max(0, x - r), min(image.shape[1], x + r)
        start_y, end_y = max(0, y - r), min(image.shape[0], y + r)
        cropped_coin = image[start_y:end_y, start_x:end_x]

        if cropped_coin.shape[0] == 0 or cropped_coin.shape[1] == 0:
            context.fail("Обрезка привела к пустому изображению")
            return context

        resized_coin = cv2.resize(cropped_coin, self.size, interpolation=cv2.INTER_AREA)

        mask = np.zeros(self.size, dtype="uint8")
        cv2.circle(mask, (self.size[0] // 2, self.size[1] // 2), self.size[0] // 2, 255, -1)
        masked_coin = cv2.bitwise_and(resized_coin, resized_coin, mask=mask)

        context.processed_image = masked_coin
        
        b, g, r_channels = cv2.split(masked_coin)
        final_png = cv2.merge((b, g, r_channels, mask))
        context.debug_images["cropped_png"] = final_png

        return context

class Normalize(ProcessingStep):
    """Шаг 4: Нормализация изображения (0-1)."""
    def process(self, context: ProcessingContext) -> ProcessingContext:
        context.processed_image = context.processed_image.astype("float32") / 255.0
        return context