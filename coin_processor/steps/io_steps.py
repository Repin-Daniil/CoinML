import cv2
import os

from .base import ProcessingStep
from ..context import ProcessingContext

class SaveResults(ProcessingStep):
    """Шаг 5: Сохранение отладочных изображений."""
    def __init__(self, output_dir: str, save_detected: bool, save_cropped_png: bool):
        self.output_dir = output_dir
        self.save_detected = save_detected
        self.save_cropped_png = save_cropped_png
        
    def process(self, context: ProcessingContext) -> ProcessingContext:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        base_name = os.path.basename(context.image_path)
        name, _ = os.path.splitext(base_name)
        
        if self.save_detected and "detected" in context.debug_images:
            path = os.path.join(self.output_dir, f"{name}_detected.png")
            cv2.imwrite(path, context.debug_images["detected"])
            
        if self.save_cropped_png and "cropped_png" in context.debug_images:
            path = os.path.join(self.output_dir, f"{name}_cropped.png")
            cv2.imwrite(path, context.debug_images["cropped_png"])
            
        return context