import numpy as np
from typing import List, Dict, Any

from .context import ProcessingContext
from .steps.base import ProcessingStep

from .steps.image_steps import LoadImage, DetectCoin, CropAndMask, Normalize
from .steps.io_steps import SaveResults


class CoinProcessingPipeline:
    def __init__(self, steps: List[ProcessingStep]):
        self.steps = steps

    def process_image(self, image_path: str) -> ProcessingContext: # ⭐️ ИЗМЕНЕНИЕ ТУТ
        """
        Полный цикл обработки для одного изображения.
        Возвращает ВЕСЬ КОНТЕКСТ с результатами.
        """
        context = ProcessingContext(image_path=image_path)
        
        for step in self.steps:
            context = step(context)
            
            if context.status == "error":
                print(f"Обработка остановлена для {image_path} на шаге {step.__class__.__name__}")
                # Все равно возвращаем context, чтобы main мог прочитать context.error_message
                return context 
        
        print(f"Успешно обработано: {image_path}")
        return context
    

class PipelineBuilder:
    """
    "Фабрика", которая строит конвейер (pipeline) из конфига.
    """
    STEP_REGISTRY = {
        "LoadImage": LoadImage,
        "DetectCoin": DetectCoin,
        "CropAndMask": CropAndMask,
        "Normalize": Normalize,
        "SaveResults": SaveResults,
        # TODO: Добавь сюда "TensorFlowPredict"
    }

    def build_from_config(self, config: Dict[str, Any]) -> CoinProcessingPipeline:
        """
        Собирает объект CoinProcessingPipeline на основе словаря конфига.
        """
        steps = []
        global_output_dir = config.get("output_dir", "output")

        for step_config in config.get("pipeline_steps", []):
            step_name = step_config["name"]
            step_params = step_config.get("params", {})
            
            if step_name not in self.STEP_REGISTRY:
                raise ValueError(f"Неизвестный шаг в конфиге: {step_name}")
                
            step_class = self.STEP_REGISTRY[step_name]
            
            if step_name == "SaveResults" and "output_dir" not in step_params:
                step_params["output_dir"] = global_output_dir
            
            try:
                # C++: factory->CreateInstance(params)
                steps.append(step_class(**step_params))
            except TypeError as e:
                print(f"Ошибка инициализации шага {step_name} с параметрами {step_params}")
                raise e

        return CoinProcessingPipeline(steps)