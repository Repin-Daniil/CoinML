from abc import ABC, abstractmethod
from ..context import ProcessingContext

class ProcessingStep(ABC):
    """
    Абстрактный базовый класс для одного шага в конвейере обработки.
    """
    @abstractmethod
    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Обрабатывает данные в 'context'.
        Должен вернуть измененный 'context'.
        """
        pass

    def __call__(self, context: ProcessingContext) -> ProcessingContext:
        """Позволяет вызывать экземпляр класса как функцию."""
        if context.status == "error":
            return context
        
        try:
            return self.process(context)
        except Exception as e:
            context.fail(f"Исключение на шаге {self.__class__.__name__}: {e}")
            return context