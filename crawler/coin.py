from dataclasses import dataclass


@dataclass
class Coin:
    """Модель данных монеты"""
    id: str
    name: str
    url: str