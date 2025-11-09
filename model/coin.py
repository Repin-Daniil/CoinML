from dataclasses import dataclass


@dataclass
class Coin:
    """Модель данных монеты"""
    id: str
    name: str
    url: str


@dataclass
class CoinMetadata:
    """Модель данных монеты"""
    id: str
    country: str
    metal: str
    year: int
    denomination: str
    obverse_img: str
    reverse_img: str
