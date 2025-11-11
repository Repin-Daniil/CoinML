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


@dataclass
class CoinImage:
    id: str
    condition: str
    image_obverse_url: str
    image_reverse_url: str
    s3_obverse_url: str = None
    s3_reverse_url: str = None
