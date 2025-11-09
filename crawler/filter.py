from dataclasses import dataclass
from typing import List
from model.coin import Coin


@dataclass
class FilterSettings:
    restricted_stems: List[str]


def _check_stems(text: str, stems: List[str]) -> bool:
    if not stems:
        return False

    return any(stem in text for stem in stems)


class CoinFilter:
    def __init__(self, settings: FilterSettings):
        self.stems = [s.lower() for s in settings.restricted_stems]

    def filter(self, coins: List[Coin]) -> List[Coin]:
        good, bad = [], []

        for coin in coins:
            text_to_check = coin.name.lower().strip()

            if _check_stems(text_to_check, self.stems):
                bad.append(coin)
            else:
                good.append(coin)

        if bad:
            print(f"❌ Отфильтровано {len(bad)}монет (содержат запрещенные слова).")

        return good


if __name__ == "__main__":
    filter_config = FilterSettings(
        restricted_stems=["рейх", "слаб"],
    )

    all_coins = [
        Coin(id="123", url="http", name="2 рубля 2004 года в слабе"),
    ]

    print("--- Запуск фильтрации (Блэклист) ---")
    coin_filter = CoinFilter(filter_config)
    filtered_coins = coin_filter.filter(all_coins)

    print("\n--- Подходящие монеты ({} шт.): ---".format(len(filtered_coins)))
    for good_coin in filtered_coins:
        print(f"  -> {good_coin.name}")
