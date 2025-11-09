import asyncio
import os
from dotenv import load_dotenv

from parser import CoinParser
from database import YDBBatchSaver
from filter import CoinFilter, FilterSettings

load_dotenv()

async def main():
    YDB_ENDPOINT = os.getenv("YDB_ENDPOINT")
    YDB_DATABASE = os.getenv("YDB_DATABASE")

    url = input("Адрес сайта: ")
    start_page = int(input("С какой страницы парсить? "))
    finish_page = int(input("До какой страницы парсить? "))
    condition = int(input("Какой класс сохранности? "))

    parser = CoinParser(url)
    total_saved = 0

    coin_filter = CoinFilter(FilterSettings(restricted_stems=["рейх", "слаб"]))

    with YDBBatchSaver(YDB_ENDPOINT, YDB_DATABASE) as saver:
        async for page_coins in parser.parse_pages_generator(1, start_page, finish_page):
            if page_coins:
                print(f"📦 Получен батч из {len(page_coins)} монет")

                try:
                    page_coins = coin_filter.filter(page_coins)
                    saved = saver.save_coins_batch(page_coins, condition)
                    total_saved += saved
                    print(f"✓ Сохранено {saved} монет в базу")
                except Exception as e:
                    print(f"❌ Ошибка сохранения: {e}")

    print(f"\n✅ Парсинг завершен! Всего сохранено: {total_saved} монет")


if __name__ == "__main__":
    asyncio.run(main())
