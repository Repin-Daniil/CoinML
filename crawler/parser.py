import asyncio
import random
import ssl

import aiohttp
from bs4 import BeautifulSoup
from typing import List, AsyncGenerator
from fake_useragent import UserAgent

from coin import Coin


class CoinParser:
    """Парсер каталога монет с пагинацией"""

    def __init__(self, url, timeout: int = 30):
        self.base_url = url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.ua = UserAgent()

    async def fetch_page(self, session: aiohttp.ClientSession, page: int) -> str:
        url = self.base_url.replace('PAGEN_1=1', f'PAGEN_1={page}')

        headers = {
            "User-Agent": self.ua.random,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": random.choice(["en-US,en;q=0.9", "ru-RU,ru;q=0.9"]),
            "Referer": self.base_url,
            "Connection": "keep-alive"
        }

        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        async with session.get(url, ssl=ssl_context, headers=headers, timeout=self.timeout) as response:
            response.raise_for_status()
            return await response.text()

    def parse_coins(self, html: str) -> List[Coin]:
        soup = BeautifulSoup(html, 'html.parser')
        coins = []

        # Найти все блоки с монетами
        product_items = soup.find_all('div', class_='product-item-wrapper')

        for item in product_items:
            try:
                # Название и ссылка
                name_link = item.find('a', class_='name')
                if not name_link:
                    continue

                name = name_link.get_text(strip=True)
                url = name_link.get('href', '')

                # ID монеты - пробуем несколько вариантов
                coin_id = None

                # Вариант 1: data-to-cart-btn
                cart_btn = item.find(attrs={'data-to-cart-btn': True})
                if cart_btn:
                    coin_id = cart_btn.get('data-to-cart-btn')

                # Вариант 2: id="list-item-..."
                if not coin_id:
                    list_item = item.find(id=lambda x: x and x.startswith('list-item-'))
                    if list_item:
                        coin_id = list_item.get('id').replace('list-item-', '')

                # Вариант 3: любой элемент с id внутри
                if not coin_id:
                    elem_with_id = item.find(id=True)
                    if elem_with_id:
                        coin_id = elem_with_id.get('id')

                if not coin_id:
                    coin_id = f"unknown_{hash(name)}"

                coin = Coin(
                    id=str(coin_id),
                    name=name,
                    url=url
                )
                coins.append(coin)

            except Exception as e:
                print(f"Ошибка при парсинге монеты: {e}")
                continue

        return coins

    async def parse_pages_generator(
            self,
            start_page: int,
            end_page: int
    ) -> AsyncGenerator[List[Coin], None]:
        async with aiohttp.ClientSession() as session:
            for page in range(start_page, end_page + 1):
                try:
                    print(f"Парсинг страницы {page}...")
                    html = await self.fetch_page(session, page)
                    coins = self.parse_coins(html)

                    print(f"✓ Страница {page}: найдено {len(coins)} монет")
                    yield coins

                    await asyncio.sleep(random.uniform(1.0, 2.0))

                except Exception as e:
                    print(f"✗ Ошибка на странице {page}: {e}")
                    yield []
