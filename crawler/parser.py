import asyncio
import hashlib
import logging
import random
import time
from typing import List, AsyncGenerator

import aiohttp
from bs4 import BeautifulSoup

from common.coin import Coin
from common.fetcher import Fetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class CoinParser:
    """Парсер каталога монет с пагинацией"""

    def __init__(self, url):
        self.base_url = url

    def parse_coins(self, html: str) -> List[Coin]:
        start_time = time.time()
        soup = BeautifulSoup(html, 'html.parser')
        coins = []

        product_items = soup.find_all('div', class_='product-item-wrapper')

        for item in product_items:
            try:
                name_link = item.find('a', class_='name')
                if not name_link:
                    continue

                name = name_link.get_text(strip=True)
                url = name_link.get('href', '')
                coin_id = hashlib.sha1(url.encode('utf-8')).hexdigest()

                coin = Coin(
                    id=str(coin_id),
                    name=name,
                    url=url
                )
                coins.append(coin)

            except Exception as e:
                logger.warning(f"Ошибка при парсинге монеты: {e}")
                continue

        parse_time = time.time() - start_time
        logger.info(f"Парсинг завершен за {parse_time:.2f}с, найдено {len(coins)} монет")
        return coins

    async def parse_pages_generator(
            self,
            start_page: int,
            end_page: int,
            min_delay: float = 3.0,
            max_delay: float = 6.0
    ) -> AsyncGenerator[List[Coin], None]:
        """
        Генератор для постраничного парсинга монет

        Args:
            start_page: начальная страница для парсинга
            end_page: последняя страница для парсинга
            min_delay: минимальная пауза между запросами (сек)
            max_delay: максимальная пауза между запросами (сек)
        """
        logger.info(f"Начало парсинга со страницы {start_page} до {end_page}")
        logger.info(f"Задержка между запросами: {min_delay}-{max_delay}с")

        fetcher = Fetcher()
        async with aiohttp.ClientSession() as session:
            for page in range(start_page, end_page + 1):
                page_start_time = time.time()

                try:
                    logger.info(f"--- Страница {page}/{end_page} ---")
                    url = self.base_url.replace('PAGEN_1=1', f'PAGEN_1={page}')
                    html = await fetcher.fetch_page(session, url)
                    coins = self.parse_coins(html)

                    yield coins

                    # Пауза перед следующим запросом
                    if page < end_page:
                        pause = random.uniform(min_delay, max_delay)
                        logger.info(f"Пауза {pause:.2f}с перед следующим запросом")
                        await asyncio.sleep(pause)

                    page_total_time = time.time() - page_start_time
                    logger.info(f"Страница {page} обработана за {page_total_time:.2f}с\n")

                except Exception as e:
                    logger.error(f"Ошибка на странице {page}: {e}")
                    yield []
