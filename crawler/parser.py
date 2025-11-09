import asyncio
import hashlib
import logging
import random
import ssl
import time
from typing import List, AsyncGenerator

import aiohttp
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

from coin import Coin

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


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

        start_time = time.time()
        async with session.get(url, ssl=ssl_context, headers=headers, timeout=self.timeout) as response:
            response.raise_for_status()
            html = await response.text()
            request_time = time.time() - start_time
            logger.info(f"Запрос выполнен за {request_time:.2f}с")
            return html

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
            end_page: int,
            start_page: int = 1,
            min_delay: float = 3.0,
            max_delay: float = 6.0
    ) -> AsyncGenerator[List[Coin], None]:
        """
        Генератор для постраничного парсинга монет

        Args:
            end_page: последняя страница для парсинга
            start_page: начальная страница (по умолчанию 1)
            min_delay: минимальная пауза между запросами (сек)
            max_delay: максимальная пауза между запросами (сек)
        """
        logger.info(f"Начало парсинга со страницы {start_page} до {end_page}")
        logger.info(f"Задержка между запросами: {min_delay}-{max_delay}с")

        async with aiohttp.ClientSession() as session:
            for page in range(start_page, end_page + 1):
                page_start_time = time.time()

                try:
                    logger.info(f"--- Страница {page}/{end_page} ---")
                    html = await self.fetch_page(session, page)
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
