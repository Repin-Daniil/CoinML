import logging
import ssl
import random
import time

import aiohttp
from fake_useragent import UserAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class Fetcher:
    def __init__(self, timeout: int = 30):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.ua = UserAgent()
        self.session_user_agent = self.ua.random

    async def fetch_page(self, session: aiohttp.ClientSession, url, ) -> str:
        headers = {
            "User-Agent": self.session_user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": random.choice(["en-US,en;q=0.9", "ru-RU,ru;q=0.9"]),
            "Referer": url,
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
