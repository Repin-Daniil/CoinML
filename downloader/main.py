import os

import logging

from datetime import datetime

from dotenv import load_dotenv

from downloader.service import CoinImageService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'coin_parser_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()


def main():
    YDB_ENDPOINT = os.getenv("YDB_ENDPOINT")
    YDB_DATABASE = os.getenv("YDB_DATABASE")
    BASE_URL = os.getenv("BASE_URL")

    # Проверка переменных окружения
    if not all([YDB_ENDPOINT, YDB_DATABASE, BASE_URL]):
        logger.error("❌ Не все переменные окружения установлены!")
        return

    service = CoinImageService(
        ydb_endpoint=YDB_ENDPOINT,
        ydb_database=YDB_DATABASE,
        base_url=BASE_URL,
        batch_size=20,
        min_delay=2,
        max_delay=3
    )

    service.run()


if __name__ == "__main__":
    main()
