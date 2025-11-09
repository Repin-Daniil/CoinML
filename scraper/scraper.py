import asyncio
import os
import random
import logging
import time
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

from lib.coin_repository import CoinYdbRepository
from parser import CoinMetadataParser

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


class CoinParserService:
    def __init__(self, ydb_endpoint: str, ydb_database: str, base_url: str,
                 batch_size: int = 20, min_delay: float = 3, max_delay: float = 4):
        self.ydb_endpoint = ydb_endpoint
        self.ydb_database = ydb_database
        self.base_url = base_url
        self.batch_size = batch_size
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.parser = CoinMetadataParser()

        # Статистика
        self.stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'total_time': 0,
            'start_time': None
        }

    async def process_coin(self, coin, repository: CoinYdbRepository) -> bool:
        """Обработка одной монеты с замером времени"""
        coin_start = time.time()

        try:
            logger.info(f"🔄 Начало обработки монеты {coin.id}")

            # Парсинг метаданных
            parse_start = time.time()
            coin_metadata = await self.parser.get_coin_metadata(coin.id, self.base_url + coin.url)
            parse_time = time.time() - parse_start

            if coin_metadata:
                # Сохранение в БД
                save_start = time.time()
                repository.add_coin_metadata(coin_metadata)
                save_time = time.time() - save_start

                coin_total_time = time.time() - coin_start
                logger.info(
                    f"✅ Монета {coin.id} обработана успешно | "
                    f"Парсинг: {parse_time:.2f}с | "
                    f"Сохранение: {save_time:.2f}с | "
                    f"Всего: {coin_total_time:.2f}с"
                )
                self.stats['successful'] += 1
                return True
            else:
                logger.warning(f"❌ Не удалось спарсить данные монеты {coin.id}")
                repository.increment_retry_count(coin.id)
                self.stats['failed'] += 1
                return False

        except Exception as e:
            coin_total_time = time.time() - coin_start
            logger.error(
                f"❌ Ошибка при обработке монеты {coin.id}: {e} | "
                f"Время до ошибки: {coin_total_time:.2f}с",
                exc_info=True
            )
            try:
                repository.increment_retry_count(coin.id)
            except Exception as repo_error:
                logger.error(f"Ошибка при обновлении счетчика retry для {coin.id}: {repo_error}")

            self.stats['failed'] += 1
            return False
        finally:
            self.stats['processed'] += 1

    async def process_batch(self) -> bool:
        """Обработка одного батча монет"""
        try:
            with CoinYdbRepository(self.ydb_endpoint, self.ydb_database, "coins_train") as repository:
                batch_start = time.time()
                coins = repository.get_new_coins_batch(self.batch_size)
                fetch_time = time.time() - batch_start

                if not coins:
                    logger.info("📭 Нет монет для обработки")
                    return False

                logger.info(
                    f"📦 Получен батч из {len(coins)} монет | "
                    f"Время получения: {fetch_time:.2f}с"
                )

                for i, coin in enumerate(coins, 1):
                    logger.info(f"[{i}/{len(coins)}] Обработка монеты {coin.id}")

                    await self.process_coin(coin, repository)

                    # Пауза между монетами (кроме последней)
                    if i < len(coins):
                        pause = random.uniform(self.min_delay, self.max_delay)
                        logger.info(f"⏳ Пауза {pause:.2f}с перед следующей монетой")
                        await asyncio.sleep(pause)

                batch_total_time = time.time() - batch_start
                logger.info(
                    f"📊 Батч обработан | "
                    f"Всего: {len(coins)} | "
                    f"Успешно: {self.stats['successful']} | "
                    f"Ошибки: {self.stats['failed']} | "
                    f"Время батча: {batch_total_time:.2f}с"
                )

                return True

        except Exception as e:
            logger.error(f"❌ Критическая ошибка при обработке батча: {e}", exc_info=True)
            return False

    def log_statistics(self):
        """Вывод общей статистики"""
        if self.stats['start_time']:
            uptime = time.time() - self.stats['start_time']
            avg_time = uptime / self.stats['processed'] if self.stats['processed'] > 0 else 0
            success_rate = (self.stats['successful'] / self.stats['processed'] * 100) if self.stats[
                                                                                             'processed'] > 0 else 0

            logger.info(
                f"\n{'=' * 60}\n"
                f"📈 СТАТИСТИКА\n"
                f"{'=' * 60}\n"
                f"Время работы: {uptime / 3600:.2f}ч ({uptime:.0f}с)\n"
                f"Обработано монет: {self.stats['processed']}\n"
                f"Успешно: {self.stats['successful']}\n"
                f"Ошибок: {self.stats['failed']}\n"
                f"Success rate: {success_rate:.1f}%\n"
                f"Среднее время на монету: {avg_time:.2f}с\n"
                f"{'=' * 60}"
            )

    async def run(self, max_iterations: Optional[int] = None):
        """Основной цикл работы скрипта"""
        self.stats['start_time'] = time.time()
        iteration = 0
        consecutive_empty_batches = 0
        max_empty_batches = 3

        logger.info("🚀 Запуск парсера монет")
        logger.info(f"Размер батча: {self.batch_size}")
        logger.info(f"Задержка между монетами: {self.min_delay}-{self.max_delay}с")

        try:
            while True:
                iteration += 1
                logger.info(f"\n{'=' * 60}")
                logger.info(f"🔄 Итерация #{iteration}")
                logger.info(f"{'=' * 60}")

                has_coins = await self.process_batch()

                if not has_coins:
                    consecutive_empty_batches += 1
                    if consecutive_empty_batches >= max_empty_batches:
                        logger.info(f"🏁 Нет монет для обработки после {max_empty_batches} попыток. Завершение работы.")
                        break

                    wait_time = 60 * consecutive_empty_batches
                    logger.info(f"⏰ Ожидание {wait_time}с перед следующей попыткой...")
                    await asyncio.sleep(wait_time)
                else:
                    consecutive_empty_batches = 0

                # Вывод статистики каждые 10 итераций
                if iteration % 10 == 0:
                    self.log_statistics()

                # Проверка лимита итераций
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"🏁 Достигнут лимит итераций: {max_iterations}")
                    break

                # Небольшая пауза между батчами
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("\n⚠️ Получен сигнал остановки (Ctrl+C)")
        except Exception as e:
            logger.error(f"💥 Критическая ошибка в основном цикле: {e}", exc_info=True)
        finally:
            self.log_statistics()
            logger.info("👋 Парсер остановлен")


async def main():
    YDB_ENDPOINT = os.getenv("YDB_ENDPOINT")
    YDB_DATABASE = os.getenv("YDB_DATABASE")
    BASE_URL = os.getenv("BASE_URL")

    # Проверка переменных окружения
    if not all([YDB_ENDPOINT, YDB_DATABASE, BASE_URL]):
        logger.error("❌ Не все переменные окружения установлены!")
        return

    service = CoinParserService(
        ydb_endpoint=YDB_ENDPOINT,
        ydb_database=YDB_DATABASE,
        base_url=BASE_URL,
        batch_size=20,
        min_delay=3,
        max_delay=4
    )

    await service.run()


if __name__ == "__main__":
    asyncio.run(main())