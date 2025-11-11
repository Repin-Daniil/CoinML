import logging
import random
import tempfile
import time
from pathlib import Path
from typing import Optional

from common.extractor import CoinExtractor
from common.fetcher import Fetcher
from storage import ObjectStorage
from common.repository import CoinYdbRepository

from common.coin import CoinImage

logger = logging.getLogger(__name__)


class CoinImageService:
    def __init__(self, ydb_endpoint: str, ydb_database: str, base_url: str,
                 batch_size: int = 20, min_delay: float = 3, max_delay: float = 4):
        self.repository = CoinYdbRepository(ydb_endpoint, ydb_database, "coins").connect()
        self.storage = ObjectStorage("perception-coins")
        self.extractor = CoinExtractor()
        self.base_url = base_url
        self.batch_size = batch_size
        self.min_delay = min_delay
        self.max_delay = max_delay

        self.stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'total_time': 0,
            'start_time': None
        }

    def process_coin_images(self, coin: CoinImage):
        coin_start = time.time()

        try:
            logger.info(f"🔄 Начало обработки монеты {coin.id}")
            start_time = time.time()

            with (tempfile.TemporaryDirectory() as tmp_dir):
                tmp_path = Path(tmp_dir)

                obverse_downloaded, reverse_downloaded = self.download_coin_images(coin, tmp_path)

                if not (obverse_downloaded and reverse_downloaded):
                    logger.error(f"❌ Не удалось загрузить изображения для монеты {coin.id}")
                    return False

                obverse_s3_url, reverse_s3_url = self.upload_to_s3(coin, "raw", obverse_downloaded, reverse_downloaded)

                if not (obverse_s3_url and reverse_s3_url):
                    logger.error(f"❌ Не удалось загрузить в S3 для монеты {coin.id}")
                    return False

                coin.s3_obverse_url = obverse_s3_url
                coin.s3_reverse_url = reverse_s3_url

                obverse_cropped, reverse_cropped = self.crop_image(tmp_path, coin, obverse_downloaded, reverse_downloaded)
                obverse_cropped_s3_url, reverse_cropped_s3_url = self.upload_to_s3(coin, "dataset", obverse_cropped,
                                                                           reverse_cropped)
                if not (obverse_cropped_s3_url and reverse_cropped_s3_url):
                    logger.error(f"❌ Не удалось загрузить в S3 обрезанную монету {coin.id}")
                    return False

                logger.info(f"💾 Сохранение в БД для монеты {coin.id}")
                try:
                    self.repository.add_s3_images(coin)
                except Exception as e:
                    logger.error(f"❌ Ошибка сохранения в БД для монеты {coin.id}: {e}")
                    return False

                total_time = time.time() - start_time
                logger.info(f"✅ Монета {coin.id} успешно обработана за {total_time:.2f}с ")

                return True

        except Exception as e:
            coin_total_time = time.time() - coin_start
            logger.error(
                f"❌ Ошибка при обработке монеты {coin.id}: {e} | "
                f"Время до ошибки: {coin_total_time:.2f}с",
                exc_info=True
            )
            try:
                self.repository.increment_retry_count(coin.id)
            except Exception as repo_error:
                logger.error(f"Ошибка при обновлении счетчика retry для {coin.id}: {repo_error}")

            self.stats['failed'] += 1
            return False
        finally:
            self.stats['processed'] += 1

    def download_coin_images(self, coin: CoinImage, tmp_path):
        logger.info(f"📥 Загрузка изображений для монеты {coin.id}")

        obverse_original = tmp_path / f"{coin.id}_obverse_original.jpg"
        reverse_original = tmp_path / f"{coin.id}_reverse_original.jpg"

        download_start = time.time()

        fetcher = Fetcher()
        obverse_downloaded = fetcher.download_image(self.base_url + coin.image_obverse_url, obverse_original)
        reverse_downloaded = fetcher.download_image(self.base_url + coin.image_reverse_url, reverse_original)

        if obverse_downloaded and reverse_downloaded:
            download_time = time.time() - download_start
            logger.info(f"⏱️ Загрузка заняла {download_time:.2f}с")

            return obverse_original, reverse_original

        return None, None

    def crop_image(self, tmp_path, coin, obverse_path, reverse_path):
        obverse_cropped = tmp_path / f"{coin.id}_obverse_cropped.jpg"
        reverse_cropped = tmp_path / f"{coin.id}_reverse_cropped.jpg"

        logger.info(f"✂️ Обрезка изображений для монеты {coin.id}")
        crop_start = time.time()

        obverse_cropped_ok = self.extractor.extract_and_save(
            obverse_path,
            obverse_cropped
        )

        reverse_cropped_ok = self.extractor.extract_and_save(
            reverse_path,
            reverse_cropped
        )

        if not (obverse_cropped_ok and reverse_cropped_ok):
            logger.error(f"❌ Не удалось обрезать изображения для монеты {coin.id}")
            return None, None

        crop_time = time.time() - crop_start
        logger.info(f"⏱️ Обрезка заняла {crop_time:.2f}с")

        return obverse_cropped, reverse_cropped


    def upload_to_s3(self, coin: CoinImage, folder: str, obverse_downloaded, reverse_downloaded):
        logger.info(f"☁️ Загрузка в S3 для монеты {coin.id}")
        s3_start = time.time()

        s3_obverse_key = f"coins/{folder}/{coin.condition}/{coin.id}_obverse.jpg"
        s3_reverse_key = f"coins/{folder}/{coin.condition}/{coin.id}_reverse.jpg"

        obverse_s3_url = self.storage.upload(obverse_downloaded, s3_obverse_key)
        reverse_s3_url = self.storage.upload(reverse_downloaded, s3_reverse_key)

        if obverse_s3_url and reverse_s3_url:
            s3_time = time.time() - s3_start
            logger.info(f"⏱️ Загрузка в S3 заняла {s3_time:.2f}с")

        return obverse_s3_url, reverse_s3_url


    def process_batch(self) -> bool:
        """Обработка одного батча монет"""
        try:
            batch_start = time.time()
            coins = self.repository.get_coins_image_batch(self.batch_size)
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

                self.process_coin_images(coin)

                # Пауза между монетами (кроме последней)
                if i < len(coins):
                    pause = random.uniform(self.min_delay, self.max_delay)
                    logger.info(
                        f"⏳ Пауза {pause:.2f}с перед следующей монетой")
                    time.sleep(pause)

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


    def run(self, max_iterations: Optional[int] = None):
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

                has_coins = self.process_batch()

                if not has_coins:
                    consecutive_empty_batches += 1
                    if consecutive_empty_batches >= max_empty_batches:
                        logger.info(f"🏁 Нет монет для обработки после {max_empty_batches} попыток. Завершение работы.")
                        break

                    wait_time = 60 * consecutive_empty_batches
                    logger.info(f"⏰ Ожидание {wait_time}с перед следующей попыткой...")
                    time.sleep(wait_time)
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
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("\n⚠️ Получен сигнал остановки (Ctrl+C)")
        except Exception as e:
            logger.error(f"💥 Критическая ошибка в основном цикле: {e}", exc_info=True)
        finally:
            self.log_statistics()
            logger.info("👋 Парсер остановлен")
