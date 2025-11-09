import os
from typing import List
import ydb
from coin import Coin


class YDBBatchSaver:
    """Сохранение данных в YDB батчами"""

    def __init__(self, endpoint: str, database: str):
        self.endpoint = endpoint
        self.database = database
        self.driver = None
        self.pool = None

    def __enter__(self):
        """Инициализация подключения"""
        driver_config = ydb.DriverConfig(
            endpoint=self.endpoint,
            database=self.database,
            credentials=ydb.iam.ServiceAccountCredentials.from_file(
                os.environ["YDB_SERVICE_ACCOUNT_KEY_FILE_CREDENTIALS"]
            )
        )

        self.driver = ydb.Driver(driver_config)
        try:
            self.driver.wait(timeout=5, fail_fast=True)
        except Exception as e:
            print(f"Ошибка подключения к YDB: {e}")
            self.driver.stop()
            raise

        self.pool = ydb.SessionPool(self.driver)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Закрытие подключения"""
        if self.pool:
            self.pool.stop()
        if self.driver:
            self.driver.stop()

    def save_coins_batch(
            self,
            coins: List[Coin],
            condition: int
    ) -> int:
        if not coins:
            return 0

        def execute_batch(session):
            for coin in coins:
                query = f"""
                        DECLARE $url AS String;
                        DECLARE $coin_id AS String;
                        DECLARE $title AS String;
                        DECLARE $condition AS Int8;
                        DECLARE $status AS String;

                        $status = "new";
                        $coin_id = "{coin.id}";
                        $url = "{coin.url}";
                        $title = "{coin.name}";
                        $condition = {condition};

                        INSERT INTO coins (
                             status,
                             coin_id,
                             source_url,
                             title,
                             condition
                            )
                        VALUES ($status, $coin_id, $url, $title, $condition);
                    """

                session.transaction().execute(query, commit_tx=True)

            return len(coins)

        try:
            return self.pool.retry_operation_sync(execute_batch)
        except ydb.Error as e:
            print(f"Ошибка при пакетной вставке в YDB: {e}")
            return 0


if __name__ == '__main__':
    YDB_ENDPOINT = "grpcs://ydb.serverless.yandexcloud.net:2135"
    YDB_DATABASE = "/ru-central1/b1ghvb4orjqska1u1sio/etn5ttv3p6f6la9136co"

    page_coins = [Coin("1233", "title", "http")]

    with YDBBatchSaver(YDB_ENDPOINT, YDB_DATABASE) as saver:
        saved = saver.save_coins_batch(page_coins, 1)
