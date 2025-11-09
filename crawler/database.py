import os
from typing import List
import ydb
from dotenv import load_dotenv
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

    @staticmethod
    def _escape_string(s: str) -> str:
        """Экранирование строк для YQL"""
        return s.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")

    def save_coins_batch(
            self,
            coins: List[Coin],
            condition: int
    ) -> int:
        if not coins:
            return 0

        def execute_batch(session):
            values_list = []
            for coin in coins:
                escaped_url = self._escape_string(coin.url)
                escaped_title = self._escape_string(coin.name)
                escaped_id = self._escape_string(coin.id)

                values_list.append(
                    f'("{escaped_id}", "{escaped_url}", "{escaped_title}", {condition}, "new")'
                )

            values_str = ",\n".join(values_list)

            query = f"""
                UPSERT INTO coins (coin_id, source_url, title, condition, status)
                VALUES
                {values_str};
            """

            session.transaction().execute(query, commit_tx=True)
            return len(coins)

        try:
            return self.pool.retry_operation_sync(execute_batch)
        except ydb.Error as e:
            print(f"Ошибка при пакетной вставке в YDB: {e}")
            return 0


load_dotenv()

if __name__ == '__main__':
    YDB_ENDPOINT = os.getenv("YDB_ENDPOINT")
    YDB_DATABASE = os.getenv("YDB_DATABASE")

    page_coins = [
        Coin("1233", "title", "http"),
        Coin("1234", "title2", "http2"),
        Coin("1235", "title's with quote", "http3")
    ]

    with YDBBatchSaver(YDB_ENDPOINT, YDB_DATABASE) as saver:
        saved = saver.save_coins_batch(page_coins, 1)
        print(f"Сохранено монет: {saved}")
