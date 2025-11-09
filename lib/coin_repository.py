import os
from typing import List
import ydb
from dotenv import load_dotenv
from model.coin import Coin, CoinMetadata


class CoinYdbRepository:
    """Сохранение данных в YDB батчами"""

    def __init__(self, endpoint: str, database: str, table: str):
        self.endpoint = endpoint
        self.database = database
        self.table = table
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

    def insert_new_coins_batch(
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
                UPSERT INTO {self.table} (coin_id, source_url, title, condition, status)
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

    def get_new_coins_batch(self, batch_size: int) -> List[Coin]:
        """Получает батч новых монет из базы данных"""

        def execute_query(session):
            query = f"""
                SELECT coin_id, source_url, retry_count
                FROM {self.table}
                WHERE status = "new"
                LIMIT {batch_size};
            """

            result_sets = session.transaction().execute(
                query,
                commit_tx=True
            )

            new_coins = []
            for row in result_sets[0].rows:
                coin = Coin(
                    id=row.coin_id.decode('utf-8') if isinstance(row.coin_id, bytes) else row.coin_id,
                    name="",
                    url=row.source_url.decode('utf-8') if isinstance(row.source_url, bytes) else row.source_url
                )
                new_coins.append(coin)

            return new_coins

        try:
            return self.pool.retry_operation_sync(execute_query)
        except ydb.Error as e:
            print(f"Ошибка при получении батча монет из YDB: {e}")
            return []

    def add_coin_metadata(self, coin: CoinMetadata) -> bool:
        """Обновляет метаданные монеты и помечает её как 'scrapped_metadata'"""

        def execute_update(session):
            escaped_id = self._escape_string(coin.id)
            escaped_metal = self._escape_string(coin.metal)
            escaped_nominal = self._escape_string(coin.denomination)
            escaped_country = self._escape_string(coin.country)
            escaped_obverse = self._escape_string(coin.obverse_img)
            escaped_reverse = self._escape_string(coin.reverse_img)

            query = f"""
                UPDATE {self.table}
                SET
                    status = "scrapped_metadata",
                    metal = "{escaped_metal}",
                    nominal = "{escaped_nominal}",
                    coin_year = {coin.year},
                    country = "{escaped_country}",
                    image_url_obverse = "{escaped_obverse}",
                    image_url_reverse = "{escaped_reverse}",
                    scraped_at =  CurrentUtcTimestamp(),
                    retry_count = 0
                WHERE coin_id = "{escaped_id}";
            """

            session.transaction().execute(query, commit_tx=True)
            return True

        try:
            return self.pool.retry_operation_sync(execute_update)
        except ydb.Error as e:
            print(f"Ошибка при обновлении метаданных монеты {coin.id}: {e}")
            return False

    def increment_retry_count(self, coin_id: str) -> bool:
        """Увеличивает retry_count для монеты, при достижении лимита (>=5) помечает её как failed"""

        def execute_update(session):
            escaped_id = self._escape_string(coin_id)

            query = f"""
                UPDATE {self.table}
                SET
                    retry_count = retry_count + CAST(1 AS Uint8),
                    status = CASE
                        WHEN retry_count + CAST(1 AS Uint8) >= CAST(5 AS Uint8) THEN "failed"
                        ELSE status
                    END
                WHERE coin_id = "{escaped_id}";
            """

            session.transaction().execute(query, commit_tx=True)
            return True

        try:
            return self.pool.retry_operation_sync(execute_update)
        except ydb.Error as e:
            print(f"Ошибка при увеличении retry_count для монеты {coin_id}: {e}")
            return False


load_dotenv()

if __name__ == '__main__':
    YDB_ENDPOINT = os.getenv("YDB_ENDPOINT")
    YDB_DATABASE = os.getenv("YDB_DATABASE")
    TABLE = "coins_train"

    coins = [
        Coin("1233", "title", "/catalog/products/m2_91978_3_rublya_2025_goda_pridnestrove_drevnie_kreposti_na_dnestre_khotinskaya_krepost/"),
        Coin("1234", "title2", "/catalog/products/k10_5804_1_24_talera_1754_goda_saksoniya/"),
        Coin("1235", "title's with quote", "/catalog/products/k10_11098_50_santimov_1942_goda_frantsuzskaya_ekvatorialnaya_afrika/")
    ]

    # coin_metadata = CoinMetadata(id="1233", metal="gold", year=1234, country="Russia", denomination="Рубль")
    with CoinYdbRepository(YDB_ENDPOINT, YDB_DATABASE, TABLE) as saver:
        saved = saver.insert_new_coins_batch(coins, 1)
        print(f"Сохранено монет: {saved}")

        # 2
        # coins = saver.get_new_coins_batch(20)
        # for coin in coins:
        #     print(coin.url)

        # saver.add_coin_metadata(coin_metadata)

        # saver.increment_retry_count(coin_metadata.id)