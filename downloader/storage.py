import boto3

import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class ObjectStorage:
    def __init__(self, bucket_name: str):
        self.client = boto3.session.Session().client(
            service_name='s3',
            endpoint_url='https://storage.yandexcloud.net'
        )
        self.bucket_name = bucket_name

    def upload(self, file_path: Path, s3_key: str) -> Optional[str]:
        """Загружает файл в S3 и возвращает URL."""
        try:
            self.client.upload_file(
                str(file_path),
                self.bucket_name,
                s3_key,
                # ExtraArgs={'ContentType': 'image/jpg'}
            )

            logger.info(f"☁️ Загружено в S3: {s3_key}")
            return s3_key
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки в S3 {s3_key}: {e}")
            return None

load_dotenv()

if __name__ == "__main__":
    storage = ObjectStorage("perception-coins")
    storage.upload("try.txt", f"coins/test.txt")