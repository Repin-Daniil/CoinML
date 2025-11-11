import logging
import aiohttp
from bs4 import BeautifulSoup
from common.coin import CoinMetadata
from common.fetcher import Fetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class CoinMetadataParser:
    def parse_coin_info(self, html_content: str, coin_id: str) -> CoinMetadata:
        """Парсит HTML и возвращает объект CoinMetadata"""
        soup = BeautifulSoup(html_content, "html.parser")

        result = {
            "country": None,
            "denomination": None,
            "year": None,
            "metal": None,
            "obverse_img": None,
            "reverse_img": None,
        }

        field_mapping = {
            "Страна": "country",
            "Номинал": "denomination",
            "Год": "year",
            "Металл": "metal"
        }

        # Метод 1: meta-теги
        meta_props = soup.find_all("span", itemprop="additionalProperty")
        for prop in meta_props:
            name_meta = prop.find("meta", itemprop="name")
            value_meta = prop.find("meta", itemprop="value")
            if name_meta and value_meta:
                field_name = name_meta.get("content")
                field_value = value_meta.get("content")
                if field_name in field_mapping:
                    result[field_mapping[field_name]] = field_value

        # Метод 2: dl/dt/dd
        if not all(result.values()):
            dl_elements = soup.find_all("dl")
            for dl in dl_elements:
                dt_elements = dl.find_all("dt")
                dd_elements = dl.find_all("dd")
                for dt, dd in zip(dt_elements, dd_elements):
                    field_name = dt.get_text(strip=True)
                    field_value = dd.get_text(strip=True)
                    if field_name in field_mapping:
                        key = field_mapping[field_name]
                        if result[key] is None:
                            result[key] = field_value

        # Метод 3: парсинг изображений (inner — без водяных знаков)
        img_inner = soup.select_one(".img-inner.img-inner--item-mobile")
        if img_inner:
            front_img = img_inner.find("img", class_="front")
            back_img = img_inner.find("img", class_="back")
            if front_img and front_img.get("src"):
                result["obverse_img"] = front_img["src"]
            if back_img and back_img.get("src"):
                result["reverse_img"] = back_img["src"]

        # Возврат экземпляра CoinMetadata
        return CoinMetadata(
            id=coin_id,
            country=result["country"] or "",
            metal=result["metal"] or "",
            year=int(result["year"] or 0),
            denomination=result["denomination"] or "",
            obverse_img=result["obverse_img"] or "",
            reverse_img=result["reverse_img"] or "",
        )

    async def get_coin_metadata(self, coin_id: str, url: str) -> CoinMetadata | None:
        """Загружает страницу и возвращает CoinMetadata"""
        fetcher = Fetcher()
        async with aiohttp.ClientSession() as session:
            try:
                html = await fetcher.fetch_page(session, url)
                coin = self.parse_coin_info(html, coin_id)
                return coin
            except Exception as e:
                logger.error(f"Ошибка при парсинге {url}: {e}")
                return None
