from datetime import datetime
import json
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import requests
import logging


logger = logging.getLogger(__name__)


class MLDataLoader:
    """
    Загрузчик исторических данных для ML.
    Тянет архив за 10 лет из Open-Meteo Archive API.
    Сохраняет в S3: ml/archive/year=YYYY/.
    """

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    DAILY_VARIABLES = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]

    def __init__(self):
        pass

    # ==========================================
    # 1. СЕТКА КООРДИНАТ
    # ==========================================
    def _generate_grid(
        self,
        lat_min: float = 42.0,
        lat_max: float = 45.0,
        lon_min: float = 130.0,
        lon_max: float = 134.0,
        step: float = 0.5,
    ) -> list:
        """Генерирует сетку координат."""
        points = []
        lat = lat_min
        while lat <= lat_max:
            lon = lon_min
            while lon <= lon_max:
                points.append({"lat": lat, "lon": lon})
                lon += step
            lat += step
        return points

    # ==========================================
    # 2. ЗАПРОС ЗА ГОД
    # ==========================================
    def _fetch_year(
        self,
        latitudes: list,
        longitudes: list,
        year: int,
    ) -> dict:
        """
        Тянет данные за один год.
        Archive API не даёт больше 1 года за запрос.
        """
        params = {
            "latitude": latitudes,
            "longitude": longitudes,
            "daily": ",".join(self.DAILY_VARIABLES),
            "timezone": "Asia/Vladivostok",
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
        }

        response = requests.get(
            self.BASE_URL,
            params=params,
            timeout=180,
        )
        response.raise_for_status()
        return response.json()

    # ==========================================
    # 3. СОХРАНЕНИЕ В S3
    # ==========================================
    def _save_to_s3(
        self,
        data: dict,
        year: int,
        bucket: str = 'weather-data',
    ) -> str:
        """Сохраняет JSON в S3 с партиционированием по году."""
        s3_hook = S3Hook(aws_conn_id='s3_conn')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        key = (
            f"ml/archive/"
            f"year={year}/"
            f"archive_{year}_{timestamp}.json"
        )

        s3_hook.load_string(
            string_data=json.dumps(data, ensure_ascii=False),
            key=key,
            bucket_name=bucket,
            replace=True,
        )

        logger.info(f"✅ Сохранено: s3://{bucket}/{key}")
        return key

    # ==========================================
    # 4. ОСНОВНОЙ МЕТОД
    # ==========================================
    def load(
        self,
        start_year: int = 2015,
        end_year: int = 2025,
        step: float = 0.5,
        bucket: str = 'weather-data',
    ) -> list:
        """
        Тянет архив за указанные годы и сохраняет в S3.

        Returns:
            list: Список S3 ключей
        """
        # 1. Генерируем сетку
        grid_points = self._generate_grid(step=step)
        latitudes = [p["lat"] for p in grid_points]
        longitudes = [p["lon"] for p in grid_points]

        logger.info(f"📊 Сетка: {len(grid_points)} точек")
        logger.info(f"📅 Годы: {start_year} – {end_year}")
        logger.info(f"📦 Всего файлов: {end_year - start_year + 1}")

        saved_keys = []

        # 2. Тянем по годам
        for year in range(start_year, end_year + 1):
            logger.info(f"\n📥 Загрузка за {year}...")

            try:
                data = self._fetch_year(latitudes, longitudes, year)
                key = self._save_to_s3(data, year, bucket)
                saved_keys.append(key)

            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Ошибка за {year}: {e}")
                continue

        logger.info(f"\n✅ Всего сохранено: {len(saved_keys)} файлов")
        return saved_keys