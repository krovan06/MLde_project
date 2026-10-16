from datetime import datetime, timedelta
import json
from typing import Optional
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import openmeteo_requests
import requests_cache
from retry_requests import retry


class WeatherDataLoader:
    """Загрузчик: скачивает из Open-Meteo и кладёт сырой JSON в S3."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    DAILY_VARIABLES = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]

    def __init__(self):
        self.today = datetime.now()

    # ==========================================
    # 1. ЗАПРОС К API
    # ==========================================
    def _OpenMeteo_requests(self, **kwargs):
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)  # type: ignore

        params = {
            "latitude": kwargs.get("latitude", 43.029),
            "longitude": kwargs.get("longitude", 131.895),
            "daily": kwargs.get("daily", ",".join(self.DAILY_VARIABLES)),
            "timezone": kwargs.get("timezone", "Asia/Vladivostok"),
            "start_date": kwargs.get("start_date", self.today.strftime("%Y-%m-%d")),
            "end_date": kwargs.get("end_date", (self.today + timedelta(days=7)).strftime("%Y-%m-%d")),
        }
        params.update(kwargs)

        return openmeteo.weather_api(self.BASE_URL, params=params)

    # ==========================================
    # 2. ГЕНЕРАЦИЯ СЕТКИ
    # ==========================================
    def _generate_grid(
        self,
        lat_min: float = 42.0,
        lat_max: float = 45.0,
        lon_min: float = 130.0,
        lon_max: float = 134.0,
        step: float = 0.5,
    ) -> list:
        points = []
        lat = lat_min
        while lat <= lat_max:
            lon = lon_min
            while lon <= lon_max:
                points.append({'lat': lat, 'lon': lon})
                lon += step
            lat += step
        return points

    # ==========================================
    # 3. СЕРИАЛИЗАЦИЯ (универсальная)
    # ==========================================
    def _serialize_response(self, response) -> dict:
        """Превращает объект в JSON-сериализуемый словарь."""
        daily = response.Daily()

        times = list(range(daily.Time(), daily.TimeEnd(), daily.Interval()))

        variables = {}
        for i, var_name in enumerate(self.DAILY_VARIABLES):
            variables[var_name] = daily.Variables(i).ValuesAsNumpy().tolist()

        return {
            "time": times,
            "variables": variables,
        }

    # ==========================================
    # 4. ОСНОВНОЙ МЕТОД
    # ==========================================
    def fetch_grid_weather(
        self,
        step: float = 0.5,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        bucket: str = 'weather-data',
        prefix: str = 'staging',
    ) -> str:
        """Скачивает и сохраняет сырой JSON в S3."""
        # 1. Сетка
        grid_points = self._generate_grid(step=step)
        latitudes = [p['lat'] for p in grid_points]
        longitudes = [p['lon'] for p in grid_points]

        # 2. Запрос
        responses = self._OpenMeteo_requests(
            latitude=latitudes,
            longitude=longitudes,
            start_date=start_date,
            end_date=end_date,
        )

        # 3. Собираем сырые данные
        results = []
        for i, response in enumerate(responses):
            results.append({
                'latitude': grid_points[i]['lat'],
                'longitude': grid_points[i]['lon'],
                'data': self._serialize_response(response),
            })

        # 4. Сохраняем в S3
        s3_hook = S3Hook(aws_conn_id='s3_conn')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        key = f"{prefix}/weather_grid_{timestamp}.json"

        s3_hook.load_string(
            string_data=json.dumps(results, ensure_ascii=False),
            key=key,
            bucket_name=bucket,
            replace=True,
        )

        print(f"✅ Данные сохранены: s3://{bucket}/{key}")
        return key