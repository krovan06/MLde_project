"""
Создаёт VIEW weather_map в ClickHouse.
VIEW объединяет fact_weather (API) и weather_detailed (MLP).
"""

from clickhouse_driver import Client as ClickHouseClient
import logging


logger = logging.getLogger(__name__)


class WeatherMapView:
    """
    Создаёт VIEW weather_map в ClickHouse.
    
    VIEW объединяет:
    - fact_weather (source='api') — данные от Open-Meteo.
    - weather_detailed (source='mlp') — данные от MLP.
    
    Использует CREATE OR REPLACE — идемпотентно.
    """

    def __init__(self, database: str = 'weather_db'):
        self.client = ClickHouseClient(
            host='clickhouse', port=9000,
            user='default', password='default',
            database=database
        )
        self.database = database

    # ==========================================
    # 1. ПРОВЕРКА: ЕСТЬ ЛИ weather_map
    # ==========================================
    def _view_exists(self) -> bool:
        """Проверяет, существует ли VIEW weather_map."""
        result = self.client.execute(f"""
            SELECT count(*) 
            FROM system.tables 
            WHERE database = '{self.database}' 
            AND name = 'weather_map'
        """)
        return result[0][0] > 0  # type: ignore

    # ==========================================
    # 2. ПРОВЕРКА: ЕСТЬ ЛИ ИСТОЧНИКИ
    # ==========================================
    def _check_sources(self):
        """Проверяет, что все источники существуют."""
        required = ['fact_weather', 'dim_location', 'weather_detailed']
        
        result = self.client.execute(f"""
            SELECT name 
            FROM system.tables 
            WHERE database = '{self.database}'
        """)
        existing = {row[0] for row in result}  # type: ignore

        missing = [t for t in required if t not in existing]

        if missing:
            raise ValueError(
                f"❌ Отсутствуют таблицы: {missing}. "
                f"Сначала запусти build_star и ml_inference."
            )

        logger.info("✅ Все источники на месте")

    # ==========================================
    # 3. СОЗДАНИЕ VIEW
    # ==========================================
    def create_view(self) -> str:
        """
        Создаёт VIEW weather_map.
        
        Returns:
            str: 'created' | 'updated' | 'exists'
        """
        logger.info("=" * 60)
        logger.info("🏗️ СОЗДАНИЕ VIEW weather_map")
        logger.info("=" * 60)

        # 1. Проверяем источники
        self._check_sources()

        # 2. Проверяем, есть ли VIEW
        existed_before = self._view_exists()

        # 3. Создаём или заменяем
        self.client.execute(f"""
            CREATE OR REPLACE VIEW {self.database}.weather_map AS
            SELECT
                f.date AS date,
                dl.latitude AS latitude,
                dl.longitude AS longitude,
                f.temperature_max AS temperature,
                f.precipitation AS precipitation,
                f.wind_speed_max AS wind_speed,
                'api' AS source
            FROM {self.database}.fact_weather f
            JOIN {self.database}.dim_location dl 
                ON f.location_id = dl.location_id

            UNION ALL

            SELECT
                date AS date,
                latitude AS latitude,
                longitude AS longitude,
                temperature_max AS temperature,
                CAST(NULL AS Nullable(Float32)) AS precipitation,
                CAST(NULL AS Nullable(Float32)) AS wind_speed,
                'mlp' AS source
            FROM {self.database}.weather_detailed
        """)

        # 4. Результат
        status = 'updated' if existed_before else 'created'

        if existed_before:
            logger.info("✅ VIEW weather_map обновлён")
        else:
            logger.info("✅ VIEW weather_map создан")

        # 5. Статистика
        count = self.client.execute(
            f"SELECT count(*) FROM {self.database}.weather_map"
        )[0][0]  # type: ignore

        logger.info(f"📊 Всего строк: {count}")
        logger.info("=" * 60)

        return status

    # ==========================================
    # 4. СТАТИСТИКА
    # ==========================================
    def get_stats(self) -> dict:
        """Возвращает статистику по источникам."""
        result = self.client.execute(f"""
            SELECT 
                source,
                count(*) AS cnt
            FROM {self.database}.weather_map
            GROUP BY source
        """)

        stats = {row[0]: row[1] for row in result}  # type: ignore
        return stats