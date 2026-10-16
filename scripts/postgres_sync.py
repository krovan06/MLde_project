"""
Синхронизация weather_map из ClickHouse в PostgreSQL.
weather_map — VIEW, объединяющий fact_weather (API) и weather_detailed (MLP).
"""

from clickhouse_driver import Client as ClickHouseClient
import psycopg2
import os
import logging


logger = logging.getLogger(__name__)


class PostgresSync:
    """
    Синхронизирует weather_map из ClickHouse в PostgreSQL.
    
    Что синхронизирует:
    - weather_map (VIEW) — объединение API и MLP данных.
    
    Что НЕ синхронизирует:
    - Витрины (daily_weather_stats, season_stats, ...).
      Они остаются в ClickHouse для аналитики.
    """

    def __init__(self, database: str = 'weather_db'):
        self.ch = ClickHouseClient(
            host='clickhouse',
            port=9000,
            user='default',
            password='default',
            database=database
        )
        self.database = database

        self.pg = psycopg2.connect(
            host=os.getenv('PG_HOST', 'postgres_stream'),
            port=int(os.getenv('PG_PORT', 5432)),
            user=os.getenv('PG_USER', 'stream'),
            password=os.getenv('PG_PASSWORD', 'stream'),
            database=os.getenv('PG_DATABASE', 'weather_stream')
        )

    # ==========================================
    # 1. СОЗДАНИЕ ТАБЛИЦЫ weather_map
    # ==========================================
    def _create_weather_map_table(self):
        """Создаёт таблицу weather_map в PostgreSQL."""
        cursor = self.pg.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weather_map (
                date DATE,
                latitude FLOAT,
                longitude FLOAT,
                temperature FLOAT,
                precipitation FLOAT,
                wind_speed FLOAT,
                source VARCHAR(10),
                PRIMARY KEY (date, latitude, longitude, source)
            )
        """)

        # Индекс для быстрого чтения
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_weather_map_date 
            ON weather_map (date DESC)
        """)

        self.pg.commit()
        cursor.close()

        logger.info("✅ Таблица weather_map проверена/создана")

    # ==========================================
    # 2. СИНХРОНИЗАЦИЯ weather_map
    # ==========================================
    def _sync_weather_map(self) -> int:
        """Синхронизирует weather_map из ClickHouse."""
        logger.info("📊 Синхронизация weather_map...")

        # 1. Читаем VIEW из ClickHouse
        rows = self.ch.execute("""
            SELECT 
                date, latitude, longitude,
                temperature, precipitation, wind_speed,
                source
            FROM weather_db.weather_map
        """)

        if not rows:
            logger.info("📭 weather_map: нет данных")
            return 0

        # 2. UPSERT в PostgreSQL
        cursor = self.pg.cursor()

        upsert_query = """
            INSERT INTO weather_map 
                (date, latitude, longitude, temperature, 
                 precipitation, wind_speed, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date, latitude, longitude, source)
            DO UPDATE SET
                temperature = EXCLUDED.temperature,
                precipitation = EXCLUDED.precipitation,
                wind_speed = EXCLUDED.wind_speed
        """

        for row in rows:  # type: ignore
            cursor.execute(upsert_query, row)

        self.pg.commit()
        cursor.close()

        logger.info(f"✅ weather_map: {len(rows)} строк")  # type: ignore
        return len(rows)  # type: ignore

    # ==========================================
    # 3. ОСНОВНОЙ МЕТОД
    # ==========================================
    def sync_all(self) -> int:
        """Синхронизирует weather_map."""
        logger.info("=" * 60)
        logger.info("📊 СИНХРОНИЗАЦИЯ В POSTGRESQL")
        logger.info("=" * 60)

        # 1. Создаём таблицу
        self._create_weather_map_table()

        # 2. Синхронизируем
        total = self._sync_weather_map()

        logger.info("=" * 60)
        logger.info(f"✅ Всего синхронизировано: {total} строк")
        logger.info("=" * 60)

        return total