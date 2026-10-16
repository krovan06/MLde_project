from clickhouse_driver import Client as ClickHouseClient
import logging


logger = logging.getLogger(__name__)


class StarBuilder:
    """
    Строит Star Schema из weather_clean (Silver+ → Gold).
    Знает маппинг между переменными API и колонками ClickHouse.
    """

    # Маппинг: имя в weather_clean → имя в fact_weather
    VARIABLE_MAPPING = {
        "temperature_2m_max": "temperature_max",
        "temperature_2m_min": "temperature_min",
        "precipitation_sum": "precipitation",
        "wind_speed_10m_max": "wind_speed_max",
    }

    def __init__(
        self,
        host: str = 'clickhouse',
        port: int = 9000,
        user: str = 'default',
        password: str = 'default',
        database: str = 'weather_db',
        source_table: str = 'weather_clean',  # ← читаем из Silver+
    ):
        self.client = ClickHouseClient(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )
        self.database = database
        self.source_table = source_table

    # ==========================================
    # 1. СОЗДАНИЕ ТАБЛИЦ STAR SCHEMA
    # ==========================================
    def _create_dim_location(self):
        self.client.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.database}.dim_location (
                location_id UInt32,
                latitude Float32,
                longitude Float32,
                region String DEFAULT 'Приморский край',
                city_name String DEFAULT 'unknown'
            ) ENGINE = ReplacingMergeTree()
            ORDER BY location_id
        """)
        logger.info("✅ dim_location проверена/создана")

    def _create_dim_date(self):
        self.client.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.database}.dim_date (
                date Date,
                year UInt16,
                month UInt8,
                day UInt8,
                weekday UInt8,
                season String
            ) ENGINE = ReplacingMergeTree()
            ORDER BY date
        """)
        logger.info("✅ dim_date проверена/создана")

    def _create_fact_weather(self):
        self.client.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.database}.fact_weather (
                date Date,
                location_id UInt32,
                temperature_max Float32,
                temperature_min Float32,
                precipitation Float32,
                wind_speed_max Float32,
                ingested_at DateTime DEFAULT now()
            ) ENGINE = ReplacingMergeTree(ingested_at)
            PARTITION BY toYYYYMM(date)
            ORDER BY (location_id, date)
        """)
        logger.info("✅ fact_weather проверена/создана")

    # ==========================================
    # 2. ЗАПОЛНЕНИЕ DIM_LOCATION
    # ==========================================
    def _populate_dim_location(self):
        """Добавляет новые локации из weather_clean в dim_location."""
        self.client.execute(f"""
            INSERT INTO {self.database}.dim_location (location_id, latitude, longitude)
            SELECT
                (SELECT max(location_id) FROM {self.database}.dim_location FINAL) 
                + rowNumberInAllBlocks() + 1 AS location_id,
                latitude,
                longitude
            FROM (
                SELECT DISTINCT latitude, longitude
                FROM {self.database}.{self.source_table}
                WHERE (latitude, longitude) NOT IN (
                    SELECT latitude, longitude 
                    FROM {self.database}.dim_location FINAL
                )
            )
        """)
        logger.info("✅ dim_location заполнена")

    # ==========================================
    # 3. ЗАПОЛНЕНИЕ DIM_DATE
    # ==========================================
    def _populate_dim_date(self):
        """Добавляет новые даты из weather_clean в dim_date."""
        self.client.execute(f"""
            INSERT INTO {self.database}.dim_date (date, year, month, day, weekday, season)
            SELECT
                date,
                toYear(date) AS year,
                toMonth(date) AS month,
                toDayOfMonth(date) AS day,
                toDayOfWeek(date) AS weekday,
                multiIf(
                    toMonth(date) IN (12, 1, 2), 'winter',
                    toMonth(date) IN (3, 4, 5), 'spring',
                    toMonth(date) IN (6, 7, 8), 'summer',
                    'autumn'
                ) AS season
            FROM (
                SELECT DISTINCT date
                FROM {self.database}.{self.source_table}
                WHERE date NOT IN (
                    SELECT date FROM {self.database}.dim_date FINAL
                )
            )
        """)
        logger.info("✅ dim_date заполнена")

    # ==========================================
    # 4. ЗАПОЛНЕНИЕ FACT_WEATHER
    # ==========================================
    def _populate_fact_weather(self):
        """
        Заполняет fact_weather из weather_clean + dim_location.
        Данные уже чистые (аномалии удалены в DataCleaner).
        """
        # Проверяем, какие переменные есть в weather_clean
        existing = self.client.execute(
            f"DESCRIBE TABLE {self.database}.{self.source_table}"
        )
        existing_cols = {row[0] for row in existing} #type: ignore

        # Формируем SELECT только для тех переменных, которые есть
        select_parts = []
        for api_name, ch_name in self.VARIABLE_MAPPING.items():
            if api_name in existing_cols:
                select_parts.append(f"w.{api_name} AS {ch_name}")
            else:
                logger.warning(
                    f"⚠️ Переменная {api_name} отсутствует в "
                    f"{self.source_table}, будет NULL"
                )
                select_parts.append(f"NULL AS {ch_name}")

        select_str = ',\n                '.join(select_parts)

        query = f"""
            INSERT INTO {self.database}.fact_weather 
                (date, location_id, temperature_max, temperature_min, 
                 precipitation, wind_speed_max)
            SELECT
                w.date,
                dl.location_id,
                {select_str}
            FROM {self.database}.{self.source_table} w
            JOIN {self.database}.dim_location dl 
                ON w.latitude = dl.latitude AND w.longitude = dl.longitude
        """

        self.client.execute(query)
        logger.info("✅ fact_weather заполнена")

    # ==========================================
    # 5. ОСНОВНОЙ МЕТОД
    # ==========================================
    def build_all(self) -> dict:
        """
        Полный цикл: weather_clean → Star Schema.
        
        Returns:
            dict: Статистика по каждой таблице
        """
        logger.info("=" * 60)
        logger.info("🏗️ ПОСТРОЕНИЕ STAR SCHEMA")
        logger.info(f"📂 Источник: {self.source_table}")
        logger.info("=" * 60)

        # 1. Создаём таблицы
        self._create_dim_location()
        self._create_dim_date()
        self._create_fact_weather()

        # 2. Заполняем
        self._populate_dim_location()
        self._populate_dim_date()
        self._populate_fact_weather()

        # 3. Собираем статистику
        result = self._get_stats()

        logger.info("=" * 60)
        logger.info(f"✅ Star Schema построена: {result}")
        logger.info("=" * 60)

        return result

    def _get_stats(self) -> dict:
        """Возвращает количество строк в каждой таблице."""
        stats = {}
        for table in ('fact_weather', 'dim_location', 'dim_date'):
            rows = self.client.execute(
                f"SELECT count(*) FROM {self.database}.{table} FINAL"
            )
            stats[table] = rows[0][0] if rows else 0 #type: ignore
        return stats