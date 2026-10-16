from clickhouse_driver import Client as ClickHouseClient
import logging
from datetime import datetime


logger = logging.getLogger(__name__)


class MLFeatureEngineer:
    """
    Feature Engineer для ML-ветки.
    Читает weather_archive_clean → создаёт фичи → ml_features.
    
    Что делает:
    - Нормализует latitude, longitude (0..1).
    - Sin/cos для month (цикличность).
    - Sin/cos для weekday (цикличность).
    - Нормализует target (temperature_2m_max).
    
    Почему:
    - MLP не понимает «сырые» координаты и даты.
    - Sin/cos решает проблему цикличности (декабрь близко к январю).
    - Нормализация target ускоряет обучение.
    """

    def __init__(
        self,
        host: str = 'clickhouse',
        port: int = 9000,
        user: str = 'default',
        password: str = 'default',
        database: str = 'weather_db',
        source_table: str = 'weather_archive_clean',
        target_table: str = 'ml_features',
    ):
        self.client = ClickHouseClient(
            host=host, port=port, user=user,
            password=password, database=database
        )
        self.database = database
        self.source_table = source_table
        self.target_table = target_table

    # ==========================================
    # 1. ПОЛУЧЕНИЕ ГРАНИЦ И СТАТИСТИК
    # ==========================================
    def _get_stats(self) -> dict:
        """
        Получает min/max/mean/std для нормализации.
        Сохраняет их, чтобы использовать те же значения при инференсе.
        """
        query = f"""
            SELECT
                min(latitude) AS lat_min,
                max(latitude) AS lat_max,
                min(longitude) AS lon_min,
                max(longitude) AS lon_max,
                avg(temperature_2m_max) AS temp_mean,
                stddevPop(temperature_2m_max) AS temp_std,
                avg(temperature_2m_min) AS temp_min_mean,
                stddevPop(temperature_2m_min) AS temp_min_std,
                avg(precipitation_sum) AS precip_mean,
                stddevPop(precipitation_sum) AS precip_std,
                avg(wind_speed_10m_max) AS wind_mean,
                stddevPop(wind_speed_10m_max) AS wind_std
            FROM {self.database}.{self.source_table}
        """

        result = self.client.execute(query)
        row = result[0] #type: ignore

        stats = {
            'lat_min': row[0], 'lat_max': row[1],
            'lon_min': row[2], 'lon_max': row[3],
            'temp_mean': row[4], 'temp_std': row[5],
            'temp_min_mean': row[6], 'temp_min_std': row[7],
            'precip_mean': row[8], 'precip_std': row[9],
            'wind_mean': row[10], 'wind_std': row[11],
        }

        logger.info(f"📊 Статистика для нормализации:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        return stats

    # ==========================================
    # 2. СОЗДАНИЕ ТАБЛИЦЫ ml_features
    # ==========================================
    def _create_features_table(self):
        """Создаёт ml_features с фичами для MLP."""
        self.client.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.database}.{self.target_table} (
                -- Метаданные
                date Date,
                latitude Float32,
                longitude Float32,
                
                -- ✅ Фичи для MLP
                lat_norm Float32,      -- нормализованная широта
                lon_norm Float32,      -- нормализованная долгота
                month_sin Float32,     -- цикличность месяца
                month_cos Float32,
                weekday_sin Float32,   -- цикличность дня недели
                weekday_cos Float32,
                
                -- ✅ Target (нормализованные)
                temp_max_norm Float32,
                temp_min_norm Float32,
                precip_norm Float32,
                wind_norm Float32
            ) ENGINE = MergeTree()
            ORDER BY (date, latitude, longitude)
        """)

        logger.info(f"✅ Таблица {self.target_table} проверена/создана")

    # ==========================================
    # 3. ГЕНЕРАЦИЯ ФИЧ (SQL)
    # ==========================================
    def _populate_features(self, stats: dict):
        """
        Генерирует фичи через SQL.
        
        Формулы:
        - lat_norm = (lat - lat_min) / (lat_max - lat_min)
        - month_sin = sin(2π × month / 12)
        - month_cos = cos(2π × month / 12)
        - temp_norm = (temp - mean) / std
        """
        self.client.execute(f"""
            INSERT INTO {self.database}.{self.target_table}
            SELECT
                date,
                latitude,
                longitude,
                
                -- ✅ Нормализация координат (0..1)
                (latitude - {stats['lat_min']}) 
                    / ({stats['lat_max']} - {stats['lat_min']}) AS lat_norm,
                (longitude - {stats['lon_min']}) 
                    / ({stats['lon_max']} - {stats['lon_min']}) AS lon_norm,
                
                -- ✅ Sin/cos для месяца (цикличность)
                sin(2 * pi() * toMonth(date) / 12) AS month_sin,
                cos(2 * pi() * toMonth(date) / 12) AS month_cos,
                
                -- ✅ Sin/cos для дня недели (цикличность)
                sin(2 * pi() * toDayOfWeek(date) / 7) AS weekday_sin,
                cos(2 * pi() * toDayOfWeek(date) / 7) AS weekday_cos,
                
                -- ✅ Нормализация target (Z-score)
                (temperature_2m_max - {stats['temp_mean']}) 
                    / {stats['temp_std']} AS temp_max_norm,
                (temperature_2m_min - {stats['temp_min_mean']}) 
                    / {stats['temp_min_std']} AS temp_min_norm,
                (precipitation_sum - {stats['precip_mean']}) 
                    / {stats['precip_std']} AS precip_norm,
                (wind_speed_10m_max - {stats['wind_mean']}) 
                    / {stats['wind_std']} AS wind_norm
            FROM {self.database}.{self.source_table}
        """)

        logger.info("✅ Фичи сгенерированы")

    # ==========================================
    # 4. ОСНОВНОЙ МЕТОД
    # ==========================================
    def build_features(self) -> dict:
        """
        Полный цикл: weather_archive_clean → ml_features.
        
        Returns:
            dict: Статистика + параметры нормализации
        """
        logger.info("=" * 60)
        logger.info("🔧 ГЕНЕРАЦИЯ ФИЧ ДЛЯ ML")
        logger.info("=" * 60)

        # 1. Получаем статистики для нормализации
        stats = self._get_stats()

        # 2. Создаём таблицу
        self._create_features_table()

        # 3. Очищаем (перезапись)
        self.client.execute(f"TRUNCATE TABLE {self.database}.{self.target_table}")

        # 4. Генерируем фичи
        self._populate_features(stats)

        # 5. Статистика
        rows_count = self.client.execute(
            f"SELECT count(*) FROM {self.database}.{self.target_table}"
        )[0][0] #type: ignore

        result = {
            'table_name': self.target_table,
            'rows_count': rows_count,
            'normalization_stats': stats,
        }

        logger.info("=" * 60)
        logger.info(f"✅ Фичи готовы: {rows_count} строк")
        logger.info("=" * 60)

        return result