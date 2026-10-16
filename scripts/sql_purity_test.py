from clickhouse_driver import Client as ClickHouseClient
import logging
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class DataQualityChecker:
    """
    Проверяет и чистит weather_raw → weather_clean.
    
    Что делает:
    1. Проверяет качество weather_raw (диагностика).
    2. Создаёт weather_clean с ReplacingMergeTree.
    3. Фильтрует аномалии через INSERT SELECT.
    4. Дедуплицирует через OPTIMIZE FINAL.
    """

    def __init__(
        self,
        host: str = 'clickhouse',
        port: int = 9000,
        user: str = 'default',
        password: str = 'default',
        database: str = 'weather_db',
        table_name: str = 'weather_raw',      # ← источник
        target_table: str = 'weather_clean',  # ← цель
    ):
        self.client = ClickHouseClient(
            host=host, port=port, user=user,
            password=password, database=database
        )
        self.database = database
        self.table_name = table_name
        self.target_table = target_table
        self.problems = []

    # ==========================================
    # 1. ПРОВЕРКА КАЧЕСТВА (диагностика)
    # ==========================================
    def check_quality(self) -> dict:
        """Проверяет weather_raw и возвращает статистику проблем."""
        logger.info("🔍 ПРОВЕРКА КАЧЕСТВА weather_raw")

        query = f"""
            SELECT
                count(*) AS total_rows,
                countIf(date IS NULL) AS null_dates,
                countIf(latitude IS NULL OR longitude IS NULL) AS null_coords,
                countIf(latitude < -90 OR latitude > 90 
                    OR longitude < -180 OR longitude > 180) AS invalid_coords,
                countIf(date < '1900-01-01' OR date > today() + 16) AS bad_dates,
                countIf(temperature_2m_max < -60 OR temperature_2m_max > 60) AS bad_temp_max,
                countIf(temperature_2m_min < -60 OR temperature_2m_min > 60) AS bad_temp_min,
                countIf(precipitation_sum < 0 OR precipitation_sum > 1000) AS bad_precip,
                countIf(wind_speed_10m_max < 0 OR wind_speed_10m_max > 100) AS bad_wind,
                count(*) - count(distinct (date, latitude, longitude)) AS duplicates
            FROM {self.database}.{self.table_name}
        """

        result = self.client.execute(query)
        row = result[0] #type: ignore

        stats = {
            'total_rows': row[0],
            'null_dates': row[1],
            'null_coords': row[2],
            'invalid_coords': row[3],
            'bad_dates': row[4],
            'bad_temp_max': row[5],
            'bad_temp_min': row[6],
            'bad_precip': row[7],
            'bad_wind': row[8],
            'duplicates': row[9],
        }

        total_problems = sum(v for k, v in stats.items() if k != 'total_rows') #type: ignore

        logger.info(f"📊 Всего строк: {stats['total_rows']}")
        logger.info(f"⚠️ Всего проблем: {total_problems}")

        for key, value in stats.items():
            if key != 'total_rows' and value > 0: #type: ignore
                logger.warning(f"  - {key}: {value}")

        return stats

    # ==========================================
    # 2. СОЗДАНИЕ weather_clean
    # ==========================================
    def _create_clean_table(self):
        """Создаёт weather_clean с той же структурой, что weather_raw."""
        existing = self.client.execute(
            f"DESCRIBE TABLE {self.database}.{self.table_name}"
        )

        columns_sql = []
        for row in existing: #type: ignore
            col_name = row[0]
            col_type = row[1]
            columns_sql.append(f'"{col_name}" {col_type}')

        columns_str = ', '.join(columns_sql)

        self.client.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.database}.{self.target_table} (
                {columns_str}
            ) ENGINE = ReplacingMergeTree()
            ORDER BY (date, latitude, longitude)
        """)

        logger.info(f"✅ Таблица {self.target_table} проверена/создана")

    # ==========================================
    # 3. ОЧИСТКА (главный метод)
    # ==========================================
    def clean(self) -> dict:
        """Читает weather_raw, фильтрует, пишет в weather_clean."""
        logger.info("=" * 60)
        logger.info("🧹 ОЧИСТКА weather_raw → weather_clean")
        logger.info("=" * 60)

        # 1. Диагностика до
        quality_before = self.check_quality()

        # 2. Создаём таблицу
        self._create_clean_table()

        # 3. Очищаем weather_clean
        self.client.execute(f"TRUNCATE TABLE {self.database}.{self.target_table}")

        # 4. INSERT SELECT с фильтрами
        self.client.execute(f"""
            INSERT INTO {self.database}.{self.target_table}
            SELECT *
            FROM {self.database}.{self.table_name}
            WHERE
                date IS NOT NULL
                AND latitude IS NOT NULL AND longitude IS NOT NULL
                AND latitude BETWEEN -90 AND 90
                AND longitude BETWEEN -180 AND 180
                AND date >= '1900-01-01' AND date <= today() + 16
                AND (temperature_2m_max IS NULL OR temperature_2m_max BETWEEN -60 AND 60)
                AND (temperature_2m_min IS NULL OR temperature_2m_min BETWEEN -60 AND 60)
                AND (precipitation_sum IS NULL OR precipitation_sum BETWEEN 0 AND 1000)
                AND (wind_speed_10m_max IS NULL OR wind_speed_10m_max BETWEEN 0 AND 100)
        """)

        # 5. Дедупликация
        self.client.execute(f"""
            OPTIMIZE TABLE {self.database}.{self.target_table} FINAL
        """)

        # 6. Статистика после
        clean_count = self.client.execute(
            f"SELECT count(*) FROM {self.database}.{self.target_table} FINAL"
        )[0][0] #type: ignore

        result = {
            'raw_rows': quality_before['total_rows'],
            'clean_rows': clean_count,
            'removed_rows': quality_before['total_rows'] - clean_count,
            'quality_before': quality_before,
        }

        logger.info("=" * 60)
        logger.info(f"✅ Было: {result['raw_rows']}, стало: {result['clean_rows']}")
        logger.info(f"🧹 Удалено: {result['removed_rows']}")
        logger.info("=" * 60)

        return result

    # ==========================================
    # 4. СТАРЫЙ ИНТЕРФЕЙС (для совместимости с DAG)
    # ==========================================
    def check_and_decide(self, auto_clean: bool = True) -> dict:
        """
        Обёртка для совместимости с DAG.
        Просто вызывает clean().
        """
        if not auto_clean:
            logger.info("⚠️ auto_clean=false — только проверка")
            return {
                'check_result': self.check_quality(),
                'cleaned': False,
                'status': 'checked_only',
            }

        return self.clean()