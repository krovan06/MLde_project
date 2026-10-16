import pandas as pd
from datetime import datetime
import json
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from clickhouse_driver import Client as ClickHouseClient
import logging


logger = logging.getLogger(__name__)


class MLDataSaver:
    """
    Saver для ML-ветки.
    Читает ml/archive/ из S3 → грузит в weather_archive (ClickHouse).
    
    Максимально гибкий: работает с любыми переменными из API.
    Не чистит данные — это задача DataCleaner.
    """

    def __init__(self):
        self.s3_hook = S3Hook(aws_conn_id='s3_conn')
        self.client = ClickHouseClient(
            host='clickhouse',
            port=9000,
            user='default',
            password='default',
            database='weather_db'
        )

    # ==========================================
    # 1. ЧТЕНИЕ ИЗ S3
    # ==========================================
    def _read_from_s3(
        self,
        bucket: str = 'weather-data',
        prefix: str = 'ml/archive/',
    ) -> list:
        """Читает все JSON-файлы из ml/archive/ и объединяет их."""
        s3_files = self.s3_hook.list_keys(
            bucket_name=bucket,
            prefix=prefix
        )

        if not s3_files:
            raise ValueError(f"❌ Нет файлов в s3://{bucket}/{prefix}")

        logger.info(f"📂 Найдено файлов: {len(s3_files)}")

        all_data = []
        for key in s3_files:
            raw = self.s3_hook.read_key(key=key, bucket_name=bucket)
            all_data.extend(json.loads(raw))

        logger.info(f"📊 Точек: {len(all_data)}")
        return all_data

    # ==========================================
    # 2. УНИВЕРСАЛЬНЫЙ ПАРСИНГ
    # ==========================================
    def _parse_json_to_dataframe(self, json_data: list) -> pd.DataFrame:
        """
        Универсальный парсинг JSON → DataFrame.
        
        Формат входа:
        [
            {
                "latitude": 43.0,
                "longitude": 131.0,
                "daily": {
                    "time": ["2015-01-01", ...],
                    "temperature_2m_max": [...],
                    ...
                }
            }
        ]
        
        Формат выхода:
        | date | latitude | longitude | temperature_2m_max | ... |
        """
        all_rows = []

        for point in json_data:
            lat = point['latitude']
            lon = point['longitude']
            daily = point['daily']

            dates = daily['time']
            
            # ✅ Все переменные КРОМЕ time
            variables = {
                k: v for k, v in daily.items() if k != 'time'
            }

            # Для каждого дня — строка
            for i in range(len(dates)):
                row = {
                    'date': dates[i],
                    'latitude': lat,
                    'longitude': lon,
                }

                # ✅ Динамически добавляем все переменные
                for var_name, values in variables.items():
                    row[var_name] = values[i]

                all_rows.append(row)

        df = pd.DataFrame(all_rows)
        df['date'] = pd.to_datetime(df['date'])

        logger.info(f"📊 DataFrame: {len(df)} строк, {len(df.columns)} колонок")
        logger.info(f"📊 Колонки: {list(df.columns)}")

        return df

    # ==========================================
    # 3. СОЗДАНИЕ ТАБЛИЦЫ
    # ==========================================
    def _create_table_if_not_exists(
        self,
        df: pd.DataFrame,
        table_name: str = 'weather_archive',
    ):
        """Создаёт таблицу с колонками из DataFrame."""
        type_map = {
            'float64': 'Float32',
            'float32': 'Float32',
            'int64': 'Int64',
            'int32': 'Int32',
            'datetime64[ns]': 'Date',
            'object': 'String',
            'bool': 'UInt8',
        }

        columns_sql = []
        for col in df.columns:
            if col == 'date':
                columns_sql.append(f'"{col}" Date')
            elif col in ('latitude', 'longitude'):
                columns_sql.append(f'"{col}" Float32')
            else:
                dtype = str(df[col].dtype)
                ch_type = type_map.get(dtype, 'String')
                columns_sql.append(f'"{col}" Nullable({ch_type})')

        columns_str = ', '.join(columns_sql)

        self.client.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {columns_str}
            ) ENGINE = MergeTree()
            ORDER BY (date, latitude, longitude)
        """)

        logger.info(f"✅ Таблица {table_name} проверена/создана")

    # ==========================================
    # 4. SCHEMA EVOLUTION
    # ==========================================
    def _evolve_schema(
        self,
        df: pd.DataFrame,
        table_name: str = 'weather_archive',
    ):
        """Добавляет новые колонки, если они появились в DataFrame."""
        existing = self.client.execute(f"DESCRIBE TABLE {table_name}")
        existing_cols = {row[0] for row in existing}  #type: ignore

        new_cols = set(df.columns) - existing_cols

        for col in new_cols:
            dtype = str(df[col].dtype)
            type_map = {
                'float64': 'Float32',
                'float32': 'Float32',
                'int64': 'Int64',
                'int32': 'Int32',
                'object': 'String',
            }
            ch_type = type_map.get(dtype, 'String')

            self.client.execute(f"""
                ALTER TABLE {table_name}
                ADD COLUMN IF NOT EXISTS "{col}" Nullable({ch_type})
            """)
            logger.info(f"✅ Добавлена колонка: {col} ({ch_type})")

    # ==========================================
    # 5. ЗАГРУЗКА В CLICKHOUSE
    # ==========================================
    def _load_to_clickhouse(
        self,
        df: pd.DataFrame,
        table_name: str = 'weather_archive',
    ) -> int:
        """Грузит DataFrame в ClickHouse."""
        if df.empty:
            logger.warning("📭 DataFrame пуст")
            return 0

        # 1. Создаём таблицу
        self._create_table_if_not_exists(df, table_name)

        # 2. Эволюция схемы
        self._evolve_schema(df, table_name)

        # 3. Преобразуем date в date-объекты
        df = df.copy()
        df['date'] = df['date'].dt.date

        # 4. Вставка
        columns = list(df.columns)
        columns_str = ', '.join([f'"{c}"' for c in columns])

        data_tuples = df.to_records(index=False).tolist()

        self.client.execute(
            f'INSERT INTO {table_name} ({columns_str}) VALUES',
            data_tuples,
            types_check=True
        )

        logger.info(f"✅ Загружено в {table_name}: {len(df)} записей")
        return len(df)

    # ==========================================
    # 6. ОРКЕСТРАЦИЯ
    # ==========================================
    def process(
        self,
        bucket: str = 'weather-data',
        prefix: str = 'ml/archive/',
        table_name: str = 'weather_archive',
    ) -> dict:
        """
        Полный цикл: S3 → ClickHouse (weather_archive).
        
        Returns:
            dict: Статистика загрузки
        """
        logger.info("=" * 60)
        logger.info("📥 ЗАГРУЗКА АРХИВА В CLICKHOUSE")
        logger.info("=" * 60)

        # 1. Читаем
        json_data = self._read_from_s3(bucket, prefix)

        # 2. Парсим
        df = self._parse_json_to_dataframe(json_data)

        # 3. Грузим
        rows_count = self._load_to_clickhouse(df, table_name)

        result = {
            'table_name': table_name,
            'rows_count': rows_count,
            'columns': list(df.columns),
        }

        logger.info("=" * 60)
        logger.info(f"✅ Загрузка завершена: {result}")
        logger.info("=" * 60)

        return result