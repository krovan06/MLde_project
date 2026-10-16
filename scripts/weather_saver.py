import pandas as pd
from datetime import datetime
import json
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from clickhouse_driver import Client as ClickHouseClient


class WeatherDataSaver:
    """Saver: читает staging → парсит → грузит в weather_raw (Silver)."""

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
    # 1. ЧТЕНИЕ ИЗ STAGING
    # ==========================================
    def _read_from_s3(
        self,
        bucket: str = 'weather-data',
        staging_prefix: str = 'staging'
    ) -> list:
        """Читает все JSON-файлы из staging и объединяет их."""
        staging_files = self.s3_hook.list_keys(
            bucket_name=bucket,
            prefix=staging_prefix
        )

        if not staging_files:
            raise ValueError(f"❌ Staging пуст: {staging_prefix}")

        all_data = []
        for key in staging_files:
            raw = self.s3_hook.read_key(key=key, bucket_name=bucket)
            all_data.extend(json.loads(raw))

        print(f"📂 Прочитано файлов: {len(staging_files)}, точек: {len(all_data)}")
        return all_data

    # ==========================================
    # 2. УНИВЕРСАЛЬНЫЙ ПАРСИНГ
    # ==========================================
    def _parse_json_to_dataframe(self, json_data: list) -> pd.DataFrame:
        """
        Универсальный парсинг: не знает про конкретные переменные.
        Превращает список точек в плоский DataFrame.
        """
        all_rows = []

        for point in json_data:
            lat = point['latitude']
            lon = point['longitude']
            data = point['data']

            times = data['time']              # Unix timestamps
            variables = data['variables']     # динамический словарь

            # Unix → дата
            dates = [
                datetime.utcfromtimestamp(t).strftime("%Y-%m-%d")
                for t in times
            ]

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

        print(f"📊 DataFrame: {len(df)} строк, {len(df.columns)} колонок")
        print(f"📊 Колонки: {list(df.columns)}")

        return df

    # ==========================================
    # 3. СОЗДАНИЕ ТАБЛИЦЫ В CLICKHOUSE
    # ==========================================
    def _create_table_if_not_exists(self, df: pd.DataFrame, table_name: str):
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

        print(f"✅ Таблица {table_name} проверена/создана")

    # ==========================================
    # 4. SCHEMA EVOLUTION
    # ==========================================
    def _evolve_schema(self, df: pd.DataFrame, table_name: str):
        """Добавляет новые колонки в ClickHouse, если они появились."""
        existing = self.client.execute(f"DESCRIBE TABLE {table_name}")
        existing_cols = {row[0] for row in existing}

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
            print(f"✅ Добавлена колонка: {col} ({ch_type})")

    # ==========================================
    # 5. ЗАГРУЗКА В CLICKHOUSE
    # ==========================================
    def _load_to_clickhouse(self, df: pd.DataFrame, table_name: str = 'weather_raw'):
        """Грузит DataFrame в ClickHouse."""
        if df.empty:
            print("📭 DataFrame пуст")
            return 0

        # 1. Создаём таблицу
        self._create_table_if_not_exists(df, table_name)

        # 2. Эволюция схемы
        self._evolve_schema(df, table_name)

        # ✅ 3. Преобразуем date в date-объекты (не datetime)
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

        print(f"✅ Загружено в {table_name}: {len(df)} записей")
        return len(df)

    # ==========================================
    # 6. ОРКЕСТРАЦИЯ
    # ==========================================
    def process(
        self,
        bucket: str = 'weather-data',
        staging_prefix: str = 'staging',
        table_name: str = 'weather_raw'
    ) -> dict:
        """Полный цикл: staging → weather_raw."""
        # 1. Читаем
        json_data = self._read_from_s3(bucket, staging_prefix)

        # 2. Парсим
        df = self._parse_json_to_dataframe(json_data)

        # 3. Грузим
        rows_count = self._load_to_clickhouse(df, table_name)

        return {
            'table_name': table_name,
            'rows_count': rows_count,
            'columns': list(df.columns),
        }