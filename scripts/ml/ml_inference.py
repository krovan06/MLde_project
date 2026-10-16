"""
Инференс ML-модели.
Загружает model.pt из S3 → генерирует сетку → weather_detailed.
"""

import io
import numpy as np
import torch 
from datetime import datetime
from typing import Optional
from clickhouse_driver import Client as ClickHouseClient
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

from scripts.ml.ml_model import WeatherMLP


# ==========================================
# КОНФИГУРАЦИЯ
# ==========================================
FEATURE_COLS = [
    'lat_norm', 'lon_norm',
    'month_sin', 'month_cos',
    'weekday_sin', 'weekday_cos',
]

# Границы для нормализации (те же, что в FeatureEngineer)
NORM_STATS = {
    'lat_min': 42.0, 'lat_max': 45.0,
    'lon_min': 130.0, 'lon_max': 134.0,
}

# Гиперпараметры модели (те же, что в Trainer)
HIDDEN_DIMS = [64, 128, 64]

# Сетка для инференса
GRID_STEP = 0.1


class MLInference:
    """
    Инференс MLP.
    
    Отличия от Trainer:
    - Не обучает.
    - Генерирует детализированную сетку.
    - Пишет в weather_detailed.
    """

    def __init__(self):
        self.client = ClickHouseClient(
            host='clickhouse', port=9000,
            user='default', password='default',
            database='weather_db'
        )
        self.s3_hook = S3Hook(aws_conn_id='s3_conn')

    # ==========================================
    # 1. ЗАГРУЗКА МОДЕЛИ ИЗ S3
    # ==========================================
    def _load_model(
        self,
        model_key: str,
        bucket: str = 'weather-data',
    ) -> WeatherMLP:
        """Загружает model.pt из S3."""
        print(f"📥 Загрузка модели: s3://{bucket}/{model_key}")

        # ✅ Читаем бинарный файл через get_key
        s3_object = self.s3_hook.get_key(
            key=model_key,
            bucket_name=bucket,
        )
        model_bytes = s3_object.get()['Body'].read()

        # Создаём модель той же архитектуры
        model = WeatherMLP(
            input_dim=len(FEATURE_COLS),
            hidden_dims=HIDDEN_DIMS,
            output_dim=1,
        )

        # Загружаем веса
        buffer = io.BytesIO(model_bytes)
        model.load_state_dict(torch.load(buffer, map_location='cpu'))
        model.eval()

        print("✅ Модель загружена")
        return model

    # ==========================================
    # 2. ГЕНЕРАЦИЯ ДЕТАЛИЗИРОВАННОЙ СЕТКИ
    # ==========================================
    def _generate_grid(
        self,
        date: str,
        lat_min: float = 42.0,
        lat_max: float = 45.0,
        lon_min: float = 130.0,
        lon_max: float = 134.0,
        step: float = GRID_STEP,
    ) -> list:
        """Генерирует детализированную сетку для даты."""
        points = []
        lat = lat_min
        while lat <= lat_max:
            lon = lon_min
            while lon <= lon_max:
                points.append({
                    'lat': round(lat, 4),
                    'lon': round(lon, 4),
                    'date': date,
                })
                lon += step
            lat += step
        return points

    # ==========================================
    # 3. ФИЧИ ДЛЯ СЕТКИ
    # ==========================================
    def _build_features(self, points: list) -> torch.Tensor:
        """Строит фичи для сетки."""
        features = []

        for p in points:
            # Нормализация координат
            lat_norm = (
                (p['lat'] - NORM_STATS['lat_min'])
                / (NORM_STATS['lat_max'] - NORM_STATS['lat_min'])
            )
            lon_norm = (
                (p['lon'] - NORM_STATS['lon_min'])
                / (NORM_STATS['lon_max'] - NORM_STATS['lon_min'])
            )

            # Sin/cos для месяца и дня недели
            dt = datetime.strptime(p['date'], '%Y-%m-%d')
            month = dt.month
            weekday = dt.weekday()

            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            weekday_sin = np.sin(2 * np.pi * weekday / 7)
            weekday_cos = np.cos(2 * np.pi * weekday / 7)

            features.append([
                lat_norm, lon_norm,
                month_sin, month_cos,
                weekday_sin, weekday_cos,
            ])

        return torch.tensor(features, dtype=torch.float32)

    # ==========================================
    # 4. ПРЕДСКАЗАНИЕ
    # ==========================================
    def _predict(self, model: WeatherMLP, points: list) -> list:
        """Предсказывает температуру для сетки."""
        X = self._build_features(points)

        with torch.no_grad():
            y_pred = model(X).numpy().flatten()

        results = []
        for i, p in enumerate(points):
            results.append({
                'date': p['date'],
                'latitude': p['lat'],
                'longitude': p['lon'],
                'temperature_max': float(y_pred[i]),
                'interpolated': 1,
            })

        return results

    # ==========================================
    # 5. СОЗДАНИЕ ТАБЛИЦЫ
    # ==========================================
    def _create_table_if_not_exists(self):
        """Создаёт weather_detailed."""
        self.client.execute("""
            CREATE TABLE IF NOT EXISTS weather_db.weather_detailed (
                date Date,
                latitude Float32,
                longitude Float32,
                temperature_max Float32,
                interpolated UInt8 DEFAULT 1
            ) ENGINE = MergeTree()
            ORDER BY (date, latitude, longitude)
        """)
        print("✅ Таблица weather_detailed проверена/создана")

    # ==========================================
    # 6. СОХРАНЕНИЕ В CLICKHOUSE
    # ==========================================
    def _save_to_clickhouse(
        self,
        results: list,
        table_name: str = 'weather_detailed',
    ) -> int:
        """Сохраняет результаты в weather_detailed."""
        if not results:
            print("📭 Нет результатов для сохранения")
            return 0

        # Создаём таблицу
        self._create_table_if_not_exists()

        # Удаляем старые данные за эти даты
        dates = set(r['date'] for r in results)
        for d in dates:
            self.client.execute(
                f"ALTER TABLE weather_db.{table_name} "
                f"DELETE WHERE date = '{d}'"
            )

        # ✅ Преобразуем date-строки в date-объекты
        data_tuples = [
            (
                datetime.strptime(r['date'], '%Y-%m-%d').date(),
                r['latitude'],
                r['longitude'],
                r['temperature_max'],
                r['interpolated'],
            )
            for r in results
        ]

        # Вставка
        self.client.execute(
            f"INSERT INTO weather_db.{table_name} "
            f"(date, latitude, longitude, temperature_max, interpolated) "
            f"VALUES",
            data_tuples,
        )

        print(f"✅ Сохранено: {len(results)} точек")
        return len(results)

    # ==========================================
    # 7. ОСНОВНОЙ МЕТОД
    # ==========================================
    def run(
        self,
        model_key: str,
        dates: Optional[list] = None,
        bucket: str = 'weather-data',
    ) -> dict:
        """
        Полный цикл инференса.

        Args:
            model_key: Путь к model.pt в S3
            dates: Список дат для инференса (по умолчанию — сегодня)
            bucket: Bucket в S3

        Returns:
            dict: Статистика инференса
        """
        print("=" * 60)
        print("🚀 ИНФЕРЕНС ML-МОДЕЛИ")
        print("=" * 60)

        # 1. Загружаем модель
        model = self._load_model(model_key, bucket)

        # 2. Даты
        if dates is None:
            dates = [datetime.now().strftime('%Y-%m-%d')]

        # 3. Обрабатываем каждую дату
        all_results = []
        for date in dates:
            print(f"\n📅 Обработка {date}...")
            points = self._generate_grid(date)
            results = self._predict(model, points)
            all_results.extend(results)
            print(f"  Предсказано: {len(results)} точек")

        # 4. Сохраняем в ClickHouse
        total = self._save_to_clickhouse(all_results)

        result = {
            'total_points': total,
            'dates': dates,
        }

        print("=" * 60)
        print(f"✅ ИНФЕРЕНС ЗАВЕРШЁН: {total} точек")
        print("=" * 60)

        return result