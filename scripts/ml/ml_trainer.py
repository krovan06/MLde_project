"""
Тренер финальной ML-модели для продакшна.
Читает ml_features → обучает → сохраняет model.pt в S3.
"""

import io
import json
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime
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
TARGET_COL = 'temp_max_norm'

HIDDEN_DIMS = [64, 128, 64]
EPOCHS = 50
LR = 1e-3
RANDOM_SEED = 42


class MLTrainer:
    """
    Тренер финальной модели для продакшна.
    
    Отличия от ml_research:
    - Не экспериментирует (фиксированные гиперпараметры).
    - Сохраняет модель в S3.
    - Сохраняет метрики в S3.
    """

    def __init__(self):
        self.client = ClickHouseClient(
            host='clickhouse', port=9000,
            user='default', password='default',
            database='weather_db'
        )
        self.s3_hook = S3Hook(aws_conn_id='s3_conn')
        torch.manual_seed(RANDOM_SEED)

    # ==========================================
    # 1. ЗАГРУЗКА ДАННЫХ
    # ==========================================
    def _load_features(self) -> pd.DataFrame:
        """Загружает ml_features из ClickHouse."""
        print("📥 Загрузка ml_features...")

        result = self.client.execute(
            "SELECT * FROM weather_db.ml_features",
            with_column_types=True,
        )
        columns = [col[0] for col in result[1]] #type: ignore
        df = pd.DataFrame(result[0], columns=columns) #type: ignore

        print(f"✅ Загружено: {len(df)} строк")
        return df

    # ==========================================
    # 2. ОБУЧЕНИЕ
    # ==========================================
    def _train_model(self, df: pd.DataFrame) -> tuple:
        """Обучает модель на всех данных."""
        print("🧠 Обучение модели...")

        # Данные
        X = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
        y = torch.tensor(df[TARGET_COL].values, dtype=torch.float32).reshape(-1, 1)

        # Модель
        model = WeatherMLP(
            input_dim=len(FEATURE_COLS),
            hidden_dims=HIDDEN_DIMS,
            output_dim=1,
        )

        # Оптимизатор + loss
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.MSELoss()

        # Обучение
        history = []
        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            history.append(loss.item())

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: loss={loss.item():.4f}")

        return model, history

    # ==========================================
    # 3. МЕТРИКИ
    # ==========================================
    def _compute_metrics(self, model: WeatherMLP, df: pd.DataFrame) -> dict:
        """Считает финальные метрики на обучающих данных."""
        model.eval()

        X = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
        y = torch.tensor(df[TARGET_COL].values, dtype=torch.float32).reshape(-1, 1)

        with torch.no_grad():
            y_pred = model(X)

        mse = torch.mean((y - y_pred) ** 2).item()
        mae = torch.mean(torch.abs(y - y_pred)).item()

        ss_res = torch.sum((y - y_pred) ** 2)
        ss_tot = torch.sum((y - y.mean()) ** 2)
        r2 = (1 - ss_res / ss_tot).item()

        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rows': len(df),
        }

    # ==========================================
    # 4. СОХРАНЕНИЕ В S3
    # ==========================================
    def _save_model_to_s3(
        self,
        model: WeatherMLP,
        bucket: str = 'weather-data',
    ) -> str:
        """Сохраняет model.state_dict() в S3."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        key = f"ml/models/model_{timestamp}.pt"

        # Сериализация в байты
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)

        self.s3_hook.load_bytes(
            bytes_data=buffer.getvalue(),
            key=key,
            bucket_name=bucket,
            replace=True,
        )

        print(f"✅ Модель сохранена: s3://{bucket}/{key}")
        return key

    def _save_metrics_to_s3(
        self,
        metrics: dict,
        bucket: str = 'weather-data',
    ) -> str:
        """Сохраняет метрики в S3."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        key = f"ml/models/metrics_{timestamp}.json"

        self.s3_hook.load_string(
            string_data=json.dumps(metrics, indent=2),
            key=key,
            bucket_name=bucket,
            replace=True,
        )

        print(f"✅ Метрики сохранены: s3://{bucket}/{key}")
        return key

    # ==========================================
    # 5. ОСНОВНОЙ МЕТОД
    # ==========================================
    def train(self, bucket: str = 'weather-data') -> dict:
        """
        Полный цикл: ml_features → обучение → S3.

        Returns:
            dict: model_key, metrics_key, metrics
        """
        print("=" * 60)
        print("🚀 ОБУЧЕНИЕ МОДЕЛИ")
        print("=" * 60)

        # 1. Загрузка
        df = self._load_features()

        # 2. Обучение
        model, history = self._train_model(df)

        # 3. Метрики
        metrics = self._compute_metrics(model, df)
        metrics['timestamp'] = datetime.now().isoformat()
        metrics['hidden_dims'] = HIDDEN_DIMS
        metrics['epochs'] = EPOCHS

        print(f"\n📊 Метрики:")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  R²:  {metrics['r2']:.4f}")

        # 4. Сохранение
        model_key = self._save_model_to_s3(model, bucket)
        metrics_key = self._save_metrics_to_s3(metrics, bucket)

        result = {
            'model_key': model_key,
            'metrics_key': metrics_key,
            'metrics': metrics,
        }

        print("=" * 60)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print("=" * 60)

        return result