"""
Research script для ML-ветки.
Сохраняет результаты в S3.
"""

import io
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # ← для работы без GUI
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datetime import datetime
from clickhouse_driver import Client as ClickHouseClient
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

from scripts.ml.ml_model import WeatherMLP


# ==========================================
# КОНФИГУРАЦИЯ
# ==========================================
RANDOM_SEED = 42
EPOCHS = 50
BUCKET = 'weather-data'
REPORTS_PREFIX = 'ml/reports'


class MLResearch:
    """Исследование ML-модели. Сохраняет результаты в S3."""

    def __init__(self):
        self.client = ClickHouseClient(
            host='clickhouse', port=9000,
            user='default', password='default',
            database='weather_db'
        )
        self.s3_hook = S3Hook(aws_conn_id='s3_conn')
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    # ==========================================
    # СОХРАНЕНИЕ В S3
    # ==========================================
    def _save_plot_to_s3(self, fig, key: str):
        """Сохраняет matplotlib-график в S3."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)

        s3_key = f"{REPORTS_PREFIX}/{key}"
        self.s3_hook.load_bytes(
            bytes_data=buffer.getvalue(),
            key=s3_key,
            bucket_name=BUCKET,
            replace=True,
        )
        plt.close(fig)
        print(f"  💾 s3://{BUCKET}/{s3_key}")

    def _save_json_to_s3(self, data: dict, key: str):
        """Сохраняет JSON в S3."""
        s3_key = f"{REPORTS_PREFIX}/{key}"
        self.s3_hook.load_string(
            string_data=json.dumps(data, indent=2, default=str),
            key=s3_key,
            bucket_name=BUCKET,
            replace=True,
        )
        print(f"  💾 s3://{BUCKET}/{s3_key}")

    # ==========================================
    # 1. ЗАГРУЗКА ДАННЫХ
    # ==========================================
    def load_data(self) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("📥 ЗАГРУЗКА ДАННЫХ")
        print("=" * 60)

        result = self.client.execute(
            "SELECT * FROM weather_db.ml_features",
            with_column_types=True,
        )
        columns = [col[0] for col in result[1]]
        df = pd.DataFrame(result[0], columns=columns)

        print(f"✅ Загружено: {len(df)} строк, {len(df.columns)} колонок")
        return df

    # ==========================================
    # 2. EDA
    # ==========================================
    def eda(self, df: pd.DataFrame):
        print("\n" + "=" * 60)
        print("📊 EDA (Первичный анализ)")
        print("=" * 60)

        # 1. Карта температур
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(
            df['longitude'], df['latitude'],
            c=df['temp_max_norm'], cmap='coolwarm', alpha=0.5,
        )
        fig.colorbar(scatter, label='Normalized Temperature')
        ax.set_title('Температурная карта')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        self._save_plot_to_s3(fig, 'eda/temperature_map.png')

        # 2. Температура по месяцам
        df = df.copy()
        df['month'] = pd.to_datetime(df['date']).dt.month
        fig, ax = plt.subplots(figsize=(12, 6))
        for month in range(1, 13):
            subset = df[df['month'] == month]
            if len(subset) > 0:
                ax.hist(
                    subset['temp_max_norm'], bins=30,
                    alpha=0.5, label=f'Month {month}',
                )
        ax.legend()
        ax.set_title('Распределение температур по месяцам')
        self._save_plot_to_s3(fig, 'eda/temperature_by_month.png')

        # 3. Корреляционная матрица
        fig, ax = plt.subplots(figsize=(10, 8))
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()
        im = ax.imshow(corr, cmap='coolwarm', aspect='auto')
        fig.colorbar(im)
        ax.set_xticks(range(len(corr)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr)))
        ax.set_yticklabels(corr.columns)
        ax.set_title('Корреляционная матрица')
        self._save_plot_to_s3(fig, 'eda/correlation_matrix.png')

        print("✅ EDA сохранён в S3")

    # ==========================================
    # 3. SPLIT
    # ==========================================
    def split_data(self, df: pd.DataFrame) -> tuple:
        print("\n" + "=" * 60)
        print("✂️ РАЗДЕЛЕНИЕ ДАННЫХ")
        print("=" * 60)

        df_sorted = df.sort_values(['latitude', 'longitude']).reset_index(drop=True)
        n = len(df_sorted)
        train_end = int(0.7 * n)
        valid_end = int(0.85 * n)

        train = df_sorted.iloc[:train_end]
        valid = df_sorted.iloc[train_end:valid_end]
        test = df_sorted.iloc[valid_end:]

        print(f"Train: {len(train)} ({len(train)/n*100:.1f}%)")
        print(f"Valid: {len(valid)} ({len(valid)/n*100:.1f}%)")
        print(f"Test:  {len(test)} ({len(test)/n*100:.1f}%)")

        return train, valid, test

    # ==========================================
    # 4. BASELINE
    # ==========================================
    def train_baseline(self, train, valid, test) -> dict:
        print("\n" + "=" * 60)
        print("🧠 BASELINE MLP")
        print("=" * 60)

        feature_cols = [
            'lat_norm', 'lon_norm',
            'month_sin', 'month_cos',
            'weekday_sin', 'weekday_cos',
        ]
        target_col = 'temp_max_norm'

        X_train = torch.tensor(train[feature_cols].values, dtype=torch.float32)
        y_train = torch.tensor(train[target_col].values, dtype=torch.float32).reshape(-1, 1)
        X_valid = torch.tensor(valid[feature_cols].values, dtype=torch.float32)
        y_valid = torch.tensor(valid[target_col].values, dtype=torch.float32).reshape(-1, 1)
        X_test = torch.tensor(test[feature_cols].values, dtype=torch.float32)
        y_test = torch.tensor(test[target_col].values, dtype=torch.float32).reshape(-1, 1)

        model = WeatherMLP(
            input_dim=len(feature_cols),
            hidden_dims=[64, 128, 64],
            output_dim=1,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        history = {'train_loss': [], 'valid_loss': []}

        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                y_valid_pred = model(X_valid)
                valid_loss = loss_fn(y_valid_pred, y_valid).item()

            history['train_loss'].append(loss.item())
            history['valid_loss'].append(valid_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: train={loss.item():.4f}, valid={valid_loss:.4f}")

        # Оценка на test
        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test)
            test_loss = loss_fn(y_test_pred, y_test).item()
            test_mae = torch.mean(torch.abs(y_test_pred - y_test)).item()
            ss_res = torch.sum((y_test - y_test_pred) ** 2)
            ss_tot = torch.sum((y_test - y_test.mean()) ** 2)
            test_r2 = (1 - ss_res / ss_tot).item()

        print(f"\n📊 Test: loss={test_loss:.4f}, MAE={test_mae:.4f}, R²={test_r2:.4f}")

        # График loss
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history['train_loss'], label='Train')
        ax.plot(history['valid_loss'], label='Valid')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_title('Loss curves (baseline)')
        self._save_plot_to_s3(fig, 'baseline_loss.png')

        # Сохраняем метрики
        metrics = {
            'test_loss': test_loss,
            'test_mae': test_mae,
            'test_r2': test_r2,
        }
        return metrics

    # ==========================================
    # 5. DEPTH × WIDTH
    # ==========================================
    def experiment_depth_width(self, train, valid) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("🔬 ЭКСПЕРИМЕНТЫ: ГЛУБИНА × ШИРИНА")
        print("=" * 60)

        feature_cols = [
            'lat_norm', 'lon_norm',
            'month_sin', 'month_cos',
            'weekday_sin', 'weekday_cos',
        ]
        target_col = 'temp_max_norm'

        X_train = torch.tensor(train[feature_cols].values, dtype=torch.float32)
        y_train = torch.tensor(train[target_col].values, dtype=torch.float32).reshape(-1, 1)
        X_valid = torch.tensor(valid[feature_cols].values, dtype=torch.float32)
        y_valid = torch.tensor(valid[target_col].values, dtype=torch.float32).reshape(-1, 1)

        results = []
        configs = [
            {'depth': 1, 'width': 32},
            {'depth': 1, 'width': 64},
            {'depth': 2, 'width': 64},
            {'depth': 2, 'width': 128},
            {'depth': 3, 'width': 64},
            {'depth': 3, 'width': 128},
            {'depth': 4, 'width': 64},
        ]

        for config in configs:
            hidden_dims = [config['width']] * config['depth']

            model = WeatherMLP(
                input_dim=len(feature_cols),
                hidden_dims=hidden_dims,
                output_dim=1,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            for epoch in range(EPOCHS):
                model.train()
                optimizer.zero_grad()
                y_pred = model(X_train)
                loss = loss_fn(y_pred, y_train)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                y_valid_pred = model(X_valid)
                valid_loss = loss_fn(y_valid_pred, y_valid).item()
                mae = torch.mean(torch.abs(y_valid_pred - y_valid)).item()
                ss_res = torch.sum((y_valid - y_valid_pred) ** 2)
                ss_tot = torch.sum((y_valid - y_valid.mean()) ** 2)
                r2 = (1 - ss_res / ss_tot).item()

            results.append({
                'depth': config['depth'],
                'width': config['width'],
                'params': sum(p.numel() for p in model.parameters()),
                'valid_loss': valid_loss,
                'mae': mae,
                'r2': r2,
            })

            print(f"Depth={config['depth']}, Width={config['width']}: "
                  f"loss={valid_loss:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        df_results = pd.DataFrame(results)

        # Сохраняем CSV в S3
        csv_str = df_results.to_csv(index=False)
        s3_key = f"{REPORTS_PREFIX}/experiments/depth_width_results.csv"
        self.s3_hook.load_string(
            string_data=csv_str,
            key=s3_key,
            bucket_name=BUCKET,
            replace=True,
        )
        print(f"  💾 s3://{BUCKET}/{s3_key}")

        return df_results

    # ==========================================
    # 6. ГРАДИЕНТЫ
    # ==========================================
    def extract_gradients(self, train):
        print("\n" + "=" * 60)
        print("📉 ИЗВЛЕЧЕНИЕ ГРАДИЕНТОВ")
        print("=" * 60)

        feature_cols = [
            'lat_norm', 'lon_norm',
            'month_sin', 'month_cos',
            'weekday_sin', 'weekday_cos',
        ]
        target_col = 'temp_max_norm'

        X_train = torch.tensor(train[feature_cols].values, dtype=torch.float32)
        y_train = torch.tensor(train[target_col].values, dtype=torch.float32).reshape(-1, 1)

        model = WeatherMLP(
            input_dim=len(feature_cols),
            hidden_dims=[64, 128, 64],
            output_dim=1,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        x = X_train[0:1].clone().requires_grad_(True)
        output = model(x)
        output.backward()

        gradients = x.grad.detach().numpy().flatten().tolist()

        # Сохраняем в S3
        self._save_json_to_s3(
            {'feature_names': feature_cols, 'gradients': gradients},
            'gradients/gradient_analysis.json',
        )

        return gradients

    # ==========================================
    # 7. АКТИВАЦИИ
    # ==========================================
    def plot_activations(self):
        print("\n" + "=" * 60)
        print("📈 ФУНКЦИИ АКТИВАЦИИ")
        print("=" * 60)

        x = np.linspace(-5, 5, 1000)

        def relu(x): return np.maximum(0, x)
        def sigmoid(x): return 1 / (1 + np.exp(-x))
        def tanh(x): return np.tanh(x)
        def leaky_relu(x, alpha=0.01): return np.where(x > 0, x, alpha * x)
        def elu(x, alpha=1.0): return np.where(x > 0, x, alpha * (np.exp(x) - 1))
        def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

        activations = {
            'ReLU': relu,
            'Sigmoid': sigmoid,
            'Tanh': tanh,
            'LeakyReLU': leaky_relu,
            'ELU': elu,
            'GELU': gelu,
        }

        fig, axes = plt.subplots(2, 3, figsize=(14, 10))
        for i, (name, func) in enumerate(activations.items()):
            ax = axes[i // 3, i % 3]
            ax.plot(x, func(x))
            ax.set_title(name)
            ax.grid(True)

        plt.tight_layout()
        self._save_plot_to_s3(fig, 'activations/activation_functions.png')
        print("✅ Графики активаций сохранены в S3")

    # ==========================================
    # 8. ОСНОВНОЙ МЕТОД
    # ==========================================
    def run(self):
        print("\n" + "=" * 60)
        print("🚀 ЗАПУСК ИССЛЕДОВАНИЯ")
        print("=" * 60)

        df = self.load_data()
        self.eda(df)
        train, valid, test = self.split_data(df)

        baseline_metrics = self.train_baseline(train, valid, test)
        df_results = self.experiment_depth_width(train, valid)
        self.extract_gradients(train)
        self.plot_activations()

        # Итоговые метрики
        best_config = df_results.loc[df_results['valid_loss'].idxmin()].to_dict()
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'baseline': baseline_metrics,
            'best_config': best_config,
        }
        self._save_json_to_s3(metrics, 'metrics.json')

        print("\n" + "=" * 60)
        print("✅ ИССЛЕДОВАНИЕ ЗАВЕРШЕНО")
        print(f"📁 Результаты: s3://{BUCKET}/{REPORTS_PREFIX}/")
        print("=" * 60)


if __name__ == "__main__":
    research = MLResearch()
    research.run()