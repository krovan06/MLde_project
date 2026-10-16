"""
DAG для обучения ML-модели.
Запускается вручную (schedule=None).
Изолирован от основного ETL.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
from scripts.ml.ml_trainer import MLTrainer


default_args = {
    'owner': 'oleg',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# ==========================================
# ТАСКА: ОБУЧЕНИЕ МОДЕЛИ
# ==========================================
def train_model_task(**context):
    """Обучает MLP и сохраняет в S3."""
    trainer = MLTrainer()
    result = trainer.train(bucket='weather-data')
    
    context['ti'].xcom_push(key='model_key', value=result['model_key'])
    context['ti'].xcom_push(key='metrics', value=result['metrics'])
    
    print(f"✅ Обучение завершено")
    print(f"📦 Модель: {result['model_key']}")
    print(f"📊 MAE: {result['metrics']['mae']:.4f}")
    print(f"📊 R²:  {result['metrics']['r2']:.4f}")
    
    return result


# ==========================================
# DAG
# ==========================================
with DAG(
    dag_id='ml_training_pipeline',
    default_args=default_args,
    description='Обучение ML-модели (ручной запуск)',
    schedule=None,  # ← ручной запуск
    catchup=False,
    tags=['ml', 'training', 's3'],
) as dag:

    start = DummyOperator(task_id='start')

    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model_task,
    )

    end = DummyOperator(task_id='end')

    start >> train >> end #type: ignore