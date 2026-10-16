from airflow import DAG
from airflow.operators.python import PythonOperator #type: ignore
from airflow.operators.dummy import DummyOperator #type: ignore
from airflow.models import Variable
from kafka import KafkaProducer
from datetime import datetime, timedelta
from scripts.metno_loader import MetNoLoader
import json


default_args = {
    'owner': 'oleg',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def fetch_and_produce(**context):
    """Тянет прогноз из MET Norway и отправляет в Kafka."""
    # 1. Собираем User-Agent
    app_name = Variable.get("metno_app_name")
    email = Variable.get("contact_email")
    user_agent = f"{app_name} {email}"

    # 2. Тянем прогноз
    loader = MetNoLoader(user_agent=user_agent)
    data = loader.fetch_forecast(lat=43.0, lon=131.0)

    timeseries = data.get('properties', {}).get('timeseries', [])
    print(f"✅ Данные получены: {len(timeseries)} записей")

    # 3. Отправляем в Kafka
    producer = KafkaProducer(
        bootstrap_servers='kafka:9092',
        enable_idempotence=True,
        acks='all',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    future = producer.send('weather_hourly_raw', value=data)
    metadata = future.get(timeout=10)
    
    producer.flush()
    producer.close()

    print(
        f"✅ Отправлено в Kafka: "
        f"topic={metadata.topic}, "
        f"partition={metadata.partition}, "
        f"offset={metadata.offset}"
    )
    
    return "sent"


with DAG(
    dag_id='metno_stream_pipeline',
    default_args=default_args,
    description='MET Norway 24h прогноз → Kafka (каждые 30 мин)',
    schedule='*/30 * * * *',
    catchup=False,
    max_active_runs=1,
    tags=['metno', 'stream', 'kafka'],
) as dag:

    start = DummyOperator(task_id='start')
    fetch = PythonOperator(task_id='fetch_and_produce', python_callable=fetch_and_produce)
    end = DummyOperator(task_id='end')
    
    start >> fetch >> end #type: ignore