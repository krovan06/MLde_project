from kafka import KafkaConsumer
import psycopg2
import json
import os
import logging
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class WeatherConsumer:
    """Consumer стриминга MET Norway → PostgreSQL."""

    def __init__(self):
        # Kafka
        kafka_bootstrap = os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092')
        topic = os.getenv('KAFKA_TOPIC', 'weather_hourly_raw')
        group_id = os.getenv('KAFKA_GROUP', 'weather_stream_group')

        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=kafka_bootstrap,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id=group_id,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')) # type: ignore
        )

        # PostgreSQL
        self.conn = psycopg2.connect(
            host=os.getenv('PG_HOST', 'postgres_stream'),
            port=int(os.getenv('PG_PORT', 5432)),
            user=os.getenv('PG_USER', 'stream'),
            password=os.getenv('PG_PASSWORD', 'stream'),
            database=os.getenv('PG_DATABASE', 'weather_stream')
        )

        logger.info(f"✅ Подключено к Kafka: {kafka_bootstrap}, topic: {topic}")
        logger.info(f"✅ Подключено к PostgreSQL: {os.getenv('PG_HOST')}/{os.getenv('PG_DATABASE')}")

    def _parse_timeseries(self, data: dict) -> list:
        """Парсит JSON MET Norway в плоские строки."""
        rows = []
        timeseries = data.get('properties', {}).get('timeseries', [])

        for entry in timeseries:
            ts = entry.get('time')
            details = entry.get('data', {}).get('instant', {}).get('details', {})

            # Осадки на следующий час
            next_1h = entry.get('data', {}).get('next_1_hours', {})
            precipitation = next_1h.get('details', {}).get('precipitation_amount', 0)

            rows.append({
                'timestamp': ts,
                'latitude': 43.0,
                'longitude': 131.0,
                'temperature': details.get('air_temperature'),
                'wind_speed': details.get('wind_speed'),
                'humidity': details.get('relative_humidity'),
                'precipitation': precipitation
            })

        return rows

    def _upsert(self, rows: list):
        """UPSERT в PostgreSQL (обновление при конфликте)."""
        cursor = self.conn.cursor()

        query = """
            INSERT INTO weather_realtime
                (timestamp, latitude, longitude, temperature, wind_speed, humidity, precipitation)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (timestamp, latitude, longitude)
            DO UPDATE SET
                temperature = EXCLUDED.temperature,
                wind_speed = EXCLUDED.wind_speed,
                humidity = EXCLUDED.humidity,
                precipitation = EXCLUDED.precipitation,
                ingested_at = NOW()
        """

        for row in rows:
            cursor.execute(query, (
                row['timestamp'],
                row['latitude'],
                row['longitude'],
                row['temperature'],
                row['wind_speed'],
                row['humidity'],
                row['precipitation']
            ))

        self.conn.commit()
        cursor.close()

    def run(self):
        """Бесконечный цикл чтения из Kafka."""
        logger.info("🚀 Consumer запущен. Ждём сообщения из Kafka...")

        try:
            for message in self.consumer:
                logger.info(f"📥 Получено сообщение (partition={message.partition}, offset={message.offset})")

                # 1. Парсим JSON
                rows = self._parse_timeseries(message.value)
                if not rows:
                    logger.warning("⚠️ Пустой timeseries — пропускаем")
                    continue

                # 2. UPSERT в PostgreSQL
                self._upsert(rows)
                logger.info(f"✅ UPSERT в PostgreSQL: {len(rows)} строк")

        except KeyboardInterrupt:
            logger.info("⛔ Consumer остановлен пользователем")
        except Exception as e:
            logger.error(f"❌ Ошибка Consumer: {e}")
            raise
        finally:
            self.consumer.close()
            self.conn.close()


if __name__ == "__main__":
    consumer = WeatherConsumer()
    consumer.run()