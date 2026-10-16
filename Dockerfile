FROM apache/airflow:2.8.0

# Копируем файл с зависимостями
COPY requirements.txt /requirements.txt

# Устанавливаем зависимости от пользователя airflow (НЕ root)
USER airflow
RUN pip install --no-cache-dir -r /requirements.txt