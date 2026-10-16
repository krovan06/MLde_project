from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class S3Manager:
    """
    Управляет S3: архивация staging → history, retention.
    НЕ проверяет хеши, НЕ чистит staging (это делает архивация).
    """

    def __init__(
        self,
        bucket: str = 'weather-data',
        staging_prefix: str = 'staging',
        history_prefix: str = 'history',
        days_to_keep: int = 3,
    ):
        self.bucket = bucket
        self.staging_prefix = staging_prefix
        self.history_prefix = history_prefix
        self.days_to_keep = days_to_keep
        self.s3_hook = S3Hook(aws_conn_id='s3_conn')

    # ==========================================
    # 1. АРХИВАЦИЯ STAGING → HISTORY
    # ==========================================
    def archive_staging(self) -> dict:
        """
        Перекладывает файлы из staging в history с партиционированием по дате.
        Очищает staging.
        """
        staging_files = self.s3_hook.list_keys(
            bucket_name=self.bucket,
            prefix=self.staging_prefix
        )

        if not staging_files:
            logger.info("📭 Staging пуст")
            return {'archived': 0, 'cleared': 0}

        now = datetime.now()
        history_folder = (
            f"{self.history_prefix}/"
            f"year={now.year}/"
            f"month={now.month:02d}/"
            f"day={now.day:02d}/"
        )

        archived = 0
        for staging_key in staging_files:
            # 1. Читаем
            data = self.s3_hook.read_key(key=staging_key, bucket_name=self.bucket)

            # 2. Формируем путь в history
            filename = staging_key.split('/')[-1]
            history_key = f"{history_folder}{filename}"

            # 3. Сохраняем в history
            self.s3_hook.load_string(
                string_data=data,
                key=history_key,
                bucket_name=self.bucket,
                replace=True
            )

            # 4. Удаляем из staging
            self.s3_hook.delete_objects(
                bucket=self.bucket,
                keys=[staging_key]
            )
            archived += 1

        logger.info(f"✅ Архивировано: {archived} файлов")
        return {'archived': archived, 'cleared': archived}

    # ==========================================
    # 2. ОЧИСТКА HISTORY
    # ==========================================
    def clean_history(self) -> dict:
        """Удаляет файлы в history старше N дней."""
        cutoff_date = datetime.now() - timedelta(days=self.days_to_keep)

        all_keys = self.s3_hook.list_keys(
            bucket_name=self.bucket,
            prefix=self.history_prefix
        )

        if not all_keys:
            logger.info("📭 History пуст")
            return {'deleted': 0, 'kept': 0}

        # Группируем по папкам
        folders = {}
        for key in all_keys:
            folder = '/'.join(key.split('/')[:-1])
            if folder not in folders:
                folders[folder] = []
            folders[folder].append(key)

        deleted = 0
        kept = 0

        for folder, files in folders.items():
            try:
                # Парсим путь: history/year=2026/month=10/day=09
                parts = folder.split('/')
                if len(parts) < 4 or not parts[1].startswith('year='):
                    logger.warning(f"⚠️ Старый формат, пропускаем: {folder}")
                    kept += len(files)
                    continue

                year = int(parts[1].split('=')[1])
                month = int(parts[2].split('=')[1])
                day = int(parts[3].split('=')[1])
                folder_date = datetime(year, month, day)

                if folder_date < cutoff_date:
                    self.s3_hook.delete_objects(
                        bucket=self.bucket,
                        keys=files
                    )
                    deleted += len(files)
                else:
                    kept += len(files)
            except (IndexError, ValueError) as e:
                logger.warning(f"⚠️ Не удалось распарсить {folder}: {e}")
                kept += len(files)

        logger.info(f"🧹 History: удалено {deleted}, оставлено {kept}")
        return {'deleted': deleted, 'kept': kept}

    # ==========================================
    # 3. ПОЛНЫЙ ЦИКЛ
    # ==========================================
    def run(self) -> dict:
        """Полный цикл: архивация + retention."""
        archive_result = self.archive_staging()
        clean_result = self.clean_history()
        return {
            'archive': archive_result,
            'clean': clean_result
        }