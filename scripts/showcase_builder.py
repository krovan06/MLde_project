from clickhouse_driver import Client as ClickHouseClient
import logging

logger = logging.getLogger(__name__)


class ShowcaseBuilder:
    def __init__(
        self,
        host: str = 'clickhouse',
        port: int = 9000,
        user: str = 'default',
        password: str = 'default',
        database: str = 'weather_db',
    ):
        self.client = ClickHouseClient(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        self.database = database

    # ==========================================
    # 1. ЕЖЕДНЕВНАЯ СТАТИСТИКА ПО РЕГИОНУ
    # ==========================================
    def build_daily_stats(self):
        """Витрина: ежедневная статистика по дням."""
        view_name = 'daily_weather_stats'
        
        query = f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT
                d.date AS date,
                d.year AS year,
                d.month AS month,
                d.day AS day,
                d.weekday AS weekday,
                d.season AS season,
                round(AVG(f.temperature_max), 1) AS avg_temp_max,
                round(AVG(f.temperature_min), 1) AS avg_temp_min,
                MIN(f.temperature_min) AS min_temp,
                MAX(f.temperature_max) AS max_temp,
                round(SUM(f.precipitation), 2) AS total_precipitation,
                round(AVG(f.wind_speed_max), 1) AS avg_wind_speed,
                COUNT(*) AS points_count
            FROM {self.database}.fact_weather f
            JOIN {self.database}.dim_date d ON f.date = d.date
            GROUP BY d.date, d.year, d.month, d.day, d.weekday, d.season
            ORDER BY d.date DESC
        """
        self.client.execute(query)
        logger.info(f"✅ Витрина {view_name} создана")

    # ==========================================
    # 2. ТОП-10 САМЫХ ТЁПЛЫХ ЛОКАЦИЙ
    # ==========================================
    def build_warmest_locations(self):
        """Витрина: топ-10 локаций по средней температуре."""
        view_name = 'warmest_locations'
        
        query = f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT
                l.location_id AS location_id,
                l.latitude AS latitude,
                l.longitude AS longitude,
                l.region AS region,
                l.city_name AS city_name,
                round(AVG(f.temperature_max), 1) AS avg_temp_max,
                MAX(f.temperature_max) AS max_temp,
                MIN(f.temperature_min) AS min_temp,
                COUNT(*) AS days_count
            FROM {self.database}.fact_weather f
            JOIN {self.database}.dim_location l ON f.location_id = l.location_id
            GROUP BY l.location_id, l.latitude, l.longitude, l.region, l.city_name
            ORDER BY avg_temp_max DESC
            LIMIT 10
        """
        self.client.execute(query)
        logger.info(f"✅ Витрина {view_name} создана")

    # ==========================================
    # 3. ТРЕНД ОСАДКОВ
    # ==========================================
    def build_precipitation_trend(self):
        """Витрина: тренд осадков по дням с накоплением."""
        view_name = 'precipitation_trend'
        
        query = f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT
                d.date AS date,
                d.season AS season,
                round(SUM(f.precipitation), 2) AS total_precipitation,
                round(AVG(f.precipitation), 2) AS avg_precipitation,
                MAX(f.precipitation) AS max_precipitation,
                round(SUM(SUM(f.precipitation)) OVER (ORDER BY d.date), 2) AS cumulative_precipitation
            FROM {self.database}.fact_weather f
            JOIN {self.database}.dim_date d ON f.date = d.date
            GROUP BY d.date, d.season
            ORDER BY d.date
        """
        self.client.execute(query)
        logger.info(f"✅ Витрина {view_name} создана")

    # ==========================================
    # 4. НОВАЯ ВИТРИНА: СТАТИСТИКА ПО СЕЗОНАМ
    # ==========================================
    def build_season_stats(self):
        """Витрина: статистика по сезонам (использует dim_date)."""
        view_name = 'season_stats'
        
        query = f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT
                d.season AS season,
                round(AVG(f.temperature_max), 1) AS avg_temp_max,
                round(AVG(f.temperature_min), 1) AS avg_temp_min,
                round(SUM(f.precipitation), 2) AS total_precipitation,
                round(AVG(f.wind_speed_max), 1) AS avg_wind_speed,
                COUNT(*) AS days_count
            FROM {self.database}.fact_weather f
            JOIN {self.database}.dim_date d ON f.date = d.date
            GROUP BY d.season
            ORDER BY d.season
        """
        self.client.execute(query)
        logger.info(f"✅ Витрина {view_name} создана")

    # ==========================================
    # 5. НОВАЯ ВИТРИНА: СТАТИСТИКА ПО РЕГИОНАМ
    # ==========================================
    def build_region_stats(self):
        """Витрина: статистика по регионам (использует dim_location)."""
        view_name = 'region_stats'
        
        query = f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT
                l.region AS region,
                round(AVG(f.temperature_max), 1) AS avg_temp_max,
                round(AVG(f.temperature_min), 1) AS avg_temp_min,
                round(SUM(f.precipitation), 2) AS total_precipitation,
                round(AVG(f.wind_speed_max), 1) AS avg_wind_speed,
                COUNT(DISTINCT l.location_id) AS locations_count,
                COUNT(*) AS records_count
            FROM {self.database}.fact_weather f
            JOIN {self.database}.dim_location l ON f.location_id = l.location_id
            GROUP BY l.region
            ORDER BY l.region
        """
        self.client.execute(query)
        logger.info(f"✅ Витрина {view_name} создана")