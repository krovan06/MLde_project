CREATE TABLE IF NOT EXISTS weather_realtime (
    timestamp TIMESTAMPTZ NOT NULL,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    temperature FLOAT,
    wind_speed FLOAT,
    humidity FLOAT,
    precipitation FLOAT,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, latitude, longitude)
);

CREATE INDEX IF NOT EXISTS idx_weather_realtime_timestamp 
ON weather_realtime (timestamp DESC);