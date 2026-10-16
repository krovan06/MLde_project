import requests


class MetNoLoader:
    """Загрузчик почасового прогноза из MET Norway API."""

    BASE_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"

    def __init__(self, user_agent: str):
        self.headers = {"User-Agent": user_agent}

    def fetch_forecast(self, lat: float, lon: float) -> dict:
        """Тянет почасовой прогноз для одной точки."""
        params = {
            "lat": round(lat, 4),
            "lon": round(lon, 4)
        }

        try:
            response = requests.get(
                self.BASE_URL,
                headers=self.headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            print(f"Прогноз получен для {lat}, {lon}")
            return data

        except requests.exceptions.RequestException as e:
            print(f"Прогноз не получен: {e}")
            raise