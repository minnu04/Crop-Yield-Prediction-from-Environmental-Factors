from __future__ import annotations

import os
import importlib
from pathlib import Path
from hashlib import sha256
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import quote_plus
from urllib.request import urlopen


BASE_DIR = Path(__file__).resolve().parents[1]
try:
    dotenv_module = importlib.import_module("dotenv")
    load_dotenv = getattr(dotenv_module, "load_dotenv", None)
    if callable(load_dotenv):
        load_dotenv(BASE_DIR / ".env")
except Exception:
    pass


@dataclass
class WeatherConfig:
    provider: str = os.getenv("WEATHER_PROVIDER", "open-meteo")
    api_key: str = os.getenv("WEATHER_API_KEY", "")
    timeout_seconds: int = int(os.getenv("WEATHER_TIMEOUT_SECONDS", "10"))


def _get_json(url: str, timeout_seconds: int = 10) -> Dict[str, Any]:
    with urlopen(url, timeout=timeout_seconds) as response:
        return __import__("json").loads(response.read().decode("utf-8"))


def _geocode_open_meteo(location: str, timeout_seconds: int) -> Optional[Dict[str, Any]]:
    if not location:
        return None

    encoded = quote_plus(location)
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded}&count=1&language=en&format=json"
    data = _get_json(url, timeout_seconds)

    results = data.get("results") or []
    if not results:
        return None

    first = results[0]
    return {
        "name": first.get("name", location),
        "country": first.get("country", ""),
        "latitude": first.get("latitude"),
        "longitude": first.get("longitude"),
    }


def _open_meteo_forecast(latitude: float, longitude: float, timeout_seconds: int) -> Dict[str, Any]:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        "&daily=temperature_2m_min,temperature_2m_max,precipitation_sum"
        "&forecast_days=1&timezone=auto"
    )
    data = _get_json(url, timeout_seconds)
    daily = data.get("daily") or {}

    rain = (daily.get("precipitation_sum") or [None])[0]
    tmin = (daily.get("temperature_2m_min") or [None])[0]
    tmax = (daily.get("temperature_2m_max") or [None])[0]

    return {
        "forecastRainfall": rain,
        "forecastTempMin": tmin,
        "forecastTempMax": tmax,
    }


def _openweather_forecast(latitude: float, longitude: float, api_key: str, timeout_seconds: int) -> Dict[str, Any]:
    if not api_key:
        raise ValueError("WEATHER_API_KEY is required for openweather provider.")

    url = (
        "https://api.openweathermap.org/data/2.5/forecast"
        f"?lat={latitude}&lon={longitude}&units=metric&appid={api_key}"
    )
    data = _get_json(url, timeout_seconds)
    entries = data.get("list") or []
    if not entries:
        raise ValueError("No forecast entries returned from provider.")

    first_day = [x for x in entries[:8] if "main" in x]
    temps = [x["main"].get("temp") for x in first_day if x["main"].get("temp") is not None]
    rainfall = sum((x.get("rain") or {}).get("3h", 0.0) for x in first_day)

    if not temps:
        raise ValueError("Temperature forecast data missing from provider response.")

    return {
        "forecastRainfall": round(float(rainfall), 2),
        "forecastTempMin": round(min(temps), 2),
        "forecastTempMax": round(max(temps), 2),
    }


def _estimate_fallback_weather(
    location: str,
    latitude: Optional[float],
    longitude: Optional[float],
) -> Dict[str, float]:
    seed_text = f"{location}|{latitude}|{longitude}".lower()
    digest = sha256(seed_text.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)

    # Deterministic estimates keep the demo functional when external weather APIs are unavailable.
    rainfall = 80 + (seed % 240)  # 80-319 mm
    lat = abs(float(latitude)) if latitude is not None else 20.0

    if lat < 12:
        temp_min_base, temp_max_base = 22.0, 34.0
    elif lat < 24:
        temp_min_base, temp_max_base = 19.0, 32.0
    elif lat < 35:
        temp_min_base, temp_max_base = 15.0, 29.0
    else:
        temp_min_base, temp_max_base = 10.0, 24.0

    offset_min = ((seed >> 8) % 7) - 3
    offset_max = ((seed >> 16) % 7) - 3

    temp_min = round(temp_min_base + offset_min * 0.6, 2)
    temp_max = round(max(temp_min + 4.0, temp_max_base + offset_max * 0.8), 2)

    return {
        "forecastRainfall": round(float(rainfall), 2),
        "forecastTempMin": temp_min,
        "forecastTempMax": temp_max,
    }


def fetch_weather_forecast(
    location: str = "",
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> Dict[str, Any]:
    config = WeatherConfig()

    try:
        lat = latitude
        lon = longitude
        resolved_name = location or ""

        if lat is None or lon is None:
            if not location:
                raise ValueError("Provide either location or latitude/longitude.")

            geo = _geocode_open_meteo(location, config.timeout_seconds)
            if not geo:
                raise ValueError("Could not resolve location for weather lookup.")

            lat = geo["latitude"]
            lon = geo["longitude"]
            resolved_name = f"{geo['name']}, {geo['country']}".strip(", ")

        if config.provider.lower() == "openweather":
            weather = _openweather_forecast(lat, lon, config.api_key, config.timeout_seconds)
            source = "openweather"
        else:
            weather = _open_meteo_forecast(lat, lon, config.timeout_seconds)
            source = "open-meteo"

        return {
            "status": "ok",
            "source": source,
            "location": {
                "name": resolved_name or "Custom coordinates",
                "latitude": lat,
                "longitude": lon,
            },
            **weather,
        }
    except Exception as error:
        estimated = _estimate_fallback_weather(location=location, latitude=latitude, longitude=longitude)
        return {
            "status": "fallback",
            "source": "estimated",
            "estimated": True,
            "message": f"Live weather unavailable: {error}. Estimated local conditions applied.",
            "location": {
                "name": location or "Manual mode",
                "latitude": latitude,
                "longitude": longitude,
            },
            **estimated,
        }
