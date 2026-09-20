"""OpenWeatherMap current conditions and 5-day forecast."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import httpx
from langchain_core.tools import tool

from config import OPENWEATHER_API_KEY

WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

_WET = {"Rain", "Thunderstorm", "Drizzle", "Snow"}


def _seed_forecast(destination: str) -> dict:
    return {
        "condition": "Partly cloudy",
        "rain_risk": False,
        "summary": f"{destination}: live weather unavailable, assuming fair conditions",
        "source": "seed_default",
    }


def _from_current(data: dict) -> dict:
    weather = (data.get("weather") or [{}])[0]
    condition = weather.get("description") or weather.get("main") or "Unknown"
    main = weather.get("main") or ""
    rain_mm = float((data.get("rain") or {}).get("1h") or (data.get("rain") or {}).get("3h") or 0)
    rain_risk = main in _WET or rain_mm > 0
    temp = (data.get("main") or {}).get("temp")
    temp_bit = f", {round(temp)}°C" if temp is not None else ""
    return {
        "condition": condition,
        "rain_risk": rain_risk,
        "summary": f"{condition}{temp_bit}",
        "source": "openweathermap",
    }


def _from_forecast_slots(slots: list[dict], day_offset: int) -> dict | None:
    target = date.today() + timedelta(days=max(0, day_offset))
    matching = []
    for slot in slots:
        dt_txt = slot.get("dt_txt")
        if not dt_txt:
            ts = slot.get("dt")
            if ts is None:
                continue
            slot_day = datetime.fromtimestamp(int(ts), tz=timezone.utc).date()
        else:
            slot_day = datetime.strptime(dt_txt[:10], "%Y-%m-%d").date()
        if slot_day == target:
            matching.append(slot)
    if not matching:
        # Clamp to the last available calendar day in the 5-day window.
        by_day: dict[date, list[dict]] = {}
        for slot in slots:
            dt_txt = slot.get("dt_txt")
            if dt_txt:
                slot_day = datetime.strptime(dt_txt[:10], "%Y-%m-%d").date()
                by_day.setdefault(slot_day, []).append(slot)
        if not by_day:
            return None
        last_day = max(by_day)
        matching = by_day[last_day]
    pops = [float(s.get("pop") or 0) for s in matching]
    rain_chance = int(round(max(pops) * 100)) if pops else 0
    # Prefer a midday slot for the condition text.
    chosen = matching[min(len(matching) // 2, len(matching) - 1)]
    weather = (chosen.get("weather") or [{}])[0]
    condition = weather.get("description") or weather.get("main") or "Unknown"
    main = weather.get("main") or ""
    rain_risk = rain_chance >= 50 or main in _WET
    temp = (chosen.get("main") or {}).get("temp")
    temp_bit = f", {round(temp)}°C" if temp is not None else ""
    return {
        "condition": condition,
        "rain_risk": rain_risk,
        "summary": f"{condition}{temp_bit}, {rain_chance}% chance of rain",
        "source": "openweathermap",
    }


@tool(parse_docstring=True)
def get_weather_forecast(destination: str, day_offset: int = 0) -> dict:
    """Get the weather forecast for a destination N days from now.

    Args:
        destination: Destination name, e.g. "Munnar".
        day_offset: Days from today (0 = today, up to 4 on the free 5-day forecast).

    Returns:
        dict with condition, rain_risk (bool), summary, source.
    """
    if OPENWEATHER_API_KEY:
        from tools.mapbox_api import geocode_location

        geo = geocode_location.invoke({"place_name": destination})
        params = {
            "lat": geo["lat"],
            "lon": geo["lng"],
            "units": "metric",
            "appid": OPENWEATHER_API_KEY,
        }
        try:
            if day_offset <= 0:
                resp = httpx.get(WEATHER_URL, params=params, timeout=5.0)
                resp.raise_for_status()
                return _from_current(resp.json())
            resp = httpx.get(FORECAST_URL, params=params, timeout=5.0)
            resp.raise_for_status()
            parsed = _from_forecast_slots(resp.json().get("list") or [], day_offset)
            if parsed:
                return parsed
        except Exception:
            pass

    return _seed_forecast(destination)


@tool(parse_docstring=True)
def check_weather_disruptions(destinations: list[str]) -> dict:
    """Cross-reference OpenWeatherMap rain risk against itinerary destinations.

    Live rain_risk is reported; the offline default is informational only
    so missing OpenWeather keys do not force a replan.

    Args:
        destinations: Destination names appearing in the itinerary.

    Returns:
        dict with issues (list of {destination, message, source, live}).
    """
    issues: list[dict] = []
    for dest in destinations:
        forecast = get_weather_forecast.invoke({"destination": dest, "day_offset": 0})
        if forecast.get("rain_risk"):
            issues.append(
                {
                    "destination": dest,
                    "message": forecast.get("summary", "rain risk"),
                    "source": forecast.get("source"),
                    "live": forecast.get("source") == "openweathermap",
                }
            )
    return {"issues": issues}
