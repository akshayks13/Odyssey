"""Weather for trip dates from Open-Meteo (free, no key): a real forecast when the dates are close,
and the same dates last year as a seasonal guide when they are not."""
from __future__ import annotations

from datetime import date, timedelta

import httpx
from langchain_core.tools import tool

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_HORIZON_DAYS = 15  # Open-Meteo forecasts up to 16 days ahead

# WMO weather codes, as published by Open-Meteo.
_CONDITIONS = {
    0: "Clear", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle", 56: "Freezing drizzle", 57: "Freezing drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain", 66: "Freezing rain", 67: "Freezing rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Rain showers", 81: "Rain showers", 82: "Violent rain showers", 85: "Snow showers", 86: "Snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with hail",
}

_cache: dict[tuple, list[dict]] = {}


def _is_rainy(rain_mm: float, rain_chance: float | None) -> bool:
    return rain_mm >= 5 or (rain_chance is not None and rain_chance >= 60)


def _fetch(lat: float, lng: float, start: date, end: date, historical: bool) -> list[dict]:
    key = (round(lat, 2), round(lng, 2), start, end, historical)
    if key in _cache:
        return _cache[key]
    fields = "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum"
    if not historical:
        fields += ",precipitation_probability_max"
    for attempt in range(2):  # one retry: a passing network blip should not remove the weather
        try:
            resp = httpx.get(
                ARCHIVE_URL if historical else FORECAST_URL,
                params={"latitude": lat, "longitude": lng, "daily": fields, "timezone": "auto",
                        "start_date": start.isoformat(), "end_date": end.isoformat()},
                timeout=10.0,
            )
            resp.raise_for_status()
            break
        except httpx.HTTPError:
            if attempt:
                raise
    daily = resp.json().get("daily") or {}
    rows = []
    for i, day in enumerate(daily.get("time") or []):
        def at(name):
            values = daily.get(name) or []
            return values[i] if i < len(values) else None

        rain_mm = float(at("precipitation_sum") or 0)
        chance = at("precipitation_probability_max")
        rows.append(
            {
                "date": day,
                "tmin": at("temperature_2m_min"),
                "tmax": at("temperature_2m_max"),
                "rain_mm": round(rain_mm, 1),
                "rain_chance": None if chance is None else int(chance),
                "condition": _CONDITIONS.get(int(at("weather_code") or 0), "Unknown"),
                "rainy": _is_rainy(rain_mm, None if chance is None else float(chance)),
            }
        )
    _cache[key] = rows
    return rows


def _same_day_last_year(d: date) -> date:
    try:
        return d.replace(year=d.year - 1)
    except ValueError:  # 29 Feb
        return d.replace(year=d.year - 1, day=28)


def trip_weather(place: str, start_date: str, days: int = 1) -> dict:
    """Weather for each trip day. Days within the forecast window get the forecast; later days get the
    same calendar day last year, marked as such. Never raises: a failure returns source "unavailable"."""
    from tools.mapbox_api import geocode_location

    try:
        start = date.fromisoformat(start_date) if start_date else date.today()
        geo = geocode_location.invoke({"place_name": place})
        if geo["lat"] is None:
            raise ValueError("unknown place")
        horizon = date.today() + timedelta(days=FORECAST_HORIZON_DAYS)
        trip_days = [start + timedelta(days=i) for i in range(max(1, days))]
        near, far = [d for d in trip_days if d <= horizon], [d for d in trip_days if d > horizon]
        rows: list[dict] = []
        if near:
            rows += [{**r, "source": "forecast"} for r in _fetch(geo["lat"], geo["lng"], near[0], near[-1], False)]
        if far:
            past = _fetch(geo["lat"], geo["lng"], _same_day_last_year(far[0]), _same_day_last_year(far[-1]), True)
            rows += [{**r, "date": d.isoformat(), "source": "last year"} for r, d in zip(past, far)]
    except Exception:  # noqa: BLE001 — weather is advice, never a reason to fail a plan
        return {"source": "unavailable", "days": [], "rainy_dates": [], "summary": "Weather unavailable"}

    rainy = [r["date"] for r in rows if r["rainy"]]
    temps = [(r["tmin"], r["tmax"]) for r in rows if r["tmin"] is not None and r["tmax"] is not None]
    sources = {r["source"] for r in rows}
    overall = "mixed" if len(sources) > 1 else (sources.pop() if sources else "unavailable")
    bits = []
    if temps:
        bits.append(f"{min(t[0] for t in temps):.0f}–{max(t[1] for t in temps):.0f}°C")
    bits.append(f"rain likely on {len(rainy)} of {len(rows)} days" if rainy else "mostly dry")
    if overall != "forecast":
        bits.append("typical for these dates (last year)")
    return {"source": overall, "days": rows, "rainy_dates": rainy, "summary": ", ".join(bits)}


@tool(parse_docstring=True)
def check_weather_disruptions(destinations: list[str], start_date: str = "", days: int = 1) -> dict:
    """Which of these places have rain or a storm on the trip dates.

    Args:
        destinations: Place names to check.
        start_date: First day, YYYY-MM-DD. Defaults to today.
        days: Number of days.

    Returns:
        dict with issues (list of {destination, message, source}).
    """
    issues = []
    for place in destinations:
        weather = trip_weather(place, start_date, days)
        stormy = [r["date"] for r in weather["days"] if "thunder" in r["condition"].lower()]
        if weather["rainy_dates"] or stormy:
            issues.append({"destination": place, "message": weather["summary"] + (f"; storms on {', '.join(stormy)}" if stormy else ""), "source": weather["source"]})
    return {"issues": issues}
