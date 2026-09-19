"""WeatherAPI.com tool: forecast lookup used by Destination Agent (to flag
weather risk when ranking destinations) and Critic (to detect weather
disruptions against the built itinerary).
"""
from __future__ import annotations

import httpx
from langchain_core.tools import tool

from config import WEATHERAPI_KEY

WEATHERAPI_FORECAST_URL = "https://api.weatherapi.com/v1/forecast.json"

# Kerala hill stations get heavy monsoon rain June-Sept; used as a
# deterministic offline fallback so the demo is reproducible without a key.
_MONSOON_RISK_DESTINATIONS = {"Munnar", "Thekkady", "Vagamon", "Wayanad"}


@tool(parse_docstring=True)
def get_weather_forecast(destination: str, day_offset: int = 0) -> dict:
    """Get the weather forecast for a destination N days from now.

    Args:
        destination: Destination name, e.g. "Munnar".
        day_offset: Days from today (0 = today, up to 13).

    Returns:
        dict with condition, rain_risk (bool), summary, source.
    """
    if WEATHERAPI_KEY:
        try:
            resp = httpx.get(
                WEATHERAPI_FORECAST_URL,
                params={
                    "key": WEATHERAPI_KEY,
                    "q": f"{destination}, India",
                    "days": min(max(day_offset + 1, 1), 14),
                    "aqi": "no",
                    "alerts": "no",
                },
                timeout=5.0,
            )
            resp.raise_for_status()
            data = resp.json()
            forecast_days = data.get("forecast", {}).get("forecastday", [])
            if forecast_days:
                idx = min(day_offset, len(forecast_days) - 1)
                day = forecast_days[idx]["day"]
                condition = day["condition"]["text"]
                rain_chance = day.get("daily_chance_of_rain", 0)
                return {
                    "condition": condition,
                    "rain_risk": rain_chance >= 50,
                    "summary": f"{condition}, {rain_chance}% chance of rain",
                    "source": "weatherapi",
                }
        except Exception:
            pass

    # Deterministic offline fallback
    risky = destination in _MONSOON_RISK_DESTINATIONS
    return {
        "condition": "Partly cloudy",
        "rain_risk": risky,
        "summary": (
            f"{destination} has monsoon rain risk in this season"
            if risky
            else f"{destination} generally clear this season"
        ),
        "source": "seed_climate_default",
    }


@tool(parse_docstring=True)
def check_weather_disruptions(destinations: list[str]) -> dict:
    """Cross-reference WeatherAPI.com rain risk against itinerary destinations.

    Live WeatherAPI rain_risk is reported; the seed monsoon default is
    informational only so offline Kerala demos are not forced into a replan.

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
                    "live": forecast.get("source") == "weatherapi",
                }
            )
    return {"issues": issues}
