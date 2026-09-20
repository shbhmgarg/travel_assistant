import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("travel-tools")

CITY_COORDINATES = {
    "singapore": {"lat": 1.3521, "lon": 103.8198},
}

WEATHER_CODE_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    95: "Thunderstorm",
}

def describe_weather_code(code) -> str:
    return WEATHER_CODE_DESCRIPTIONS.get(code, f"Unknown conditions (code {code})")


@mcp.tool(
    description=(
        "Get current conditions and a multi-day forecast for a city, "
        "including temperature, humidity, rain/precipitation chance, "
        "wind speed, and sunrise/sunset times. Use this for any "
        "time-sensitive weather question."
    )
)
async def get_weather_forecast(city: str, days: int = 3) -> dict:

    key = city.strip().lower()
    if key not in CITY_COORDINATES:
        return {
            "error": (
                f"No coordinates configured for '{city}'. This demo "
                f"only supports: {', '.join(CITY_COORDINATES)}."
            )
        }

    coords = CITY_COORDINATES[key]
    days = max(1, min(days, 7))

    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
        "daily": (
            "temperature_2m_max,temperature_2m_min,precipitation_probability_max,"
            "precipitation_sum,weather_code,wind_speed_10m_max,sunrise,sunset"
        ),
        "forecast_days": days,
        "timezone": "auto",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return {"error": f"Weather service request failed: {e}"}

    current = data.get("current", {})
    daily = data.get("daily", {})

    forecast = []
    for i, date in enumerate(daily.get("time", [])):
        forecast.append({
            "date": date,
            "conditions": describe_weather_code(daily.get("weather_code", [None])[i]),
            "max_temp_c": daily.get("temperature_2m_max", [None])[i],
            "min_temp_c": daily.get("temperature_2m_min", [None])[i],
            "precipitation_probability_pct": daily.get("precipitation_probability_max", [None])[i],
            "precipitation_total_mm": daily.get("precipitation_sum", [None])[i],
            "max_wind_kmh": daily.get("wind_speed_10m_max", [None])[i],
            "sunrise": daily.get("sunrise", [None])[i],
            "sunset": daily.get("sunset", [None])[i],
        })

    return {
        "city": city,
        "current": {
            "temp_c": current.get("temperature_2m"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "conditions": describe_weather_code(current.get("weather_code")),
            "wind_kmh": current.get("wind_speed_10m"),
        },
        "forecast": forecast,
        "source": "Open-Meteo (https://open-meteo.com/)",
    }


@mcp.tool(
    description=(
        "Convert an amount of money from one currency to another using "
        "live, current exchange rates. Use this for any budget or "
        "currency conversion question."
    )
)
async def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    from_currency = from_currency.strip().upper()
    to_currency = to_currency.strip().upper()

    params = {"amount": amount, "from": from_currency, "to": to_currency}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get("https://api.frankfurter.dev/v1/latest", params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return {"error": f"Currency conversion request failed: {e}"}

    rates = data.get("rates", {})
    if to_currency not in rates:
        return {"error": f"Could not find a rate for {to_currency}. Response: {data}"}

    return {
        "amount": amount,
        "from_currency": from_currency,
        "to_currency": to_currency,
        "converted_amount": rates[to_currency],
        "rate_date": data.get("date"),
        "source": "Frankfurter API (https://www.frankfurter.dev/)",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")