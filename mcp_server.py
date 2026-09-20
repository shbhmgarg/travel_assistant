import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("travel-tools")

CITY_COORDINATES = {
    "singapore": {"lat": 1.3521, "lon": 103.8198},
}

WEATHER_CODE_DESCRIPTIONS = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
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
        "time-sensitive weather question. If the tool returns an "
        "'error' field, relay that honestly instead of guessing the "
        "weather."
    )
)
async def get_weather_forecast(city: str, days: int = 3) -> dict:
    city_key = city.strip().lower()
    coords = CITY_COORDINATES.get(city_key)
    if coords is None:
        return {
            "error": (
                f"No coordinates configured for '{city}'. This assistant "
                f"currently only supports: {', '.join(CITY_COORDINATES)}."
            )
        }

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
            response = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException:
        return {"error": "Weather service timed out. Please try again in a moment."}
    except httpx.HTTPStatusError as exc:
        return {"error": f"Weather service returned an error (HTTP {exc.response.status_code})."}
    except httpx.RequestError as exc:
        return {"error": f"Could not reach the weather service: {exc}"}
    except (KeyError, ValueError) as exc:
        return {"error": f"Weather service returned an unexpected response format: {exc}"}

    try:
        current = data["current"]
        daily = data["daily"]

        forecast = []
        for i in range(len(daily["time"])):
            forecast.append({
                "date": daily["time"][i],
                "high_c": daily["temperature_2m_max"][i],
                "low_c": daily["temperature_2m_min"][i],
                "conditions": describe_weather_code(daily["weather_code"][i]),
                "precipitation_chance_pct": daily["precipitation_probability_max"][i],
                "precipitation_mm": daily["precipitation_sum"][i],
                "wind_kmh": daily["wind_speed_10m_max"][i],
                "sunrise": daily["sunrise"][i],
                "sunset": daily["sunset"][i],
            })

        return {
            "city": city,
            "current": {
                "temp_c": current["temperature_2m"],
                "humidity_pct": current["relative_humidity_2m"],
                "conditions": describe_weather_code(current["weather_code"]),
                "wind_kmh": current["wind_speed_10m"],
            },
            "forecast": forecast,
            "source": "Open-Meteo (api.open-meteo.com)",
        }
    except (KeyError, IndexError) as exc:
        return {"error": f"Weather service response was missing expected data: {exc}"}


@mcp.tool(
    description=(
        "Convert an amount of money from one currency to another using "
        "live, current exchange rates. Use this for any budget or "
        "currency conversion question. If the tool returns an 'error' "
        "field, relay that honestly instead of guessing a rate."
    )
)
async def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    if amount < 0:
        return {"error": "Amount must be zero or positive."}

    from_code = from_currency.strip().upper()
    to_code = to_currency.strip().upper()

    params = {"amount": amount, "from": from_code, "to": to_code}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("https://api.frankfurter.dev/v1/latest", params=params)
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException:
        return {"error": "Currency service timed out. Please try again in a moment."}
    except httpx.HTTPStatusError as exc:
        return {
            "error": (
                f"Currency service returned an error (HTTP {exc.response.status_code}). "
                f"Check that '{from_code}' and '{to_code}' are valid currency codes."
            )
        }
    except httpx.RequestError as exc:
        return {"error": f"Could not reach the currency service: {exc}"}
    except ValueError as exc:
        return {"error": f"Currency service returned an unexpected response format: {exc}"}

    try:
        converted_amount = data["rates"][to_code]
    except KeyError:
        return {"error": f"Currency service did not return a rate for '{to_code}'."}

    return {
        "amount": amount,
        "from_currency": from_code,
        "to_currency": to_code,
        "converted_amount": converted_amount,
        "rate_date": data.get("date"),
        "source": "Frankfurter (api.frankfurter.dev)",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")