"""
Tools for Planning Agent (Task 3.3).
Implements flight search, weather API, and attractions lookup.
"""
import os
import time
import random
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import httpx

logger = logging.getLogger(__name__)


class AgentTools:
    """Collection of tools for the planning agent"""

    def __init__(self):
        """Initialize agent tools"""
        self.weather_api_key = os.getenv("WEATHER_API_KEY", "")
        self.timeout = 5.0  # Tool timeout in seconds

    async def search_flights(
        self,
        origin: str,
        destination: str,
        date: str,
        max_price: Optional[float] = None,
        currency: str = "USD"
    ) -> Dict[str, Any]:
        """
        Search for flights between origin and destination.

        Args:
            origin: Departure city/airport code
            destination: Arrival city/airport code
            date: Departure date (YYYY-MM-DD)
            max_price: Maximum price filter
            currency: Currency code

        Returns:
            Dict with flights list and metadata
        """
        start_time = time.time()

        try:
            logger.info(f"🛫 Searching flights: {origin} → {destination} on {date}")

            # Mock flight search (replace with real API in production)
            await self._simulate_api_call(0.5, 1.5)

            # Generate mock flight data
            flights = self._generate_mock_flights(
                origin, destination, date, max_price, currency
            )

            execution_time = (time.time() - start_time) * 1000

            logger.info(f"✅ Found {len(flights)} flights in {execution_time:.1f}ms")

            return {
                "success": True,
                "flights": flights,
                "count": len(flights),
                "search_params": {
                    "origin": origin,
                    "destination": destination,
                    "date": date,
                    "currency": currency
                },
                "execution_time_ms": execution_time
            }

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"❌ Flight search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": execution_time
            }

    async def get_weather_forecast(
        self,
        location: str,
        start_date: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get weather forecast for a location.

        Args:
            location: City name or coordinates
            start_date: Start date (YYYY-MM-DD)
            days: Number of days to forecast

        Returns:
            Dict with forecast data
        """
        start_time = time.time()

        try:
            logger.info(f"🌤️  Getting weather for {location} ({days} days)")

            # Use real weather API if key is available
            if self.weather_api_key:
                forecast = await self._fetch_real_weather(location, start_date, days)
            else:
                # Mock weather data
                await self._simulate_api_call(0.3, 0.8)
                forecast = self._generate_mock_weather(location, start_date, days)

            execution_time = (time.time() - start_time) * 1000

            logger.info(f"✅ Weather forecast retrieved in {execution_time:.1f}ms")

            return {
                "success": True,
                "location": location,
                "forecast": forecast,
                "days": len(forecast),
                "execution_time_ms": execution_time
            }

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"❌ Weather fetch failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": execution_time
            }

    async def search_attractions(
        self,
        location: str,
        categories: Optional[List[str]] = None,
        min_rating: float = 4.0,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Search for tourist attractions in a location.

        Args:
            location: City name
            categories: Filter by categories (museum, beach, park, etc)
            min_rating: Minimum rating (0-5)
            max_results: Maximum number of results

        Returns:
            Dict with attractions list
        """
        start_time = time.time()

        try:
            logger.info(f"🎭 Searching attractions in {location}")

            # Mock attractions search (replace with real API like Google Places)
            await self._simulate_api_call(0.4, 1.0)

            attractions = self._generate_mock_attractions(
                location, categories, min_rating, max_results
            )

            execution_time = (time.time() - start_time) * 1000

            logger.info(f"✅ Found {len(attractions)} attractions in {execution_time:.1f}ms")

            return {
                "success": True,
                "location": location,
                "attractions": attractions,
                "count": len(attractions),
                "execution_time_ms": execution_time
            }

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"❌ Attractions search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": execution_time
            }

    # Helper methods

    async def _simulate_api_call(self, min_delay: float, max_delay: float):
        """Simulate API call latency"""
        import asyncio
        delay = random.uniform(min_delay, max_delay)
        await asyncio.sleep(delay)

    def _generate_mock_flights(
        self,
        origin: str,
        destination: str,
        date: str,
        max_price: Optional[float],
        currency: str
    ) -> List[Dict[str, Any]]:
        """Generate mock flight data"""
        airlines = ["Air NZ", "Qantas", "Emirates", "Singapore Airlines", "Cathay Pacific"]

        flights = []
        base_price = 300 if currency == "USD" else 450 if currency == "NZD" else 300

        for i in range(random.randint(3, 6)):
            price = base_price + random.randint(-100, 200)

            if max_price and price > max_price:
                continue

            departure_hour = random.randint(6, 20)
            duration = random.uniform(2.5, 14.0)

            flights.append({
                "airline": random.choice(airlines),
                "flight_number": f"{random.choice(['NZ', 'QF', 'EK', 'SQ'])}{random.randint(100, 999)}",
                "departure_time": f"{date}T{departure_hour:02d}:{random.randint(0, 59):02d}:00",
                "arrival_time": self._add_hours(date, departure_hour, duration),
                "price": round(price, 2),
                "currency": currency,
                "duration_hours": round(duration, 1),
                "stops": random.choice([0, 0, 0, 1])  # Most flights are direct
            })

        return sorted(flights, key=lambda x: x["price"])

    def _generate_mock_weather(
        self,
        location: str,
        start_date: str,
        days: int
    ) -> List[Dict[str, Any]]:
        """Generate mock weather forecast"""
        conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Rainy"]
        base_temp = 18 + random.randint(-5, 10)

        forecast = []
        current_date = datetime.strptime(start_date, "%Y-%m-%d")

        for i in range(days):
            temp = base_temp + random.randint(-3, 5)
            condition = random.choice(conditions)

            forecast.append({
                "date": (current_date + timedelta(days=i)).strftime("%Y-%m-%d"),
                "temperature_celsius": temp,
                "temperature_fahrenheit": round(temp * 9/5 + 32, 1),
                "condition": condition,
                "precipitation_chance": 0.1 if "Sunny" in condition else 0.7 if "Rain" in condition else 0.3,
                "humidity": random.randint(50, 85),
                "wind_speed_kmh": random.randint(5, 30)
            })

        return forecast

    def _generate_mock_attractions(
        self,
        location: str,
        categories: Optional[List[str]],
        min_rating: float,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Generate mock attractions"""
        # Sample attractions by city
        attractions_db = {
            "Auckland": [
                {"name": "Sky Tower", "category": "landmark", "rating": 4.5, "price": "$$"},
                {"name": "Auckland War Memorial Museum", "category": "museum", "rating": 4.6, "price": "$"},
                {"name": "Waiheke Island", "category": "nature", "rating": 4.7, "price": "$$"},
                {"name": "Mission Bay Beach", "category": "beach", "rating": 4.4, "price": "Free"},
                {"name": "Auckland Domain", "category": "park", "rating": 4.5, "price": "Free"},
                {"name": "Kelly Tarlton's Sea Life Aquarium", "category": "attraction", "rating": 4.2, "price": "$$"},
            ],
            "Wellington": [
                {"name": "Te Papa Museum", "category": "museum", "rating": 4.7, "price": "Free"},
                {"name": "Wellington Cable Car", "category": "landmark", "rating": 4.5, "price": "$"},
                {"name": "Zealandia", "category": "nature", "rating": 4.6, "price": "$$"},
                {"name": "Oriental Bay", "category": "beach", "rating": 4.3, "price": "Free"},
            ]
        }

        # Get attractions for location or use generic ones
        base_attractions = attractions_db.get(location, attractions_db["Auckland"])

        # Filter by rating and categories
        filtered = []
        for attr in base_attractions:
            if attr["rating"] < min_rating:
                continue
            if categories and attr["category"] not in categories:
                continue

            filtered.append({
                **attr,
                "description": f"A popular {attr['category']} in {location}",
                "price_range": attr["price"],
                "estimated_duration_hours": random.uniform(1.0, 4.0)
            })

        return filtered[:max_results]

    def _add_hours(self, date: str, hour: int, duration_hours: float) -> str:
        """Add hours to a datetime string"""
        dt = datetime.strptime(f"{date}T{hour:02d}:00:00", "%Y-%m-%dT%H:%M:%S")
        dt += timedelta(hours=duration_hours)
        return dt.strftime("%Y-%m-%dT%H:%M:%S")

    async def _fetch_real_weather(
        self,
        location: str,
        start_date: str,
        days: int
    ) -> List[Dict[str, Any]]:
        """Fetch real weather data from API (if key is available)"""
        # This is a placeholder for real weather API integration
        # Example: OpenWeatherMap, WeatherAPI.com, etc.

        # For now, fall back to mock data
        return self._generate_mock_weather(location, start_date, days)


# Singleton instance
_tools_instance: Optional[AgentTools] = None


def get_agent_tools() -> AgentTools:
    """Get singleton instance of agent tools"""
    global _tools_instance
    if _tools_instance is None:
        _tools_instance = AgentTools()
    return _tools_instance
