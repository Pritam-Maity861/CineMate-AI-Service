from typing import Any, Dict, List, Optional
import httpx
from app.config import settings


class NodeAPIClient:
    """Async HTTP client connecting FastMCP tools to CineMate Node.js REST API."""

    def __init__(self, base_url: str = settings.NODE_API_URL):
        self.base_url = base_url.rstrip("/")

    def _headers(
        self,
        token: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if token:
            clean_token = token.replace("Bearer ", "").strip()
            headers["Authorization"] = f"Bearer {clean_token}"
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    async def search_movies(
        self,
        search: Optional[str] = None,
        genre_id: Optional[str] = None,
        language_id: Optional[str] = None,
        page: int = 1,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Search approved movies with keyword, genre, and language filters."""
        params = {"page": page, "limit": limit}
        if search:
            params["search"] = search
        if genre_id:
            params["genreId"] = genre_id
        if language_id:
            params["languageId"] = language_id

        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{self.base_url}/movie/get-all-movie", params=params)
            res.raise_for_status()
            return res.json().get("data", {})

    async def get_movie_details(self, movie_id: str) -> Dict[str, Any]:
        """Fetch full movie details including synopsis, cast credits, and genres."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{self.base_url}/movie/get-movie/{movie_id}")
            res.raise_for_status()
            return res.json().get("data", {})

    async def list_genres(self) -> List[Dict[str, Any]]:
        """List all available movie genres in the catalog."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{self.base_url}/movie/get-genre")
            res.raise_for_status()
            return res.json().get("data", [])

    async def list_languages(self) -> List[Dict[str, Any]]:
        """List all available movie audio languages."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{self.base_url}/movie/get-language")
            res.raise_for_status()
            return res.json().get("data", [])

    async def list_theaters(self, city: Optional[str] = None) -> List[Dict[str, Any]]:
        """List approved cinemas, cities, and amenities (IMAX, Dolby Atmos, Recliners)."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{self.base_url}/theaters")
            res.raise_for_status()
            theaters = res.json().get("data", [])
            if city:
                return [t for t in theaters if t.get("city", "").lower() == city.lower()]
            return theaters

    async def get_showtimes(
        self,
        movie_id: str,
        date: str,
        theater_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch approved showtimes for a specific movie and date (YYYY-MM-DD)."""
        params = {"movieId": movie_id, "date": date}
        if theater_id:
            params["theaterId"] = theater_id

        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{self.base_url}/showtime", params=params)
            res.raise_for_status()
            return res.json().get("data", [])

    async def get_showtime_seats(self, showtime_id: str) -> Dict[str, Any]:
        """Fetch real-time seat availability map with Redis locks + DB bookings."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{self.base_url}/showtime/{showtime_id}/seats")
            res.raise_for_status()
            return res.json().get("data", {})

    async def create_pending_booking(
        self,
        showtime_id: str,
        seat_ids: List[str],
        auth_token: str,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Lock seats in Redis and create a PENDING booking for 10 minutes."""
        payload = {"showtimeId": showtime_id, "seatIds": seat_ids}
        headers = self._headers(auth_token, idempotency_key)

        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(
                f"{self.base_url}/booking/pending-booking",
                json=payload,
                headers=headers,
            )
            res.raise_for_status()
            return res.json().get("data", {})

    async def get_booking(self, booking_id: str, auth_token: str) -> Dict[str, Any]:
        """Fetch user booking details and countdown expiry."""
        headers = self._headers(auth_token)
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(
                f"{self.base_url}/booking/{booking_id}",
                headers=headers,
            )
            res.raise_for_status()
            return res.json().get("data", {})

    async def list_my_bookings(self, auth_token: str) -> List[Dict[str, Any]]:
        """Fetch current user's booking history."""
        headers = self._headers(auth_token)
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(
                f"{self.base_url}/booking/my-bookings",
                headers=headers,
            )
            res.raise_for_status()
            return res.json().get("data", [])

    async def get_payment_link(
        self,
        booking_id: str,
        auth_token: str,
    ) -> Dict[str, Any]:
        """Generate Razorpay order and return checkoutUrl deep link."""
        payload = {"bookingId": booking_id}
        headers = self._headers(auth_token)

        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(
                f"{self.base_url}/payments/create-order",
                json=payload,
                headers=headers,
            )
            res.raise_for_status()
            return res.json().get("data", {})

    async def get_payment_status(self, booking_id: str, auth_token: str) -> str:
        """Inspect payment confirmation status."""
        booking = await self.get_booking(booking_id, auth_token)
        return booking.get("status", "UNKNOWN")


node_client = NodeAPIClient()

