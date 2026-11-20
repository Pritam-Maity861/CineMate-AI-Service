from typing import List, Optional
from fastmcp import FastMCP
from app.mcp.tools import node_client

# Initialize FastMCP Server
mcp = FastMCP("CineMate-MCP-Server")


@mcp.tool()
async def search_movies(
    search: Optional[str] = None,
    genre_id: Optional[str] = None,
    language_id: Optional[str] = None,
    page: int = 1,
    limit: int = 10,
) -> dict:
    """Search approved movies with keyword, genre, and language filters."""
    return await node_client.search_movies(search, genre_id, language_id, page, limit)


@mcp.tool()
async def get_movie_details(movie_id: str) -> dict:
    """Fetch movie details including synopsis, duration, cast, director, trailer."""
    return await node_client.get_movie_details(movie_id)


@mcp.tool()
async def list_genres() -> list:
    """List all available movie genres in CineMate."""
    return await node_client.list_genres()


@mcp.tool()
async def list_languages() -> list:
    """List all available audio languages for movies."""
    return await node_client.list_languages()


@mcp.tool()
async def list_theaters(city: Optional[str] = None) -> list:
    """List approved cinemas, cities, and amenities (IMAX, Dolby Atmos, Recliners)."""
    return await node_client.list_theaters(city)


@mcp.tool()
async def get_showtimes(
    movie_id: str,
    date: str,
    theater_id: Optional[str] = None,
) -> list:
    """Fetch approved showtimes for a specific movie and date (YYYY-MM-DD format)."""
    return await node_client.get_showtimes(movie_id, date, theater_id)


@mcp.tool()
async def get_showtime_seats(showtime_id: str) -> dict:
    """Fetch real-time seat availability map with pricing and lock status."""
    return await node_client.get_showtime_seats(showtime_id)


@mcp.tool()
async def create_pending_booking(
    showtime_id: str,
    seat_ids: List[str],
    auth_token: str,
    idempotency_key: Optional[str] = None,
) -> dict:
    """Lock selected seats for 10 minutes and create a PENDING booking."""
    return await node_client.create_pending_booking(
        showtime_id, seat_ids, auth_token, idempotency_key
    )


@mcp.tool()
async def get_booking(booking_id: str, auth_token: str) -> dict:
    """Retrieve details and countdown expiry for a held or confirmed booking."""
    return await node_client.get_booking(booking_id, auth_token)


@mcp.tool()
async def list_my_bookings(auth_token: str) -> list:
    """Retrieve booking history of the authenticated moviegoer."""
    return await node_client.list_my_bookings(auth_token)


@mcp.tool()
async def get_payment_link(booking_id: str, auth_token: str) -> dict:
    """Generate a Razorpay order and return checkoutUrl deep link for the user."""
    return await node_client.get_payment_link(booking_id, auth_token)


@mcp.tool()
async def get_payment_status(booking_id: str, auth_token: str) -> str:
    """Check whether a booking is confirmed or pending payment."""
    return await node_client.get_payment_status(booking_id, auth_token)


if __name__ == "__main__":
    mcp.run()
