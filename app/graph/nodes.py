import contextvars
import json
import logging
import re
from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from app.config import settings
from app.graph.state import AgentState
from app.mcp.tools import node_client

# Context variables to bridge LangGraph state to tool executions
active_auth_token: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "active_auth_token", default=None
)
active_generative_ui: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar("active_generative_ui", default=None)
)


def get_llm():
    """Instantiate configured LLM provider (Groq, Gemini, or OpenAI)."""
    # 1. Groq (Primary with model e.g. openai/gpt-oss-120b)
    if settings.GROQ_API_KEY or settings.DEFAULT_MODEL_PROVIDER == "groq":
        try:
            from langchain_groq import ChatGroq

            return ChatGroq(
                model=settings.GROQ_MODEL,
                api_key=settings.GROQ_API_KEY or "DUMMY_KEY",
                temperature=0.2,
            )
        except Exception:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                base_url="https://api.groq.com/openai/v1",
                model=settings.GROQ_MODEL,
                api_key=settings.GROQ_API_KEY or "DUMMY_KEY",
                temperature=0.2,
            )

    # 2. OpenAI
    if settings.DEFAULT_MODEL_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.3,
        )

    # 3. Google Gemini fallback
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY or "DUMMY_KEY",
        temperature=0.3,
    )


# -------------------------------------------------------------
# LangChain Tools wrapping NodeAPIClient for LLM Calling
# -------------------------------------------------------------


@tool
async def search_movies_tool(
    query: str = "",
    genre_id: str = "",
    language_id: str = "",
) -> str:
    """Search approved movies in CineMate catalog by keyword, genre, or language."""
    data = await node_client.search_movies(
        search=query or None,
        genre_id=genre_id or None,
        language_id=language_id or None,
    )
    movies = data.get("movies", [])
    if not movies:
        return "No movies found matching the search criteria."
    summary = []
    for m in movies:
        genres = ", ".join([g.get("name", "") for g in m.get("genres", [])])
        summary.append(
            f"ID: {m['id']} | Title: {m['title']} | Duration: {m['duration']} mins | Rating: {m.get('rating', 'N/A')}⭐ | Genres: {genres}"
        )
    return "\n".join(summary)


@tool
async def get_movie_details_tool(movie_id: str) -> str:
    """Get full details of a movie including synopsis, duration, cast, director, and trailer."""
    m = await node_client.get_movie_details(movie_id)
    if not m:
        return f"Movie with ID {movie_id} not found."

    credits = m.get("credits", [])
    actors = [
        c.get("person", {}).get("name")
        for c in credits
        if c.get("role") == "ACTOR" and c.get("person")
    ]
    directors = [
        c.get("person", {}).get("name")
        for c in credits
        if c.get("role") == "DIRECTOR" and c.get("person")
    ]
    genres = ", ".join([g.get("name", "") for g in m.get("genres", [])])

    return (
        f"Title: {m.get('title')}\n"
        f"Synopsis: {m.get('description')}\n"
        f"Duration: {m.get('duration')} mins | Rating: {m.get('rating')}⭐ | Genres: {genres}\n"
        f"Director(s): {', '.join(directors) if directors else 'N/A'}\n"
        f"Starring: {', '.join(actors[:5]) if actors else 'N/A'}\n"
        f"Trailer: {m.get('trailerUrl') or 'N/A'}"
    )


@tool
async def list_theaters_tool(city: str = "") -> str:
    """List approved cinemas and their amenities (IMAX, Dolby Atmos, Recliners) in a given city."""
    theaters = await node_client.list_theaters(city=city or None)
    if not theaters:
        return f"No approved theaters found{' in ' + city if city else ''}."
    lines = []
    for t in theaters:
        amenities = ", ".join(t.get("amenities", []))
        lines.append(
            f"ID: {t['id']} | Name: {t['name']} | City: {t['city']} | Address: {t['address']} | Amenities: {amenities or 'Standard'}"
        )
    return "\n".join(lines)


@tool
async def get_showtimes_tool(movie_id: str, date: str) -> str:
    """Fetch showtimes for a movie on a specific date (YYYY-MM-DD format)."""
    data = await node_client.get_showtimes(movie_id=movie_id, date=date)
    if not data:
        return f"No showtimes scheduled for movie {movie_id} on {date}."
    lines = []
    for theater in data:
        lines.append(f"\nTheater: {theater.get('theaterName')} ({theater.get('city')})")
        for screen in theater.get("screens", []):
            for st in screen.get("showtimes", []):
                lines.append(
                    f"  - Showtime ID: {st['showtimeId']} | Screen: {screen['screenName']} | Time: {st['startTime']}"
                )
    return "\n".join(lines)


@tool
async def get_showtime_seats_tool(showtime_id: str) -> str:
    """Check real-time seat availability map, pricing tiers, and occupied seats for a showtime."""
    data = await node_client.get_showtime_seats(showtime_id)
    seats = data.get("seats", [])
    if not seats:
        return "No seats found for this showtime."

    available = [s for s in seats if s.get("status") == "AVAILABLE"]
    booked = [s for s in seats if s.get("status") in ("BOOKED", "HELD")]

    # Group available seats by row
    row_map: Dict[str, List[str]] = {}
    for s in available:
        row = s["row"]
        row_map.setdefault(row, []).append(f"{s['number']}(₹{s['price']})")

    rows_summary = []
    for r in sorted(row_map.keys()):
        rows_summary.append(f"Row {r}: {', '.join(row_map[r])}")

    return (
        f"Showtime {showtime_id} at {data.get('theaterName', 'Cinema')} ({data.get('screenName', 'Screen')})\n"
        f"Total Seats: {len(seats)} | Available: {len(available)} | Occupied/Held: {len(booked)}\n"
        f"Available seats by row:\n" + "\n".join(rows_summary)
    )


@tool
async def book_seats_tool(showtime_id: str, seats: str) -> str:
    """
    Reserve/hold seats in the live database for 10 minutes and create an official booking order.
    Args:
        showtime_id: The UUID of the showtime.
        seats: Comma-separated seat labels (e.g. 'C3, C4' or 'A1, A2').
    """
    token = active_auth_token.get()
    if not token:
        return (
            "AUTHENTICATION_REQUIRED: To hold and book seats, you must be logged in. "
            "Please ask the user to log in to their CineMate account."
        )

    # 1. Fetch seat matrix to map labels ('C3') to database UUIDs
    try:
        seat_matrix = await node_client.get_showtime_seats(showtime_id)
    except Exception as e:
        return f"ERROR: Could not load seat layout for showtime {showtime_id}: {str(e)}"

    all_seats = seat_matrix.get("seats", [])
    if not all_seats:
        return f"ERROR: Showtime {showtime_id} has no seat data available."

    # Parse requested seats e.g. "C3, C4" -> [("C", 3), ("C", 4)]
    raw_labels = [s.strip().upper() for s in re.split(r"[, ]+", seats) if s.strip()]
    if not raw_labels:
        return "ERROR: Please specify seat names (e.g. 'C3, C4')."

    resolved_seat_ids: List[str] = []
    resolved_seat_names: List[str] = []
    total_calculated_price = 0

    for label in raw_labels:
        # Match pattern Row letter + number e.g. C3, D10
        match = re.match(r"^([A-Z])(\d+)$", label)
        matching_seat = None
        if match:
            row_letter, seat_num = match.group(1), int(match.group(2))
            matching_seat = next(
                (
                    s
                    for s in all_seats
                    if s["row"].upper() == row_letter and int(s["number"]) == seat_num
                ),
                None,
            )
        else:
            # Check by raw ID
            matching_seat = next((s for s in all_seats if s["seatId"] == label), None)

        if not matching_seat:
            return f"ERROR: Seat '{label}' does not exist in this auditorium. Please check available seats."

        if matching_seat.get("status") != "AVAILABLE":
            return f"ERROR: Seat '{label}' is currently {matching_seat.get('status').lower()} by another customer. Please choose different seats."

        resolved_seat_ids.append(matching_seat["seatId"])
        resolved_seat_names.append(f"{matching_seat['row']}{matching_seat['number']}")
        total_calculated_price += matching_seat.get("price", 0)

    # 2. Call backend to create the pending booking
    try:
        booking = await node_client.create_pending_booking(
            showtime_id=showtime_id,
            seat_ids=resolved_seat_ids,
            auth_token=token,
        )
    except Exception as e:
        return f"BOOKING_FAILED: {str(e)}"

    booking_id = booking.get("id")
    total_amount = booking.get("totalAmount") or total_calculated_price
    expires_at = booking.get("expiresAt")
    checkout_url = f"/checkout/{booking_id}"

    # Set Generative UI card
    active_generative_ui.set(
        {
            "type": "BOOKING_HOLD",
            "data": {
                "booking_id": booking_id,
                "checkout_url": checkout_url,
                "total_amount": total_amount,
                "seats": resolved_seat_names,
                "expires_at": expires_at,
                "movie_title": seat_matrix.get("movieTitle"),
                "theater_name": seat_matrix.get("theaterName"),
            },
        }
    )

    return (
        f"SUCCESS_BOOKED!\n"
        f"- Booking ID: {booking_id}\n"
        f"- Seats Held: {', '.join(resolved_seat_names)}\n"
        f"- Total Amount: ₹{total_amount}\n"
        f"- Checkout Link: {checkout_url}\n"
        f"- Hold Window: 10 minutes\n"
        f"Instruction for assistant: Tell the user their seats are now on hold. "
        f"Provide the exact checkout relative link: [Proceed to Checkout]({checkout_url}). "
        f"DO NOT invent any fake external domain (like cinemate.example.com). The UI will render a checkout button automatically."
    )


# -------------------------------------------------------------
# System Prompt & Available Tools
# -------------------------------------------------------------

SYSTEM_PROMPT = """You are CineMate AI, a smart, polite, and movie-savvy ticketing concierge for CineMate Cinemas.

Your primary capabilities:
1. Movie Discovery: Search movies by genre, language, title, and provide synopses.
2. Showtimes & Theaters: Look up approved showtimes and theaters for any date.
3. Seat Checking: View real-time availability (Available vs Booked).
4. Seat Booking: When a user chooses seats (e.g. "Book C3 and C4"), you MUST call `book_seats_tool` to hold them in the database.

STRICT CONSTRAINTS & RULES:
- NEVER pretend you held or booked seats without successfully calling `book_seats_tool`.
- NEVER invent or fabricate booking IDs or fake URLs like 'https://cinemate.example.com/checkout...'.
- After `book_seats_tool` returns SUCCESS_BOOKED with a booking ID, always provide the exact relative link: [Proceed to Checkout](/checkout/{booking_id}) where {booking_id} is the real UUID returned by the tool.
- If `book_seats_tool` returns BOOKING_FAILED or asks for authentication, inform the user clearly and guide them to log in.
- Format responses cleanly. When showing booking summaries, keep them structured and concise.
"""

AVAILABLE_TOOLS = [
    search_movies_tool,
    get_movie_details_tool,
    list_theaters_tool,
    get_showtimes_tool,
    get_showtime_seats_tool,
    book_seats_tool,
]


async def assistant_node(state: AgentState) -> Dict[str, Any]:
    """Primary reasoning node invoking LLM with tools."""
    # Store token in context variable so tools can access it
    active_auth_token.set(state.get("auth_token"))

    llm = get_llm().bind_tools(AVAILABLE_TOOLS)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    try:
        response = await llm.ainvoke(messages)
        gen_ui = active_generative_ui.get()
        return {
            "messages": [response],
            "generative_ui": gen_ui,
        }
    except Exception as e:
        logging.error(f"[LangGraph Assistant Error] {e}", exc_info=True)
        return {
            "messages": [
                AIMessage(
                    content=f"Hello! I am CineMate AI. How can I help you find movies or book seats today? (Groq/API is active)."
                )
            ]
        }
