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
    try:
        data = await node_client.search_movies(
            search=query or None,
            genre_id=genre_id or None,
            language_id=language_id or None,
        )
    except Exception as e:
        return f"ERROR: Could not connect to CineMate backend ({str(e)}). Please verify the backend server is running and reachable at {settings.NODE_API_URL}."

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
async def semantic_movie_search_tool(vibe_or_theme: str) -> str:
    """
    Search movies semantically by vibe, theme, plot description, or mood.
    Examples: 'mind-bending sci-fi with dark humor', 'emotional family drama', 'action thriller with twists'.
    """
    try:
        from app.vector.search import semantic_movie_search
        results = await semantic_movie_search(query=vibe_or_theme, top_k=5)
    except Exception as e:
        return f"ERROR: Could not perform semantic search ({str(e)}). Backend connection may be unavailable."

    if not results:
        return f"No movies found matching the theme '{vibe_or_theme}'."

    lines = [f"Semantic matches for '{vibe_or_theme}':"]
    for m in results:
        lines.append(
            f"- {m['title']} (Rating: {m.get('rating', 'N/A')}⭐ | Genres: {m.get('genres', 'N/A')} | Match: {m.get('score')})\n"
            f"  Synopsis: {m.get('description', 'N/A')}\n"
            f"  Movie ID: {m['id']}"
        )
    return "\n\n".join(lines)


@tool
async def get_movie_details_tool(movie_id: str) -> str:
    """Get full details of a movie including synopsis, duration, cast, director, and trailer."""
    try:
        m = await node_client.get_movie_details(movie_id)
    except Exception as e:
        return f"ERROR: Could not fetch movie details ({str(e)}). Backend unreachable at {settings.NODE_API_URL}."

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
    try:
        theaters = await node_client.list_theaters(city=city or None)
    except Exception as e:
        return f"ERROR: Could not list theaters ({str(e)}). Backend unreachable at {settings.NODE_API_URL}."

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
    try:
        data = await node_client.get_showtimes(movie_id=movie_id, date=date)
    except Exception as e:
        return f"ERROR: Could not fetch showtimes ({str(e)}). Backend unreachable at {settings.NODE_API_URL}."

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
    try:
        data = await node_client.get_showtime_seats(showtime_id)
    except Exception as e:
        return f"ERROR: Could not fetch seats ({str(e)}). Backend unreachable at {settings.NODE_API_URL}."

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
    full_checkout_url = f"{settings.FRONTEND_URL}/checkout/{booking_id}"

    # Set Generative UI card
    active_generative_ui.set(
        {
            "type": "BOOKING_HOLD",
            "data": {
                "booking_id": booking_id,
                "checkout_url": checkout_url,
                "full_checkout_url": full_checkout_url,
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
        f"- Workable Checkout Link: {full_checkout_url}\n"
        f"- Relative Route: {checkout_url}\n"
        f"- Hold Window: 10 minutes\n"
        f"Instruction for assistant: Tell the user their seats are now on hold for 10 minutes.\n"
        f"Give the user this exact clickable link: [Proceed to Checkout – Pay ₹{total_amount}]({full_checkout_url})\n"
        f"Do NOT wrap the link in double asterisks, and NEVER use 'cinemate.example.com'. Use the real URL: {full_checkout_url}.\n"
        f"Do NOT mention cancellation or offer any cancellation steps. Confirmed tickets cannot be cancelled."
    )


@tool
async def get_my_bookings_tool() -> str:
    """
    Fetch the list of ticket bookings made by the currently authenticated user.
    Requires user to be logged in.
    """
    token = active_auth_token.get()
    if not token:
        return (
            "AUTHENTICATION_REQUIRED: The user is currently not logged in. "
            "Please instruct the user to log in at [Log In](/login) to view their personal bookings."
        )

    try:
        bookings = await node_client.list_my_bookings(token)
    except Exception as e:
        return f"ERROR: Could not retrieve bookings: {str(e)}"

    if not bookings:
        return "You have no previous movie bookings in your account."

    summary = ["Your CineMate Bookings:"]
    for b in bookings[:10]:
        b_id = b.get("id")
        status = b.get("status", "UNKNOWN")
        total = b.get("totalAmount", 0)
        showtime = b.get("showtime", {})
        movie = showtime.get("movie", {})
        title = movie.get("title", "Movie")
        screen_obj = showtime.get("screen", {})
        theater = screen_obj.get("theater", {}).get("name") or showtime.get("theater", {}).get("name", "Cinema")
        start_time = showtime.get("startTime", "")
        seats = [
            bs.get("seat", {}).get("row", "") + str(bs.get("seat", {}).get("number", ""))
            for bs in b.get("seats", [])
        ]
        seats_str = ", ".join(filter(None, seats)) or "N/A"
        link_str = (
            f"[View & Download Ticket]({settings.FRONTEND_URL}/booking-success/{b_id})"
            if status == "CONFIRMED"
            else f"[Complete Payment]({settings.FRONTEND_URL}/checkout/{b_id})"
        )
        summary.append(
            f"- Booking ID: {b_id} | Status: {status} | Movie: {title} | Theater: {theater} | Seats: {seats_str} | Total: ₹{total} | Time: {start_time} | Link: {link_str}"
        )

    summary.append(
        "\nInstruction for assistant: If user asks for their ticket, ticket card, or ticket download for any confirmed booking, call get_booking_details_tool(booking_id=...) to display the interactive ticket card and PDF download in chat."
    )
    return "\n".join(summary)


@tool
async def get_booking_details_tool(booking_id: str) -> str:
    """
    Fetch specific booking details and status by booking UUID for the logged-in user.
    Requires user to be logged in.
    """
    token = active_auth_token.get()
    if not token:
        return (
            "AUTHENTICATION_REQUIRED: The user is currently not logged in. "
            "Please instruct the user to log in at [Log In](/login) to inspect booking details."
        )

    try:
        b = await node_client.get_booking(booking_id, token)
    except Exception as e:
        return f"ERROR: Could not retrieve booking {booking_id}: {str(e)}"

    if not b:
        return f"Booking {booking_id} was not found or you do not have permission to view it."

    status = b.get("status", "UNKNOWN")
    total = b.get("totalAmount", 0)
    showtime = b.get("showtime", {})
    movie = showtime.get("movie", {})
    title = movie.get("title", "Movie")
    poster_url = movie.get("posterUrl")
    screen_obj = showtime.get("screen", {})
    screen = screen_obj.get("name", "Screen")
    theater = screen_obj.get("theater", {}).get("name") or showtime.get("theater", {}).get("name", "Cinema")
    theater_address = screen_obj.get("theater", {}).get("address", "")
    start_time = showtime.get("startTime", "")
    seats = [
        bs.get("seat", {}).get("row", "") + str(bs.get("seat", {}).get("number", ""))
        for bs in b.get("seats", [])
    ]
    seat_labels = list(filter(None, seats))

    if status == "CONFIRMED":
        active_generative_ui.set(
            {
                "type": "TICKET_CARD",
                "data": {
                    "booking_id": booking_id,
                    "movie_title": title,
                    "poster_url": poster_url,
                    "theater_name": theater,
                    "theater_address": theater_address,
                    "screen_name": screen,
                    "showtime": start_time,
                    "seats": seat_labels,
                    "total_amount": total,
                    "status": status,
                    "ticket_url": f"/booking-success/{booking_id}",
                    "full_ticket_url": f"{settings.FRONTEND_URL}/booking-success/{booking_id}",
                    "raw_booking": b,
                },
            }
        )
        return (
            f"TICKET_READY!\n"
            f"- Booking ID: {booking_id}\n"
            f"- Status: CONFIRMED\n"
            f"- Movie: {title}\n"
            f"- Theater: {theater} ({screen})\n"
            f"- Showtime: {start_time}\n"
            f"- Seats: {', '.join(seat_labels)}\n"
            f"- Total Paid: ₹{total}\n"
            f"- Ticket URL: {settings.FRONTEND_URL}/booking-success/{booking_id}\n"
            f"Instruction for assistant: Present the official confirmed e-ticket to the user directly with the details above.\n"
            f"The CineMate chat interface has already rendered an interactive E-Ticket Card with a direct 'Download PDF' button and 'View Ticket' button for the user.\n"
            f"DO NOT apologize or say you cannot provide a ticket or download link! You MUST provide this clickable link: [View & Download Ticket]({settings.FRONTEND_URL}/booking-success/{booking_id})."
        )
    elif status == "PENDING":
        active_generative_ui.set(
            {
                "type": "BOOKING_HOLD",
                "data": {
                    "booking_id": booking_id,
                    "checkout_url": f"/checkout/{booking_id}",
                    "full_checkout_url": f"{settings.FRONTEND_URL}/checkout/{booking_id}",
                    "total_amount": total,
                    "seats": seat_labels,
                    "expires_at": b.get("expiresAt"),
                    "movie_title": title,
                    "theater_name": theater,
                },
            }
        )
        return (
            f"PENDING_BOOKING:\n"
            f"- Booking ID: {booking_id}\n"
            f"- Status: PENDING\n"
            f"- Movie: {title}\n"
            f"- Theater: {theater} ({screen})\n"
            f"- Showtime: {start_time}\n"
            f"- Seats: {', '.join(seat_labels)}\n"
            f"- Total Amount: ₹{total}\n"
            f"- Checkout Link: {settings.FRONTEND_URL}/checkout/{booking_id}\n"
            f"Instruction for assistant: Seats are on hold. Provide the checkout link: [Proceed to Checkout – Pay ₹{total}]({settings.FRONTEND_URL}/checkout/{booking_id})."
        )

    return (
        f"Booking Details:\n"
        f"- ID: {booking_id}\n"
        f"- Status: {status}\n"
        f"- Movie: {title}\n"
        f"- Theater: {theater} ({screen})\n"
        f"- Showtime: {start_time}\n"
        f"- Seats: {', '.join(seat_labels)}\n"
        f"- Total Amount: ₹{total}\n"
    )


# -------------------------------------------------------------
# Domain Guardrail & System Prompt
# -------------------------------------------------------------

def check_domain_guardrail(message_text: str) -> Optional[str]:
    """
    Deterministic domain guardrail check.
    Returns refusal message if the query is clearly out of CineMate cinema domain.
    """
    if not message_text:
        return None

    text = message_text.strip().lower()

    # Patterns for programming, code generation, technical math, and general off-topic queries
    off_topic_patterns = [
        # Coding / Programming
        r"\b(write|give|create|generate|provide|show)\s+(me\s+)?(the\s+)?(code|program|script|function|class|algorithm)\b",
        r"\b(code\s+(for|to)\s+(add|calculate|print|sort|reverse|scrape|build|parse|loop))\b",
        r"\b(in\s+)?(python|javascript|typescript|c\+\+|java|rust|golang|php|html|css|sql)\b.*\b(code|script|program|function)\b",
        r"\b(python|javascript|typescript|c\+\+|java|rust|golang|php)\s+snippet\b",
        r"\b(how\s+to\s+code|how\s+to\s+program|debug\s+this|fix\s+my\s+code)\b",
        r"\b(def\s+[a-zA-Z_]\w*\(|function\s+[a-zA-Z_]\w*\(|class\s+[A-Z]\w*:)\b",

        # Math / Equations / Homework
        r"\b(solve|calculate|evaluate|integrate|differentiate)\s+(this\s+)?(equation|math|derivative|integral|matrix|problem)\b",
        r"\b(homework\s+help|write\s+an?\s+essay|write\s+a\s+poem|solve\s+\d+[\+\-\*\/]\d+)\b",

        # General non-cinema queries
        r"\b(recipe\s+for|how\s+to\s+cook|diet\s+plan|crypto\s+price|stock\s+market)\b",
    ]

    for pattern in off_topic_patterns:
        if re.search(pattern, text):
            return (
                "I am **CineMate AI**, your dedicated cinema and movie ticketing concierge. 🎬\n\n"
                "I can only help you with:\n"
                "- 🎬 **Movies**: Synopses, cast, ratings, and trailers\n"
                "- 📍 **Cinemas & Theaters**: Locations, IMAX / Dolby Atmos amenities\n"
                "- 🕒 **Showtimes & Schedules**: Real-time screening dates & hours\n"
                "- 🎟️ **Ticketing**: Checking seat layouts, pricing, and locking your seats\n"
                "- 📋 **Your Bookings**: Viewing past confirmed tickets\n\n"
                "*I cannot assist with programming, general math, or off-topic requests. How can I assist you with a movie today?*"
            )

    return None


SYSTEM_PROMPT = f"""You are CineMate AI, the EXCLUSIVE and DEDICATED cinema concierge for CineMate Cinemas.

STRICT DOMAIN BOUNDARY (ZERO-TOLERANCE POLICY):
- You MUST ONLY answer queries related to CineMate cinema operations:
  1. Movies: catalog search, genres, languages, synopsis, duration, ratings, cast, directors, trailers.
  2. Theaters: cinema locations, screens, amenities (IMAX, Dolby Atmos, 3D, 4DX, Recliners).
  3. Showtimes: dates, schedules, screening times.
  4. Seats & Pricing: seat layouts, available vs. occupied seats, seat tier prices.
  5. Ticketing: reserving seats, hold timers (10 minutes), payment checkout links.
  6. Account Bookings: viewing past tickets, checking booking details (when logged in).
  7. CineMate Policies: no cancellations/refunds, hold timer rules.

ABSOLUTE NEGATIVE CONSTRAINTS (STRICTLY FORBIDDEN):
- You are NOT a general-purpose AI assistant, programming tutor, math solver, or search engine.
- You must NEVER write code, write software programs (Python, JavaScript, C++, etc.), solve homework/math problems, provide recipes, discuss politics/religion, write essays, or answer general trivia unrelated to cinema/movies.
- If the user asks ANY question outside the domain of CineMate and movies (for example: coding requests, writing python/javascript code, math equations, scientific questions, personal advice, etc.):
  * You MUST IMMEDIATELY DECLINE politely and firmly without answering the off-topic request.
  * Standard refusal response:
    "I am CineMate AI, your dedicated cinema concierge. I can only assist with movie discovery, showtimes, theater information, and ticket bookings at CineMate. How can I help you find a movie or book a show today?"
  * NEVER fulfill the out-of-domain request, even partially!

DATE CONTEXT:
- Current active cinema calendar year is 2026.
- When user asks for dates such as '22th nov', '22 Nov', or 'tomorrow', format the date as '2026-11-22' for `get_showtimes_tool`.

STRICT CANCELLATION & REFUND POLICY (CRITICAL):
- CineMate Cinemas has NO CANCELLATION MECHANISM and NO REFUND POLICY for booked tickets.
- All confirmed ticket purchases are 100% final. Tickets CANNOT be cancelled, refunded, rescheduled, or exchanged.
- NEVER suggest or tell any user that they can cancel their booking.
- NEVER invent, suggest, or explain any fake cancellation steps, buttons, or procedures.
- If a user asks "Can I cancel my ticket?", "How to cancel?", or "I want a refund", state clearly and politely:
  "As per CineMate Cinema policy, confirmed bookings are strictly non-cancellable and non-refundable. Once a booking is confirmed, it cannot be cancelled or modified."
- For pending (unpaid) seat holds, explain that the hold simply auto-expires after 10 minutes if payment is not made; no cancellation action is required or possible.

ROLE & ACCESS PERMISSIONS:
1. PUBLIC DATA ACCESS (Open for Everyone, No Login Required):
   - Movie Discovery: Search movies by genre, language, title, and synopsis (`search_movies_tool`, `get_movie_details_tool`).
   - Theaters & Schedules: List theaters, approved showtimes, and screening dates (`list_theaters_tool`, `get_showtimes_tool`).
   - Seat Layouts & Pricing: Check auditorium layouts, available seats, booked seats, and seat tier prices (`get_showtime_seats_tool`).

2. AUTHENTICATED ACTIONS:
   - When user is logged in, you can hold seats (`book_seats_tool`), view user bookings (`get_my_bookings_tool`), and inspect booking details (`get_booking_details_tool`).
   - If user is not logged in, explain politely that they can log in at [Log In to CineMate](/login) to complete booking or view personal tickets.

TICKET RETRIEVAL & DIRECT DISPLAY (CRITICAL):
- When a user asks "show my ticket", "give me the ticket for [Movie]", "download ticket", "send me the ticket", or queries a booking ID:
  1. Call `get_booking_details_tool(booking_id=...)` (or `get_my_bookings_tool()` first if the booking ID is not known).
  2. NEVER say "I'm sorry, but I can't provide a direct download link or file for your ticket" or ask the user to manually navigate to their account!
  3. The CineMate chat system automatically displays an interactive E-Ticket Card right inside the chat window, complete with an instant "Download PDF" button and "View Ticket" button!
  4. Present all details of the confirmed booking proudly and warmly.
  5. Always provide the direct clickable link format: `[View & Download Ticket](http://localhost:5173/booking-success/<booking-id>)`.

FORMATTING GUIDELINES:
- Use clean, structured Markdown with bullet points or compact tables.
- NEVER invent external domains (e.g. cinemate.example.com).
"""

AVAILABLE_TOOLS = [
    search_movies_tool,
    semantic_movie_search_tool,
    get_movie_details_tool,
    list_theaters_tool,
    get_showtimes_tool,
    get_showtime_seats_tool,
    book_seats_tool,
    get_my_bookings_tool,
    get_booking_details_tool,
]


async def assistant_node(state: AgentState) -> Dict[str, Any]:
    """Primary reasoning node invoking LLM with tools and dynamic auth context."""
    token = state.get("auth_token") or active_auth_token.get()
    active_auth_token.set(token)

    # 1. Deterministic Domain Guardrail: intercept off-topic, code, and non-cinema queries immediately
    latest_user_text = ""
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "content"):
            msg_type = getattr(msg, "type", "")
            if msg_type in ("human", "user"):
                latest_user_text = str(msg.content)
                break
        elif isinstance(msg, dict) and msg.get("role") in ("user", "human"):
            latest_user_text = str(msg.get("content", ""))
            break

    guardrail_refusal = check_domain_guardrail(latest_user_text)
    if guardrail_refusal:
        return {
            "messages": [AIMessage(content=guardrail_refusal)],
            "generative_ui": None,
        }

    if token:
        user_auth_context = (
            "====================================================\n"
            "ACTIVE SESSION AUTHENTICATION: [AUTHENTICATED & LOGGED IN]\n"
            "- A verified JWT auth token IS PRESENT in this session.\n"
            "- THE USER IS ALREADY LOGGED IN TO CINEMATE. DO NOT ASK OR TELL THE USER TO LOG IN!\n"
            "- When the user asks to book a movie (e.g. 'book the parasite movie on 22th nov in kolkata'):\n"
            "  Step 1: Look up the movie using `search_movies_tool(query='Parasite')`.\n"
            "  Step 2: Look up showtimes using `get_showtimes_tool` with date '2026-11-22'.\n"
            "  Step 3: Present the available theaters, screens, and showtimes found in the city.\n"
            "  Step 4: Check seats with `get_showtime_seats_tool` and ask the user which showtime and seats (e.g. C3, C4) they would like to reserve.\n"
            "  Step 5: Once showtime and seats are specified, call `book_seats_tool` to hold the seats and provide the checkout link.\n"
            "- If the user asks for their bookings, call `get_my_bookings_tool`.\n"
            "===================================================="
        )
    else:
        user_auth_context = (
            "====================================================\n"
            "ACTIVE SESSION AUTHENTICATION: [GUEST / NOT LOGGED IN]\n"
            "- No auth token is present in this session.\n"
            "- You CAN and SHOULD still search movies, showtimes, theaters, seat layouts, and prices (`search_movies_tool`, `get_showtimes_tool`, `get_showtime_seats_tool`).\n"
            "- If the user asks 'book the parasite movie on 22th nov in kolkata', FIRST look up the showtimes for Parasite on 2026-11-22 to help them!\n"
            "- Then remind them: 'To lock your seats and proceed to checkout, please [Log In to CineMate](/login)'.\n"
            "===================================================="
        )

    full_system_prompt = f"{SYSTEM_PROMPT}\n\n{user_auth_context}"
    llm = get_llm().bind_tools(AVAILABLE_TOOLS)
    messages = [SystemMessage(content=full_system_prompt)] + state["messages"]

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
