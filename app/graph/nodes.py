from typing import Any, Dict
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from app.config import settings
from app.graph.state import AgentState
from app.mcp.tools import node_client


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
        return "No movies found matching the criteria."
    summary = []
    for m in movies:
        genres = ", ".join([g.get("name", "") for g in m.get("genres", [])])
        summary.append(
            f"ID: {m['id']} | Title: {m['title']} | Duration: {m['duration']}m | Rating: {m.get('rating', 'N/A')} | Genres: {genres}"
        )
    return "\n".join(summary)


@tool
async def get_showtimes_tool(movie_id: str, date: str) -> str:
    """Fetch showtimes for a movie on a specific date (YYYY-MM-DD)."""
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
    """Check seat availability and pricing for a showtime."""
    data = await node_client.get_showtime_seats(showtime_id)
    seats = data.get("seats", [])
    if not seats:
        return "No seats found for this showtime."
    available = [s for s in seats if s.get("status") == "AVAILABLE"]
    return (
        f"Showtime {showtime_id} has {len(available)} available seats out of {len(seats)} total.\n"
        f"Available seats sample: {', '.join([f'{s['row']}{s['number']} (₹{s['price']}, {s['type']})' for s in available[:12]])}..."
    )


SYSTEM_PROMPT = """You are CineMate AI, a smart movie ticketing concierge powered by Groq and LangGraph.
Your job is to assist users in discovering movies, finding the best showtimes at approved cinemas, choosing seats, and guiding them to checkout.

Rules & Guidelines:
1. Always be conversational, concise, polite, and movie-savvy.
2. Recommend movies based on user preferences (genres, actors, themes).
3. When suggesting showtimes or seats, guide the user step-by-step.
4. If seats are locked or held by another user, politely inform them and suggest adjacent seats.
5. Never invent or hallucinate showtimes or cinemas that are not in the database.
6. When the user confirms seats, prepare the booking hold and present the checkout action.
"""

AVAILABLE_TOOLS = [search_movies_tool, get_showtimes_tool, get_showtime_seats_tool]


async def assistant_node(state: AgentState) -> Dict[str, Any]:
    """Primary reasoning node invoking LLM with tools."""
    llm = get_llm().bind_tools(AVAILABLE_TOOLS)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    try:
        response = await llm.ainvoke(messages)
        return {"messages": [response]}
    except Exception as e:
        from langchain_core.messages import AIMessage

        return {
            "messages": [
                AIMessage(
                    content=f"Hello! I'm your CineMate AI Assistant. (Please ensure GROQ_API_KEY is configured in ai-service/.env). Node.js API connection is active!"
                )
            ]
        }

