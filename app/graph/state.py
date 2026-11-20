from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Conversational State for the CineMate AI Assistant."""

    messages: Annotated[List[Dict[str, Any]], add_messages]
    auth_token: Optional[str]
    session_id: str

    # Dialogue context slots
    selected_movie: Optional[Dict[str, Any]]
    selected_theater: Optional[Dict[str, Any]]
    selected_showtime: Optional[Dict[str, Any]]
    selected_seats: Optional[List[Dict[str, Any]]]
    pending_booking: Optional[Dict[str, Any]]
    checkout_url: Optional[str]

    # Generative UI action payload emitted to React client
    generative_ui: Optional[Dict[str, Any]]

