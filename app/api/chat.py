from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Header
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from sse_starlette.sse import EventSourceResponse
from app.graph.workflow import graph
import json

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default-session"
    history: Optional[List[Dict[str, Any]]] = []


@router.post("/chat")
async def chat_endpoint(
    body: ChatRequest,
    authorization: Optional[str] = Header(None),
):
    """Streaming chat endpoint for React client via SSE."""

    async def event_generator():
        state_input = {
            "messages": [HumanMessage(content=body.message)],
            "auth_token": authorization,
            "session_id": body.session_id or "default-session",
            "selected_movie": None,
            "selected_theater": None,
            "selected_showtime": None,
            "selected_seats": None,
            "pending_booking": None,
            "checkout_url": None,
            "generative_ui": None,
        }

        try:
            final_state = await graph.ainvoke(state_input)
            last_message = final_state["messages"][-1]
            content = getattr(last_message, "content", str(last_message))

            yield {
                "event": "message",
                "data": json.dumps({"type": "TEXT", "content": content}),
            }

            if final_state.get("generative_ui"):
                yield {
                    "event": "generative_ui",
                    "data": json.dumps(final_state["generative_ui"]),
                }

            yield {"event": "done", "data": "[DONE]"}

        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }

    return EventSourceResponse(event_generator())

