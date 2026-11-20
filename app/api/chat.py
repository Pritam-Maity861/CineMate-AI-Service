from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Header, Query
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
from sse_starlette.sse import EventSourceResponse
from app.graph.workflow import graph
from app.graph.nodes import active_auth_token, active_generative_ui
import json
import logging

router = APIRouter()


class ChatMessageHistoryItem(BaseModel):
    role: str = Field(..., description="Role of the speaker: 'user' or 'assistant'")
    content: str = Field(..., description="Content of the message")


class ChatRequest(BaseModel):
    message: str = Field(..., example="What movies are playing today?")
    session_id: Optional[str] = Field("default-session", example="default-session")
    history: Optional[List[ChatMessageHistoryItem]] = Field(
        default=[],
        example=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help you find a movie today?"},
        ],
    )
    auth_token: Optional[str] = Field(
        default=None,
        description="Optional JWT token (can also be passed via Authorization header)",
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Set to true for Server-Sent Events (SSE) streaming, or false for immediate JSON",
    )


class ChatResponse(BaseModel):
    response: str
    session_id: str
    generative_ui: Optional[Dict[str, Any]] = None


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    body: ChatRequest,
    authorization: Optional[str] = Header(
        None,
        description="Optional Bearer token for authenticated booking actions",
    ),
    stream: Optional[bool] = Query(
        None,
        description="Override stream mode via query parameter",
    ),
):
    """
    CineMate Conversational Assistant Endpoint.
    - Supports immediate JSON response (default, ideal for Swagger UI & testing).
    - Supports Server-Sent Events (SSE) streaming (set stream=true).
    - Auth token is OPTIONAL for movie discovery & seat browsing; only required when holding/booking seats.
    """
    raw_token = authorization or body.auth_token
    token = None
    if raw_token:
        token = raw_token.replace("Bearer ", "").strip()
    active_auth_token.set(token)
    active_generative_ui.set(None)

    # 1. Build message list from conversation history + latest user message
    messages = []
    if body.history:
        for item in body.history:
            if item.role.lower() == "assistant":
                messages.append(AIMessage(content=item.content))
            else:
                messages.append(HumanMessage(content=item.content))

    messages.append(HumanMessage(content=body.message))

    state_input = {
        "messages": messages,
        "auth_token": token,
        "session_id": body.session_id or "default-session",
        "selected_movie": None,
        "selected_theater": None,
        "selected_showtime": None,
        "selected_seats": None,
        "pending_booking": None,
        "checkout_url": None,
        "generative_ui": None,
    }

    should_stream = stream if stream is not None else body.stream

    # 2. Streaming response (SSE) for React UI
    if should_stream:

        async def event_generator():
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
                logging.error(f"[Chat Endpoint Error] {e}", exc_info=True)
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(e)}),
                }

        return EventSourceResponse(event_generator())

    # 3. Synchronous JSON response (ideal for Swagger UI / Postman testing)
    try:
        final_state = await graph.ainvoke(state_input)
        last_message = final_state["messages"][-1]
        content = getattr(last_message, "content", str(last_message))

        return ChatResponse(
            response=content,
            session_id=body.session_id or "default-session",
            generative_ui=final_state.get("generative_ui"),
        )
    except Exception as e:
        logging.error(f"[Chat Endpoint Error] {e}", exc_info=True)
        return ChatResponse(
            response=f"Error processing request: {str(e)}",
            session_id=body.session_id or "default-session",
            generative_ui=None,
        )
