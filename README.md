# 🧠 CineMate AI Service (Agentic Cinema Concierge)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.50+-blue?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.7+-green?style=for-the-badge)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-Fast_Inference-f55036?style=for-the-badge)](https://groq.com/)
[![FastMCP](https://img.shields.io/badge/FastMCP-0.1.0+-purple?style=for-the-badge)](https://github.com/jlowin/fastmcp)

The **CineMate AI Service** is an autonomous, agentic cinema concierge and booking assistant. Built with **FastAPI**, **LangGraph**, and **FastMCP**, it offers real-time streaming conversations via **Server-Sent Events (SSE)**, multi-criteria and semantic movie search, real-time seat inspection, automatic seat holding, and interactive Generative UI cards.

---

## 📑 Table of Contents
1. [System Architecture](#-system-architecture)
2. [Project Directory Structure](#-project-directory-structure)
3. [LangGraph State Machine & Orchestrator](#-langgraph-state-machine--orchestrator)
4. [Tool Inventory & Node API Integration](#-tool-inventory--node-api-integration)
5. [Generative UI Protocol & Action Dispatcher](#-generative-ui-protocol--action-dispatcher)
6. [Semantic Search & Embedding Pipeline](#-semantic-search--embedding-pipeline)
7. [Domain Guardrails & Business Policies](#-domain-guardrails--business-policies)
8. [Streaming Server-Sent Events (SSE) Protocol](#-streaming-server-sent-events-sse-protocol)
9. [Configuration & Environment Variables](#-configuration--environment-variables)
10. [Local Development & Deployment](#-local-development--deployment)

---

## 🏛️ System Architecture

```
User in React Chat Modal
         │
         ▼  SSE / POST: /api/chat?stream=true
┌────────────────────────────────────────────────────────┐
│               FastAPI Streaming Gateway                │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│            LangGraph Cyclic State Machine              │
│  ┌──────────────────┐          ┌────────────────────┐  │
│  │  assistant_node  │ ◄──────► │     tools_node     │  │
│  │  (Groq LLM)      │          │ (FastMCP Node API) │  │
│  └──────────────────┘          └────────────────────┘  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼ HTTP (Auth Bearer Forwarding)
┌────────────────────────────────────────────────────────┐
│            CineMate Node.js Core Backend               │
│               (PostgreSQL + Redis)                     │
└────────────────────────────────────────────────────────┘
```

---

## 📁 Project Directory Structure

```
ai-service/
├── app/
│   ├── api/                 # FastAPI REST & SSE endpoints
│   │   ├── chat.py          # Unified streaming & synchronous chat routes
│   │   └── health.py        # System health & dependency check
│   ├── graph/               # LangGraph state machine & reasoning nodes
│   │   ├── graph.py         # Graph compilation, memory checkpointers, & edges
│   │   ├── nodes.py         # Assistant node, system prompt, guardrails, & tools
│   │   └── state.py         # AgentState typed dictionary & message schema
│   ├── mcp/                 # Model Context Protocol / Node.js API client
│   │   ├── server.py        # FastMCP server definition
│   │   └── tools.py         # HTTP client wrapper querying the Node.js REST API
│   ├── vector/              # Semantic RAG & vector embedding search
│   │   └── search.py        # Dense vector search (FastEmbed / Gemini / Local Hashing)
│   ├── config.py            # Pydantic v2 settings & environment variables
│   └── main.py              # Application entrypoint & CORS middleware
├── pyproject.toml           # Project metadata & dependency specifications
└── requirements.txt         # Pip dependency lockfile
```

---

## 🔄 LangGraph State Machine & Orchestrator

The conversational reasoning is modeled as a compiled **LangGraph cyclic state graph**:

### 1. State Schema (`AgentState`)
```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    auth_token: Optional[str]
    session_id: str
    generative_ui: Optional[Dict[str, Any]]
```

### 2. Node Graph Topology
- **`assistant_node`**:
  - Sets the dynamic user authentication token into execution context (`active_auth_token`).
  - Evaluates the **Domain Guardrail** against out-of-domain requests (coding, math, etc.).
  - Binds the 9 cinema tools to the LLM (Groq `openai/gpt-oss-120b`).
  - Decides whether to reply to the user or trigger tools.
- **`tools_node`**:
  - Executes tool functions using `ToolNode`.
  - Captures Generative UI events via `active_generative_ui` context variables.
- **Conditional Routing (`should_continue`)**:
  - If the assistant produces tool calls -> routes to `tools`.
  - Once tools return outputs -> loops back to `assistant` to synthesize human-friendly responses.
  - If no tool calls remain -> transitions to `END`.

---

## 🛠️ Tool Inventory & Node API Integration

All tools communicate with the Node.js backend using `NodeAPIClient` (`app/mcp/tools.py`), propagating authorization tokens transparently:

| Tool Name | Scope | Description |
|---|---|---|
| **`search_movies_tool`** | Public | Queries catalog by keyword, genre, and language. |
| **`semantic_movie_search_tool`** | Public | Dense vector search finding movies by plot synopsis, mood, or theme. |
| **`get_movie_details_tool`** | Public | Full metadata: synopsis, duration, cast credits, directors, trailer. |
| **`list_theaters_tool`** | Public | Lists approved cinema theaters and amenities (IMAX, Dolby Atmos) by city. |
| **`get_showtimes_tool`** | Public | Retrieves scheduled showtimes for a movie on a given calendar date. |
| **`get_showtime_seats_tool`** | Public | Real-time seat map combining database layout with active Redis memory locks. |
| **`book_seats_tool`** | Authenticated | Holds seats for 10 minutes in Redis and creates a pending reservation. |
| **`get_my_bookings_tool`** | Authenticated | Fetches booking history for the logged-in customer. |
| **`get_booking_details_tool`** | Authenticated | Inspects a booking, returning confirmed ticket info and generating `TICKET_CARD`. |

---

## 🎨 Generative UI Protocol & Action Dispatcher

The AI Service doesn't merely return plain text; it triggers **interactive UI cards** in the client chat window:

```
Tool Execution (nodes.py)
         │
         ▼ Sets Context Variable
active_generative_ui.set({"type": "TICKET_CARD", "data": {...}})
         │
         ▼ Packaged in SSE Event Payload
event: final
data: {"response": "...", "generative_ui": {"type": "TICKET_CARD", "data": {...}}}
         │
         ▼ Rendered in React AgentChatModal
Client renders Ticket Poster, Show Details, "Download PDF" & "View Ticket" buttons!
```

### Supported Generative UI Actions
1. **`BOOKING_HOLD`**:
   - Triggered by `book_seats_tool`.
   - Payload: `booking_id`, `total_amount`, `seats`, `checkout_url`, `expires_at`.
   - Action: Displays a 10-minute countdown badge and a direct **"Proceed to Checkout"** button.
2. **`TICKET_CARD`**:
   - Triggered by `get_booking_details_tool` when booking is `CONFIRMED`.
   - Payload: `booking_id`, `movie_title`, `poster_url`, `theater_name`, `screen_name`, `showtime`, `seats`, `total_amount`, `raw_booking`.
   - Action: Renders an official E-Ticket with an instant **"Download PDF"** button (calls jsPDF in-browser) and **"View Ticket"** button.

---

## 🔍 Semantic Search & Embedding Pipeline

The semantic movie search engine (`app/vector/search.py`) supports plot, theme, and mood queries (e.g., *"thriller with unexpected twist in dreams"*):

### Multi-Tier Fallback Hierarchy
1. **FastEmbed (Local Neural Model)**:
   - Uses `BAAI/bge-small-en-v1.5` running on CPU via ONNX Runtime.
   - **100% Free & Local** (zero API keys, zero external cost).
2. **Google Gemini Embeddings**:
   - Uses `models/text-embedding-004` if `GEMINI_API_KEY` is provided.
3. **OpenAI Embeddings**:
   - Uses `text-embedding-3-small` if `OPENAI_API_KEY` is provided.
4. **Deterministic Local Dense Vector Engine**:
   - Built-in zero-dependency 256-dimensional dense vector engine utilizing MD5/SHA-256 n-gram hashing and L2 normalization.
   - Ensures continuous semantic search functionality without any external network dependency.

---

## 🛡️ Domain Guardrails & Business Policies

### 1. Strict Cinema Domain Guardrail
- **Absolute Negative Constraint**: CineMate AI is strictly a cinema concierge.
- It will **never** write computer programs (Python, JS, C++), solve math homework, write essays, or answer general trivia.
- Queries violating domain boundaries are intercepted with a courteous refusal:
  > *"I am CineMate AI, your dedicated cinema concierge. I can only assist with movie discovery, showtimes, theater information, and ticket bookings at CineMate. How can I help you find a movie or book a show today?"*

### 2. Strict Zero-Cancellation Policy
- CineMate Cinemas operates on a strict **no-cancellation, no-refund policy** for confirmed tickets.
- The assistant is explicitly prohibited from suggesting or inventing cancellation buttons, steps, or refund procedures.

### 3. Direct Ticket Delivery Policy
- The assistant never apologizes or says it cannot provide ticket downloads.
- It leverages the Generative UI engine to present the confirmed ticket card and immediate PDF download in the chat window.

---

## 📡 Streaming Server-Sent Events (SSE) Protocol

Clients connect to `POST /api/chat?stream=true`. The stream yields chunked events:

```http
POST /api/chat?stream=true HTTP/1.1
Content-Type: application/json

{
  "message": "Book seats C3, C4 for Parasite",
  "session_id": "session-123",
  "auth_token": "ey..."
}
```

### Event Stream Chunks:
1. **Token Chunks**:
   ```
   data: {"type": "token", "content": "I have "}
   data: {"type": "token", "content": "reserved seats C3, C4 "}
   ```
2. **Final Payload**:
   ```
   event: final
   data: {
     "response": "I have reserved seats C3, C4 for 10 minutes...",
     "session_id": "session-123",
     "generative_ui": {
       "type": "BOOKING_HOLD",
       "data": { ... }
     }
   }
   ```

---

## ⚙️ Configuration & Environment Variables

Create a `.env` file in the `ai-service/` root directory:

```env
PORT=8000
ENVIRONMENT=development

# Node.js Core Backend URL
NODE_API_URL=http://localhost:8080/api/v1
FRONTEND_URL=http://localhost:5173

# Primary LLM Provider (Groq)
DEFAULT_MODEL_PROVIDER=groq
GROQ_API_KEY=gsk_your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b

# Optional Fallback Providers
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-2.5-flash
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini
```

---

## 🚀 Local Development & Deployment

### 1. Set Up Virtual Environment
```bash
cd ai-service
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Development Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI documentation will be available at `http://localhost:8000/docs`.

