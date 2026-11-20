from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.health import router as health_router
from app.api.chat import router as chat_router

app = FastAPI(
    title="CineMate AI Service (Groq + LangGraph)",
    description="Agentic Movie Booking Assistant powered by Groq, LangGraph and FastMCP",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route Mounts
app.include_router(health_router, prefix="/api", tags=["Health"])
app.include_router(chat_router, prefix="/api", tags=["Chat"])


@app.get("/")
async def root():
    return {
        "service": "CineMate AI Service",
        "provider": settings.DEFAULT_MODEL_PROVIDER,
        "model": settings.GROQ_MODEL,
        "status": "ready",
        "docs": "/docs",
    }

