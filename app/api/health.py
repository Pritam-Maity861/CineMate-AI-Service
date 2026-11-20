from fastapi import APIRouter
import httpx
from app.config import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    node_api_reachable = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            res = await client.get(f"{settings.NODE_API_URL}/movie/get-genre")
            node_api_reachable = res.status_code == 200
    except Exception:
        node_api_reachable = False

    return {
        "status": "healthy",
        "service": "ai-service",
        "node_api_url": settings.NODE_API_URL,
        "node_api_connected": node_api_reachable,
        "model_provider": settings.DEFAULT_MODEL_PROVIDER,
        "groq_model": settings.GROQ_MODEL,
    }

