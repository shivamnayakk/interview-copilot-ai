from fastapi import APIRouter, Depends
from sqlmodel import Session, text
from backend.database.session import get_session
from backend.utils.logger import logger
import time

router = APIRouter(prefix="/api", tags=["Health"])

@router.get("/health")
async def health_check(session: Session = Depends(get_session)):
    """
    Health check endpoint verifying API and Database connectivity.
    """
    db_status = "disconnected"
    try:
        # Test DB query
        session.exec(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        logger.error(f"Database connection error during health check: {e}")

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "database": db_status,
        "timestamp": time.time(),
        "service": "Interview Copilot AI Backend"
    }
