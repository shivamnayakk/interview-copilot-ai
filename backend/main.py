from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.utils.logger import logger
from backend.api import health
from backend.database.session import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database tables...")
    try:
        init_db()
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.warning(f"Database initialization deferred or error: {e}")
    yield
    logger.info("Shutting down application...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Backend API for Agentic AI Interview Preparation Platform",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for Next.js frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For hackathon agility
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include modular API routers
app.include_router(health.router)

@app.get("/", tags=["Root"])
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to Interview Copilot AI API. Visit /docs for Swagger documentation."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
