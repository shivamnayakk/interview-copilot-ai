from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Interview Copilot AI"
    OPENAI_API_KEY: str = "sk-placeholder"
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/interview_copilot"
    DEBUG: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
