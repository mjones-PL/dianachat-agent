from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    LIVEKIT_API_KEY: str
    LIVEKIT_API_SECRET: str
    LIVEKIT_WS_URL: str
    LIVEKIT_API_URL: str

    class Config:
        env_file = ".env"

settings = Settings()
