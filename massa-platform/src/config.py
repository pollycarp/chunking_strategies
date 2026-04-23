from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All configuration is read from environment variables or a .env file.
    Pydantic validates types and raises a clear error if required values are missing.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore env vars not defined here
    )

    # Database
    postgres_host: str
    postgres_port: int = 5432
    postgres_db: str
    postgres_user: str
    postgres_password: str

    # Embedding API keys
    openai_api_key: str = ""           # Required for OpenAI embeddings
    voyage_api_key: str = ""           # Optional — Voyage AI finance-domain embeddings

    # Retrieval API keys
    cohere_api_key: str = ""           # Optional — Cohere Rerank cross-encoder

    # LLM
    anthropic_api_key: str = ""        # Required for Claude agent (Phase 7)

    # Application
    env: str = "development"

    @property
    def database_dsn(self) -> str:
        """
        asyncpg connection string format:
        postgresql://user:password@host:port/dbname
        """
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


# Single shared instance — import this everywhere instead of re-instantiating
settings = Settings()
