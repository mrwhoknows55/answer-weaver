import logging
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class Settings(BaseSettings):
    """Loads configuration settings from environment variables and .env file."""

    # Reddit API Credentials
    reddit_client_id: str = Field(..., validation_alias='REDDIT_CLIENT_ID')
    reddit_client_secret: str = Field(..., validation_alias='REDDIT_CLIENT_SECRET')
    reddit_user_agent: str = Field(..., validation_alias='REDDIT_USER_AGENT')

    # Target Subreddit
    target_subreddit: str = Field("learnpython", validation_alias='TARGET_SUBREDDIT')

    # Qdrant Configuration
    qdrant_url: str = Field("http://localhost:6333", validation_alias='QDRANT_URL')
    qdrant_api_key: Optional[str] = Field(None, validation_alias='QDRANT_API_KEY')
    qdrant_collection_name: str = Field("reddit_posts", validation_alias='QDRANT_COLLECTION_NAME')

    # Embedding Model
    embedding_model_name: str = Field("all-MiniLM-L6-v2", validation_alias='EMBEDDING_MODEL_NAME')

    # Fetching parameters (optional defaults)
    fetch_limit: int = 5 # Max posts to fetch per run

    model_config = SettingsConfigDict(
        env_file='.env',         
        env_file_encoding='utf-8',
        extra='ignore'           
    )

# Instantiate settings once to be imported by other modules
try:
    settings = Settings()
    logger.info("Configuration loaded successfully.")
    # Optionally log some non-sensitive settings
    logger.info(f"Target Subreddit: {settings.target_subreddit}")
    logger.info(f"Qdrant Collection: {settings.qdrant_collection_name}")
    logger.info(f"Embedding Model: {settings.embedding_model_name}")
except Exception as e:
    logger.exception(f"Failed to load configuration: {e}")
    # Optionally, raise the exception or exit if config is critical
    raise SystemExit("Configuration error, exiting.") from e