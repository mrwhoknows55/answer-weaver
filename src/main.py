import logging
import sys
from src.data_source.reddit import fetch_subreddit_posts
from src.db.qdrant import initialize_qdrant_collection, upsert_posts_to_qdrant
from src.config import settings
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_pipeline():
    """
    Executes the full pipeline: initialize DB, fetch posts, upsert posts.
    """
    logger.info("Starting the Reddit data fetching and vector store pipeline...")

    try:
        # 1. Initialize Qdrant Collection
        logger.info("Step 1: Initializing Qdrant collection...")
        initialize_qdrant_collection()
        logger.info("Qdrant collection initialized successfully.")

        # 2. Fetch Reddit Posts
        logger.info(f"Step 2: Fetching posts from r/{settings.target_subreddit} (limit: {settings.fetch_limit})...")
        posts = fetch_subreddit_posts(settings.target_subreddit, limit=settings.fetch_limit)
        if not posts:
            logger.warning("No posts were fetched. Check subreddit name, Reddit API status, and PRAW logs.")
            logger.info("Pipeline finished: No posts to process.")
            return # Exit gracefully if no posts found
        logger.info(f"Fetched {len(posts)} posts successfully.")

        # 3. Upsert Posts to Qdrant
        logger.info("Step 3: Upserting posts into Qdrant...")
        upsert_posts_to_qdrant(posts)
        logger.info("Posts upserted successfully.")

        logger.info("Pipeline finished successfully.")

    except SystemExit:
        logger.error("Exiting due to SystemExit during pipeline execution.")
        sys.exit(1)
    except Exception as e:
        logger.exception("An unexpected error occurred during pipeline execution.")
        sys.exit(1) 


app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    # Ensure .env file exists and Qdrant is running before executing
    logger.info("Checking prerequisites: Ensure .env file is configured and Qdrant server is running.")
    # Add a small check or reminder here if needed
    # For example, check if QDRANT_URL is reachable? (could add a simple ping/health check)

    run_pipeline()
