from typing import List, Dict, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, CollectionStatus, UpdateStatus
from sentence_transformers import SentenceTransformer
import uuid

from ..config import settings, logger

# Global Qdrant client instance
qdrant_client: Optional[QdrantClient] = None

def get_qdrant_client() -> QdrantClient:
    """Initializes and returns a singleton Qdrant client instance."""
    global qdrant_client
    if qdrant_client is None:
        logger.info(f"Initializing Qdrant client for URL: {settings.qdrant_url}")
        try:
            qdrant_client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                prefer_grpc=False # Set to True if gRPC is preferred and enabled
            )
            logger.info("Qdrant client initialized and health check passed.")
        except Exception as e:
            logger.exception("Failed to initialize Qdrant client.")
            raise SystemExit("Qdrant connection error, exiting.") from e
    return qdrant_client

def initialize_qdrant_collection():
    """
    Ensures the Qdrant collection exists, creating it if necessary
    using fastembed for automatic embedding generation.
    """
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name
    embedding_model = settings.embedding_model_name

    try:
        # Check if collection exists
        collections_response = client.get_collections()
        collection_names = [col.name for col in collections_response.collections]

        if collection_name not in collection_names:
            logger.info(f"Collection '{collection_name}' not found. Creating...")
            # Create collection with fastembed configuration
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE), # Size for all-MiniLM-L6-v2
                # Specify the on-disk HNSW index for potentially better performance/memory usage
                hnsw_config=models.HnswConfigDiff(
                    m=16, # Standard value, adjust based on performance/memory trade-offs
                    ef_construct=100 # Standard value, adjust based on build time/quality trade-offs
                )
                # Qdrant client >= 1.9 automatically uses appropriate index if not specified
                # We don't explicitly configure fastembed here; it's used during upsert
            )
            logger.info(f"Collection '{collection_name}' created successfully with embedding model '{embedding_model}'.")
        else:
            logger.info(f"Collection '{collection_name}' already exists.")

        # Verify collection status (optional but good practice)
        collection_info = client.get_collection(collection_name=collection_name)
        if collection_info.status != CollectionStatus.GREEN:
            logger.warning(f"Collection '{collection_name}' status is {collection_info.status}. It might be optimizing.")

    except Exception as e:
        logger.exception(f"Failed to initialize or verify Qdrant collection '{collection_name}'.")
        raise

def upsert_posts_to_qdrant(posts: List[Dict[str, str]]):
    """
    Upserts (inserts or updates) Reddit posts into the Qdrant collection,
    generating embeddings for the combined text.
    """
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name
    embedding_model_name = settings.embedding_model_name
    # Define a consistent namespace for generating UUIDs from Reddit IDs
    REDDIT_ID_NAMESPACE = uuid.NAMESPACE_URL # Or define your own custom namespace UUID

    logger.info(f"Starting upsert of {len(posts)} posts to collection '{collection_name}' using model '{embedding_model_name}'...")

    if not posts:
        logger.warning("No posts provided to upsert.")
        return

    try:
        # 1. Initialize the embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        model = SentenceTransformer(embedding_model_name)
        logger.info("Embedding model loaded.")

        # 2. Prepare data for upsert - Generate UUIDs for IDs
        logger.info("Generating UUIDs for post IDs...")
        # Generate UUID5 from the Reddit post ID string for deterministic IDs
        point_ids = [str(uuid.uuid5(REDDIT_ID_NAMESPACE, post["id"])) for post in posts]
        documents_to_embed = [post["combined_text"] for post in posts]
        payloads = [
            {
                "reddit_id": post["id"], # Store original reddit ID in payload if needed
                "title": post["title"],
                "url": post["url"],
                "content": post["content"],
                "comments": post["comments"]
            }
            for post in posts
        ]
        logger.info("UUIDs generated.")


        # 3. Generate embeddings
        logger.info(f"Generating embeddings for {len(documents_to_embed)} documents...")
        vectors = model.encode(documents_to_embed, show_progress_bar=True).tolist()
        logger.info("Embeddings generated.")

        # 4. Create PointStruct objects using UUIDs
        points = [
            PointStruct(id=point_id, vector=vec, payload=pld)
            for point_id, vec, pld in zip(point_ids, vectors, payloads)
        ]
        logger.info(f"Prepared {len(points)} PointStruct objects.")

        # 5. Upsert points to Qdrant
        logger.info(f"Upserting {len(points)} points to collection '{collection_name}'...")
        response = client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True # Wait for the operation to complete
        )

        logger.info(f"Upsert operation status: {response.status}")
        if response.status != UpdateStatus.COMPLETED:
             logger.warning(f"Upsert operation did not complete successfully: {response}")

        logger.info("Posts upserted successfully.")

    except Exception as e:
        logger.exception(f"An error occurred during upsert to Qdrant collection '{collection_name}'.")

def read_posts_from_qdrant():
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name

    try:
        points = client.query_points(collection_name=collection_name)
        logger.info(f"Fetched {points.points.count} points from collection '{collection_name}'.")
        for point in points.points:
            title = point.payload.get('title', 'No Title')
            url = point.payload.get('url', 'No URL')
            content = point.payload.get('content', 'No Content')
            comments = point.payload.get('comments', 'No Comments')
            
            logger.info(f"Point ID: {point.id}\ntitle: {title}\nurl: {url}\ncontent: {content}\ncomments: {comments}\n{'*'*120}\n")
        return points.points
    except Exception as e:
        logger.exception(f"An error occurred while reading posts from Qdrant collection '{collection_name}'.")
        raise

if __name__ == '__main__':
    logger.info("Running qdrant (vector_store) directly for testing...")
    try:
        initialize_qdrant_collection()
        logger.info("Initialization complete. No data upserted in test mode.")
    except Exception as e:
        logger.exception(f"Failed to initialize or test Qdrant during direct execution: {e}")