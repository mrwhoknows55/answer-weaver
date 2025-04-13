import logging
from typing import List, Dict, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, CollectionStatus, UpdateStatus

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

def upsert_posts_to_qdrant(posts: List[Dict[str, str]], batch_size: int = 64):
    """
    Upserts Reddit posts into the Qdrant collection using fastembed for embeddings.
    """
    if not posts:
        logger.warning("No posts provided to upsert.")
        return

    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name
    embedding_model = settings.embedding_model_name # This tells fastembed which model to use implicitly

    logger.info(f"Starting upsert of {len(posts)} posts to collection '{collection_name}' using model '{embedding_model}'...")

    points_to_upsert = []
    documents_to_embed = []
    ids_to_upsert = []

    for post in posts:
        # Qdrant requires integer IDs for HNSW index if not using UUIDs
        # Reddit IDs are strings (base36), we need a stable mapping to int or use UUIDs
        # For simplicity here, we'll try hashing the ID, but UUIDs are generally safer
        # Alternatively, configure Qdrant to accept string IDs if using a different index or Qdrant version supports it
        # Let's use the string ID directly for now, relying on Qdrant >= 1.7 supporting string IDs
        post_id = post['id']
        ids_to_upsert.append(post_id)

        # The text that will be embedded by fastembed
        documents_to_embed.append(post['combined_text'])

        # Prepare payload (metadata) - exclude the text that's being embedded
        payload = {
            "title": post['title'],
            "url": post['url'],
            "content": post['content'], # Store original post content separately if needed
            "comments": post['comments'] # Store comments separately if needed
            # Add any other relevant metadata, e.g., creation time, score
        }
        points_to_upsert.append(models.PointStruct(id=post_id, vector=[0.0]*384, payload=payload)) # Temporary vector, will be replaced

    try:
        # Upsert using client.add which leverages fastembed
        # Note: client.add automatically handles batching internally if needed,
        # but explicitly batching the call might offer finer control or fit specific patterns.
        # For qdrant-client >= 1.7 with fastembed integration:
        response = client.add(
            collection_name=collection_name,
            documents=documents_to_embed,
            ids=ids_to_upsert,
            payload=[p.payload for p in points_to_upsert] # Provide payloads separately
            # `batch_size` within client.add controls internal batching to embedding model
            # The model `embedding_model` is inferred from the collection or defaults
        )

        if response.status == UpdateStatus.COMPLETED:
            logger.info(f"Successfully upserted/updated {len(posts)} points into '{collection_name}'.")
        else:
            logger.error(f"Qdrant upsert finished with status: {response.status}. Issues might have occurred.")

    except Exception as e:
        logger.exception(f"An error occurred during upsert to Qdrant collection '{collection_name}'.")

if __name__ == '__main__':
    # Example usage when running this script directly (for testing)
    logger.info("Running qdrant (vector_store) directly for testing...")
    # This part might fail now due to relative import change if run directly
    # Consider running via `python -m src.data_storage.qdrant` for testing
    # or temporarily changing import back for direct testing
    try:
        # Need to ensure settings are loaded, which requires config to run
        from ..config import settings # Re-import for direct execution context might be tricky
        initialize_qdrant_collection()
        logger.info("Initialization complete. No data upserted in test mode.")
        # Example of upserting dummy data (optional)
        # dummy_posts = [{"id": "test1", "title": "Test Post", "url": "http://example.com", "content": "Test content", "comments": "Test comment", "combined_text": "Test Post Test content Test comment"}]
        # upsert_posts_to_qdrant(dummy_posts)
    except Exception as e:
        logger.exception(f"Failed to initialize or test Qdrant during direct execution: {e}")