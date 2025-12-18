import numpy as np
from typing import List, Tuple, Optional
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.photo import Photo
from app.config import settings


class GalleryService:
    """
    Gallery service with native pgvector similarity search.
    Uses cosine distance for embedding comparison.
    """

    def __init__(self, similarity_threshold: float = None):
        self.similarity_threshold = similarity_threshold or settings.similarity_threshold

    async def find_best_match(
        self,
        query_embedding: np.ndarray,
        db: AsyncSession,
        threshold: float = None
    ) -> Optional[Tuple[int, float]]:
        """
        Find best matching tree using pgvector cosine similarity.

        Args:
            query_embedding: Embedding vector of query image
            db: Database session
            threshold: Minimum similarity threshold (default from config)

        Returns:
            Tuple of (tree_id, similarity_score) or None if no match above threshold
        """
        # Convert to list for pgvector
        embedding_list = query_embedding.tolist()
        min_threshold = threshold if threshold is not None else self.similarity_threshold

        # pgvector cosine distance: <=> operator (0 = identical, 2 = opposite)
        # Convert to similarity: 1 - (distance / 2)
        query = text("""
            SELECT
                tree_id,
                1 - (embedding <=> :embedding) / 2 as similarity
            FROM photos
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> :embedding
            LIMIT 1
        """)

        result = await db.execute(query, {"embedding": str(embedding_list)})
        row = result.fetchone()

        if row and row.similarity >= min_threshold:
            return (row.tree_id, float(row.similarity))

        return None

    async def get_top_k_matches(
        self,
        query_embedding: np.ndarray,
        db: AsyncSession,
        k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Get top K matching trees using pgvector.

        Args:
            query_embedding: Embedding vector of query image
            db: Database session
            k: Number of top matches to return

        Returns:
            List of (tree_id, similarity_score) tuples sorted by similarity
        """
        embedding_list = query_embedding.tolist()

        query = text("""
            SELECT 
                tree_id,
                1 - (embedding <=> :embedding) / 2 as similarity
            FROM photos
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> :embedding
            LIMIT :k
        """)

        result = await db.execute(query, {"embedding": str(embedding_list), "k": k})
        rows = result.fetchall()

        return [(row.tree_id, float(row.similarity)) for row in rows]

    async def find_matches_above_threshold(
        self,
        query_embedding: np.ndarray,
        db: AsyncSession,
        threshold: float = None
    ) -> List[Tuple[int, float]]:
        """
        Find all matches above similarity threshold.

        Args:
            query_embedding: Embedding vector
            db: Database session
            threshold: Minimum similarity (default from config)

        Returns:
            List of (tree_id, similarity) tuples
        """
        threshold = threshold or self.similarity_threshold
        embedding_list = query_embedding.tolist()

        # Max cosine distance for threshold: 2 * (1 - threshold)
        max_distance = 2 * (1 - threshold)

        query = text("""
            SELECT 
                tree_id,
                1 - (embedding <=> :embedding) / 2 as similarity
            FROM photos
            WHERE embedding IS NOT NULL
              AND embedding <=> :embedding < :max_dist
            ORDER BY embedding <=> :embedding
        """)

        result = await db.execute(query, {
            "embedding": str(embedding_list),
            "max_dist": max_distance
        })
        rows = result.fetchall()

        return [(row.tree_id, float(row.similarity)) for row in rows]
