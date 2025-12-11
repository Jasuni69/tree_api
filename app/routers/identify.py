from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from PIL import Image
import io
import time
import logging
from typing import List

from app.database import get_db
from app.schemas.photo import IdentifyResponse, HealthAssessment, TopKMatch, TopKResponse
from app.services import ReIDService, HealthService, GalleryService
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["identify"])

reid_service = ReIDService(settings.reid_model_path)
health_service = HealthService(settings.health_model_dir)
gallery_service = GalleryService()

# Image validation constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MIN_DIMENSION = 100  # Minimum width/height
MAX_DIMENSION = 4096  # Maximum width/height
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}


def validate_image(contents: bytes, image: Image.Image) -> None:
    """Validate image size, dimensions, and format."""
    # Check file size
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )

    # Check format
    if image.format and image.format.upper() not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {image.format}. Allowed: {', '.join(ALLOWED_FORMATS)}"
        )

    # Check dimensions
    width, height = image.size
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Image too small. Minimum: {MIN_DIMENSION}x{MIN_DIMENSION}px"
        )
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large. Maximum: {MAX_DIMENSION}x{MAX_DIMENSION}px"
        )


@router.post("/identify", response_model=IdentifyResponse)
async def identify_tree(
    file: UploadFile = File(...),
    threshold: float = Query(None, ge=0.0, le=1.0, description="Similarity threshold (default from config)"),
    db: AsyncSession = Depends(get_db)
):
    """
    Identify tree from uploaded photo.

    Uses Test-Time Augmentation (TTA) for best accuracy.

    - **threshold**: Minimum similarity score to consider a match (0.0-1.0)

    Returns matched tree ID (if found) + health assessment.
    """
    start_time = time.time()

    # Validate content type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Validate image
        validate_image(contents, image)

        # Extract embedding
        embed_start = time.time()
        preprocessed = reid_service.preprocess_image(image)
        query_embedding = reid_service.extract_embedding(preprocessed, use_tta=True)
        embed_time = time.time() - embed_start

        # Find match in gallery
        search_start = time.time()
        match_threshold = threshold if threshold is not None else settings.similarity_threshold
        match_result = await gallery_service.find_best_match(
            query_embedding, db, threshold=match_threshold
        )
        search_time = time.time() - search_start

        # Assess health
        health_start = time.time()
        health_data = health_service.assess_health(image)
        health_assessment = HealthAssessment(**health_data)
        health_time = time.time() - health_start

        total_time = time.time() - start_time
        logger.info(f"Identify request: embed={embed_time:.3f}s, search={search_time:.3f}s, health={health_time:.3f}s, total={total_time:.3f}s")

        if match_result:
            tree_id, confidence = match_result
            return IdentifyResponse(
                matched=True,
                tree_id=tree_id,
                confidence=confidence,
                health_assessment=health_assessment,
                message=f"Tree matched with confidence {confidence:.2f}",
                processing_time_ms=int(total_time * 1000)
            )
        else:
            return IdentifyResponse(
                matched=False,
                health_assessment=health_assessment,
                message="No matching tree found in gallery",
                processing_time_ms=int(total_time * 1000)
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in identify: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@router.post("/identify/topk", response_model=TopKResponse)
async def identify_tree_topk(
    file: UploadFile = File(...),
    k: int = Query(5, ge=1, le=20, description="Number of top matches to return"),
    threshold: float = Query(None, ge=0.0, le=1.0, description="Minimum similarity threshold"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get top K matching trees for an image.

    Uses Test-Time Augmentation (TTA) for best accuracy.
    Useful when confidence is borderline or you want to show alternatives.

    - **k**: Number of matches to return (1-20)
    - **threshold**: Only return matches above this similarity
    """
    start_time = time.time()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        validate_image(contents, image)

        # Extract embedding
        preprocessed = reid_service.preprocess_image(image)
        query_embedding = reid_service.extract_embedding(preprocessed, use_tta=True)

        # Get top K matches
        matches = await gallery_service.get_top_k_matches(query_embedding, db, k=k)

        # Filter by threshold if specified
        min_threshold = threshold if threshold is not None else 0.0
        filtered_matches = [
            TopKMatch(tree_id=tree_id, confidence=conf, rank=i+1)
            for i, (tree_id, conf) in enumerate(matches)
            if conf >= min_threshold
        ]

        total_time = time.time() - start_time
        logger.info(f"TopK request: k={k}, matches={len(filtered_matches)}, time={total_time:.3f}s")

        return TopKResponse(
            matches=filtered_matches,
            total_candidates=len(matches),
            processing_time_ms=int(total_time * 1000)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in identify/topk: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
