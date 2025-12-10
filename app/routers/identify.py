from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from PIL import Image
import io

from app.database import get_db
from app.schemas.photo import IdentifyResponse, HealthAssessment
from app.services import ReIDService, HealthService, GalleryService
from app.config import settings

router = APIRouter(prefix="/api/v1", tags=["identify"])

reid_service = ReIDService(settings.reid_model_path)
health_service = HealthService(settings.health_model_path)
gallery_service = GalleryService()


@router.post("/identify", response_model=IdentifyResponse)
async def identify_tree(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Identify tree from uploaded photo
    Returns matched tree ID (if found) + health assessment
    """
    # Validate image
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be image")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Extract embedding
        preprocessed = reid_service.preprocess_image(image)
        query_embedding = reid_service.extract_embedding(preprocessed)

        # Find match in gallery
        match_result = await gallery_service.find_best_match(query_embedding, db)

        # Assess health
        health_data = health_service.assess_health(image)
        health_assessment = HealthAssessment(**health_data)

        if match_result:
            tree_id, confidence = match_result
            return IdentifyResponse(
                matched=True,
                tree_id=tree_id,
                confidence=confidence,
                health_assessment=health_assessment,
                message=f"Tree matched with confidence {confidence:.2f}"
            )
        else:
            return IdentifyResponse(
                matched=False,
                health_assessment=health_assessment,
                message="No matching tree found in gallery"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
