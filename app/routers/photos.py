from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from PIL import Image
import io
import os
import uuid
import json

from app.database import get_db
from app.models import Photo, Tree
from app.schemas.photo import PhotoResponse
from app.services import get_reid_service, get_health_service, GalleryService
from app.config import settings

router = APIRouter(prefix="/api/v1", tags=["photos"])

gallery_service = GalleryService()


@router.get("/trees/{tree_id}/photos", response_model=List[PhotoResponse])
async def get_tree_photos(
    tree_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get all photos for a tree"""
    tree_result = await db.execute(
        select(Tree).where(Tree.id == tree_id)
    )
    tree = tree_result.scalar_one_or_none()
    if not tree:
        raise HTTPException(status_code=404, detail="Tree not found")

    result = await db.execute(
        select(Photo).where(Photo.tree_id == tree_id).order_by(Photo.created_at.desc())
    )
    photos = result.scalars().all()

    return [PhotoResponse.model_validate(photo) for photo in photos]


@router.post("/trees/{tree_id}/photos", response_model=PhotoResponse, status_code=201)
async def add_photo_to_tree(
    tree_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """Add new photo to existing tree"""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be image")

    tree_result = await db.execute(
        select(Tree).where(Tree.id == tree_id)
    )
    tree = tree_result.scalar_one_or_none()
    if not tree:
        raise HTTPException(status_code=404, detail="Tree not found")

    file_path = None
    try:
        os.makedirs(settings.upload_dir, exist_ok=True)
        file_ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(settings.upload_dir, filename)

        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        image = Image.open(io.BytesIO(contents))

        # Extract embedding with TTA for better accuracy
        reid_service = get_reid_service()
        embedding = reid_service.extract_embedding(image, use_tta=True)
        embedding_list = embedding.tolist()

        # Health assessment with TTA
        health_service = get_health_service()
        health_data = health_service.assess_health(image, use_tta=True)
        health_json = json.dumps(health_data)

        photo = Photo(
            tree_id=tree_id,
            file_path=file_path,
            embedding=embedding_list,
            health_assessment=health_json
        )
        db.add(photo)
        await db.commit()
        await db.refresh(photo)

        return PhotoResponse.model_validate(photo)

    except Exception as e:
        await db.rollback()
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error adding photo: {str(e)}")


@router.delete("/photos/{photo_id}", status_code=204)
async def delete_photo(
    photo_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete a photo"""
    result = await db.execute(
        select(Photo).where(Photo.id == photo_id)
    )
    photo = result.scalar_one_or_none()

    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")

    try:
        if os.path.exists(photo.file_path):
            os.remove(photo.file_path)
    except Exception:
        pass

    await db.delete(photo)
    await db.commit()
