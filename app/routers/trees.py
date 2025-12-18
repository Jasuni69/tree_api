from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from typing import Optional
from PIL import Image
import io
import os
import uuid
import json

from app.database import get_db
from app.models import Tree, Photo
from app.schemas.tree import TreeCreate, TreeUpdate, TreeResponse, TreeListResponse
from app.services import ReIDService, HealthService, GalleryService
from app.config import settings

router = APIRouter(prefix="/api/v1/trees", tags=["trees"])

reid_service = ReIDService(settings.reid_model_path)
health_service = HealthService(settings.health_model_dir)
gallery_service = GalleryService()


@router.post("/register", response_model=TreeResponse, status_code=201)
async def register_tree(
    address: str,
    tree_number: int,
    file: UploadFile = File(...),
    name: Optional[str] = None,
    species: Optional[str] = None,
    location_lat: Optional[float] = None,
    location_lon: Optional[float] = None,
    notes: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Register new tree with initial photo"""
    # Validate image
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be image")

    try:
        # Create tree
        tree_data = TreeCreate(
            address=address,
            tree_number=tree_number,
            name=name,
            species=species,
            location_lat=location_lat,
            location_lon=location_lon,
            notes=notes
        )
        tree = Tree(**tree_data.model_dump())
        db.add(tree)
        await db.flush()

        # Save image
        os.makedirs(settings.upload_dir, exist_ok=True)
        file_ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(settings.upload_dir, filename)

        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        # Process image
        image = Image.open(io.BytesIO(contents))

        # Extract embedding
        preprocessed = reid_service.preprocess_image(image)
        embedding = reid_service.extract_embedding(preprocessed)
        embedding_list = embedding.tolist()

        # Health assessment
        health_data = health_service.assess_health(image)
        health_json = json.dumps(health_data)

        # Create photo record
        photo = Photo(
            tree_id=tree.id,
            file_path=file_path,
            embedding=embedding_list,
            health_assessment=health_json
        )
        db.add(photo)

        await db.commit()
        await db.refresh(tree)

        # Load with photos
        result = await db.execute(
            select(Tree).where(Tree.id == tree.id).options(selectinload(Tree.photos))
        )
        tree_with_photos = result.scalar_one()

        return TreeResponse.model_validate(tree_with_photos)

    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Error registering tree: {str(e)}")


@router.get("", response_model=TreeListResponse)
async def list_trees(
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """Get paginated list of all trees"""
    if page < 1 or page_size < 1:
        raise HTTPException(status_code=400, detail="Invalid pagination parameters")

    # Get total count
    count_result = await db.execute(select(func.count(Tree.id)))
    total = count_result.scalar()

    # Get paginated trees with photos
    offset = (page - 1) * page_size
    result = await db.execute(
        select(Tree)
        .options(selectinload(Tree.photos))
        .offset(offset)
        .limit(page_size)
    )
    trees = result.scalars().all()

    return TreeListResponse(
        trees=[TreeResponse.model_validate(tree) for tree in trees],
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/{tree_id}", response_model=TreeResponse)
async def get_tree(
    tree_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get tree details with all photos"""
    result = await db.execute(
        select(Tree)
        .where(Tree.id == tree_id)
        .options(selectinload(Tree.photos))
    )
    tree = result.scalar_one_or_none()

    if not tree:
        raise HTTPException(status_code=404, detail="Tree not found")

    return TreeResponse.model_validate(tree)


@router.put("/{tree_id}", response_model=TreeResponse)
async def update_tree(
    tree_id: int,
    tree_update: TreeUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update tree metadata"""
    result = await db.execute(
        select(Tree).where(Tree.id == tree_id)
    )
    tree = result.scalar_one_or_none()

    if not tree:
        raise HTTPException(status_code=404, detail="Tree not found")

    # Update fields
    update_data = tree_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(tree, field, value)

    await db.commit()
    await db.refresh(tree)

    # Load with photos
    result = await db.execute(
        select(Tree)
        .where(Tree.id == tree_id)
        .options(selectinload(Tree.photos))
    )
    tree_with_photos = result.scalar_one()

    return TreeResponse.model_validate(tree_with_photos)


@router.delete("/{tree_id}", status_code=204)
async def delete_tree(
    tree_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete tree and all associated photos"""
    result = await db.execute(
        select(Tree)
        .where(Tree.id == tree_id)
        .options(selectinload(Tree.photos))
    )
    tree = result.scalar_one_or_none()

    if not tree:
        raise HTTPException(status_code=404, detail="Tree not found")

    # Delete photo files
    for photo in tree.photos:
        try:
            if os.path.exists(photo.file_path):
                os.remove(photo.file_path)
        except Exception:
            pass  # Continue even if file deletion fails

    # Delete from database (cascade will handle photos)
    await db.delete(tree)
    await db.commit()
