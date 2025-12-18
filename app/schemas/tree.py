from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from app.schemas.photo import PhotoResponse


class TreeBase(BaseModel):
    address: str = Field(..., min_length=1, max_length=512)
    tree_number: int = Field(..., ge=1)  # Tree 1, 2, 3... at address
    name: Optional[str] = Field(None, max_length=255)
    location_lat: Optional[float] = None
    location_lon: Optional[float] = None
    species: Optional[str] = Field(None, max_length=255)
    notes: Optional[str] = None


class TreeCreate(TreeBase):
    pass


class TreeUpdate(BaseModel):
    address: Optional[str] = Field(None, min_length=1, max_length=512)
    tree_number: Optional[int] = Field(None, ge=1)
    name: Optional[str] = Field(None, max_length=255)
    location_lat: Optional[float] = None
    location_lon: Optional[float] = None
    species: Optional[str] = Field(None, max_length=255)
    notes: Optional[str] = None


class TreeResponse(TreeBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    photos: List[PhotoResponse] = []

    class Config:
        from_attributes = True


class TreeListResponse(BaseModel):
    trees: List[TreeResponse]
    total: int
    page: int
    page_size: int


class TreeIdentifier(BaseModel):
    """For looking up tree by address + tree_number"""
    address: str
    tree_number: int
