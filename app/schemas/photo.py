from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime


class HealthAssessment(BaseModel):
    status: str  # healthy, diseased, pest_damage, etc
    confidence: float
    details: Optional[Dict[str, Any]] = None


class PhotoBase(BaseModel):
    captured_at: Optional[datetime] = None


class PhotoCreate(PhotoBase):
    tree_id: int


class PhotoResponse(PhotoBase):
    id: int
    tree_id: int
    file_path: str
    health_assessment: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class IdentifyResponse(BaseModel):
    matched: bool
    tree_id: Optional[int] = None
    confidence: Optional[float] = None
    health_assessment: Optional[HealthAssessment] = None
    message: str
