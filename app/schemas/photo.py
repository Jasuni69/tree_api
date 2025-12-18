from pydantic import BaseModel, field_validator
from typing import Optional, Dict, Any, List
from datetime import datetime
import json


class HealthAssessment(BaseModel):
    probabilities: Dict[str, float]
    predictions: Dict[str, bool]
    confidence: Dict[str, float]
    overall_confidence: float
    error: Optional[str] = None


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

    @field_validator('health_assessment', mode='before')
    @classmethod
    def parse_health_assessment(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    class Config:
        from_attributes = True


class IdentifyResponse(BaseModel):
    matched: bool
    tree_id: Optional[int] = None
    confidence: Optional[float] = None
    health_assessment: Optional[HealthAssessment] = None
    message: str
    processing_time_ms: Optional[int] = None


class TopKMatch(BaseModel):
    """Single match result in top-K response."""
    tree_id: int
    confidence: float
    rank: int


class TopKResponse(BaseModel):
    """Response for top-K matching endpoint."""
    matches: List[TopKMatch]
    total_candidates: int
    processing_time_ms: int
