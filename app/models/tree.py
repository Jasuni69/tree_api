from sqlalchemy import Column, Integer, String, Float, DateTime, Text, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class Tree(Base):
    __tablename__ = "trees"
    __table_args__ = (
        UniqueConstraint('address', 'tree_number', name='uq_address_tree_number'),
    )

    id = Column(Integer, primary_key=True, index=True)

    # Unique identifier: address + tree_number
    address = Column(String(512), nullable=False, index=True)
    tree_number = Column(Integer, nullable=False)  # Tree 1, 2, 3... at this address

    # Optional display name
    name = Column(String(255), nullable=True)

    # GPS coordinates for precise location
    location_lat = Column(Float, nullable=True)
    location_lon = Column(Float, nullable=True)

    species = Column(String(255), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    photos = relationship("Photo", back_populates="tree", cascade="all, delete-orphan")
