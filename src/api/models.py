"""
SQLAlchemy Database Models for ECG Arrhythmia API
===================================================
"""

import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from src.api.database import Base


def generate_uuid() -> str:
    """Generate a unique UUID."""
    return str(uuid.uuid4())


class Recording(Base):
    """ECG Recording metadata."""
    __tablename__ = "recordings"

    id = Column(String, primary_key=True, default=generate_uuid)
    filename = Column(String, nullable=False)
    record_name = Column(String, nullable=True)  # MIT-BIH record name
    num_samples = Column(Integer, nullable=False)
    duration_sec = Column(Float, nullable=False)
    sampling_rate = Column(Integer, nullable=False, default=360)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    analyses = relationship("Analysis", back_populates="recording", cascade="all, delete-orphan")


class Analysis(Base):
    """ECG Analysis results."""
    __tablename__ = "analyses"

    id = Column(String, primary_key=True, default=generate_uuid)
    recording_id = Column(String, ForeignKey("recordings.id"), nullable=False)
    model_name = Column(String, nullable=False, default="cnn_baseline")
    status = Column(String, nullable=False, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    processing_time_sec = Column(Float, nullable=True)

    # Summary
    total_beats = Column(Integer, nullable=True)
    normal_beats = Column(Integer, nullable=True)
    abnormal_beats = Column(Integer, nullable=True)
    overall_confidence = Column(Float, nullable=True)
    flagged_for_review = Column(Boolean, default=False)

    # Class distribution (stored as JSON)
    class_distribution = Column(JSON, nullable=True)

    # Relationships
    recording = relationship("Recording", back_populates="analyses")
    beat_predictions = relationship("BeatPrediction", back_populates="analysis", cascade="all, delete-orphan")
    abnormal_segments = relationship("AbnormalSegment", back_populates="analysis", cascade="all, delete-orphan")


class BeatPrediction(Base):
    """Individual beat prediction."""
    __tablename__ = "beat_predictions"

    id = Column(String, primary_key=True, default=generate_uuid)
    analysis_id = Column(String, ForeignKey("analyses.id"), nullable=False)
    beat_index = Column(Integer, nullable=False)
    sample_index = Column(Integer, nullable=False)
    timestamp_sec = Column(Float, nullable=False)
    predicted_class = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    probabilities = Column(JSON, nullable=False)  # {N: 0.9, S: 0.05, ...}

    # Relationships
    analysis = relationship("Analysis", back_populates="beat_predictions")


class AbnormalSegment(Base):
    """Clustered abnormal beat segment."""
    __tablename__ = "abnormal_segments"

    id = Column(String, primary_key=True, default=generate_uuid)
    analysis_id = Column(String, ForeignKey("analyses.id"), nullable=False)
    start_time_sec = Column(Float, nullable=False)
    end_time_sec = Column(Float, nullable=False)
    duration_sec = Column(Float, nullable=False)
    abnormal_beat_indices = Column(JSON, nullable=False)  # [10, 11, 12, ...]
    dominant_class = Column(String, nullable=False)
    avg_confidence = Column(Float, nullable=False)
    num_beats = Column(Integer, nullable=True)

    # Relationships
    analysis = relationship("Analysis", back_populates="abnormal_segments")
