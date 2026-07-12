"""
Pydantic Schemas for ECG Arrhythmia API
==========================================
Request and response models for the FastAPI backend.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# --- Enums ---

class AAMIClass(str, Enum):
    """AAMI beat classification classes."""
    N = "N"
    S = "S"
    V = "V"
    F = "F"
    Q = "Q"


class AnalysisStatus(str, Enum):
    """Analysis status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# --- Beat-level schemas ---

class BeatPrediction(BaseModel):
    """Single beat prediction."""
    beat_index: int = Field(..., description="Index of the beat in the recording")
    sample_index: int = Field(..., description="Sample index in the original signal")
    timestamp_sec: float = Field(..., description="Time in seconds from start")
    predicted_class: AAMIClass = Field(..., description="Predicted AAMI class")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probabilities: dict[str, float] = Field(..., description="Class probabilities")


class BeatAnnotation(BaseModel):
    """Original annotation for a beat."""
    beat_index: int
    sample_index: int
    timestamp_sec: float
    original_symbol: str
    mapped_class: AAMIClass


# --- Recording-level schemas ---

class ClassDistribution(BaseModel):
    """Distribution of beat classes."""
    class_name: AAMIClass
    count: int
    percentage: float


class AbnormalSegment(BaseModel):
    """Abnormal segment with clustered abnormal beats."""
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    abnormal_beat_indices: list[int] = Field(..., description="Beat indices in segment")
    dominant_class: AAMIClass
    avg_confidence: float
    num_beats: int


class AnalysisSummary(BaseModel):
    """Summary of the full analysis."""
    total_beats: int
    normal_beats: int
    abnormal_beats: int
    class_distribution: list[ClassDistribution]
    abnormal_segments: list[AbnormalSegment]
    overall_confidence: float
    flagged_for_review: bool = Field(
        ..., description="True if significant abnormalities detected"
    )


# --- Request schemas ---

class AnalyzeRequest(BaseModel):
    """Request to analyze an ECG recording."""
    record_name: str | None = Field(
        None, description="MIT-BIH record name (e.g., '100') for demo"
    )
    model_name: str = Field(
        "cnn_baseline", description="Model to use: cnn_baseline or cnn_lstm"
    )


class SampleRecordRequest(BaseModel):
    """Request to analyze a sample record from MIT-BIH."""
    record_name: str = Field(
        "100", description="MIT-BIH record name to analyze"
    )
    model_name: str = Field(
        "cnn_baseline", description="Model to use: cnn_baseline or cnn_lstm"
    )


# --- Response schemas ---

class UploadResponse(BaseModel):
    """Response after uploading an ECG file."""
    recording_id: str
    filename: str
    num_samples: int
    duration_sec: float
    sampling_rate: int
    message: str


class AnalysisResponse(BaseModel):
    """Full analysis response."""
    analysis_id: str
    recording_id: str
    status: AnalysisStatus
    summary: AnalysisSummary | None = None
    beat_predictions: list[BeatPrediction] | None = None
    created_at: datetime
    processing_time_sec: float | None = None


class ReportResponse(BaseModel):
    """Report response with detailed results."""
    analysis_id: str
    recording_id: str
    status: AnalysisStatus
    summary: AnalysisSummary
    beat_predictions: list[BeatPrediction]
    original_annotations: list[BeatAnnotation] | None = None
    created_at: datetime


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    model_loaded: bool
    database_connected: bool


# --- Error schemas ---

class ErrorResponse(BaseModel):
    """Error response."""
    detail: str
    error_code: str | None = None
