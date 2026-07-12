"""
API Routes for ECG Arrhythmia Detection
==========================================
FastAPI endpoints for ECG analysis.
"""

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.database import get_db, init_db
from src.api.models import AbnormalSegment, Analysis, BeatPrediction, Recording
from src.api.schemas import (
    AnalysisResponse,
    AnalysisStatus,
    HealthResponse,
    SampleRecordRequest,
)
from src.inference.pipeline import ECGAnalysisPipeline
from src.inference.predict import get_classifier

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize database on first use
_db_initialized = False


def ensure_db():
    """Ensure database tables exist."""
    global _db_initialized
    if not _db_initialized:
        init_db()
        _db_initialized = True


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    from src.api.database import check_database_connection
    classifier = get_classifier()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=classifier.is_loaded,
        database_connected=check_database_connection(),
    )


@router.get("/models")
async def list_models():
    """List available models."""
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    models = []
    model_dir = PROJECT_ROOT / "models"

    # Check CNN baseline
    baseline_path = model_dir / "best_model.pt"
    if baseline_path.exists():
        models.append({
            "name": "cnn_baseline",
            "description": "1D CNN baseline model (671K params)",
            "available": True,
        })

    # Check CNN+LSTM
    lstm_path = model_dir / "cnn_lstm" / "best_model.pt"
    if lstm_path.exists():
        models.append({
            "name": "cnn_lstm",
            "description": "CNN+LSTM model (31K params)",
            "available": True,
        })

    return {"models": models}


@router.post("/analyze/sample", response_model=AnalysisResponse)
async def analyze_sample(request: SampleRecordRequest, db: Session = Depends(get_db)):
    """
    Analyze a sample record from the MIT-BIH database.

    Use record_name="100" for a quick demo.
    """
    ensure_db()

    try:
        # Run analysis
        pipeline = ECGAnalysisPipeline(model_name=request.model_name)
        results = pipeline.analyze_mitdb_record(request.record_name)

        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])

        # Create recording
        recording = Recording(
            id=str(uuid.uuid4()),
            filename=f"mitdb_{request.record_name}",
            record_name=request.record_name,
            num_samples=0,  # Will be filled from results
            duration_sec=0,
            sampling_rate=360,
        )
        db.add(recording)
        db.flush()

        # Create analysis
        analysis = Analysis(
            id=str(uuid.uuid4()),
            recording_id=recording.id,
            model_name="cnn_baseline",
            status="completed",
            total_beats=results["total_beats"],
            normal_beats=results["normal_beats"],
            abnormal_beats=results["abnormal_beats"],
            overall_confidence=results["overall_confidence"],
            flagged_for_review=results["flagged_for_review"],
            class_distribution=results["class_distribution"],
            processing_time_sec=results["processing_time_sec"],
        )
        db.add(analysis)
        db.flush()

        # Save beat predictions
        for beat in results["beat_predictions"]:
            bp = BeatPrediction(
                id=str(uuid.uuid4()),
                analysis_id=analysis.id,
                beat_index=beat["beat_index"],
                sample_index=beat["sample_index"],
                timestamp_sec=beat["timestamp_sec"],
                predicted_class=beat["predicted_class"],
                confidence=beat["confidence"],
                probabilities=beat["probabilities"],
            )
            db.add(bp)

        # Save abnormal segments
        for seg in results["abnormal_segments"]:
            abn_seg = AbnormalSegment(
                id=str(uuid.uuid4()),
                analysis_id=analysis.id,
                start_time_sec=seg["start_time_sec"],
                end_time_sec=seg["end_time_sec"],
                duration_sec=seg["duration_sec"],
                abnormal_beat_indices=seg["abnormal_beat_indices"],
                dominant_class=seg["dominant_class"],
                avg_confidence=seg["avg_confidence"],
                num_beats=seg.get("num_beats", len(seg["abnormal_beat_indices"])),
            )
            db.add(abn_seg)

        db.commit()

        return AnalysisResponse(
            analysis_id=analysis.id,
            recording_id=recording.id,
            status=AnalysisStatus.COMPLETED,
            summary={
                "total_beats": results["total_beats"],
                "normal_beats": results["normal_beats"],
                "abnormal_beats": results["abnormal_beats"],
                "class_distribution": results["class_distribution"],
                "abnormal_segments": results["abnormal_segments"],
                "overall_confidence": results["overall_confidence"],
                "flagged_for_review": results["flagged_for_review"],
            },
            beat_predictions=results["beat_predictions"][:100],  # Limit for response
            created_at=analysis.created_at or datetime.utcnow(),
            processing_time_sec=results["processing_time_sec"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Analysis failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyses")
async def list_analyses(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db),
):
    """List all analyses."""
    ensure_db()

    analyses = db.query(Analysis).offset(skip).limit(limit).all()
    return {
        "analyses": [
            {
                "analysis_id": a.id,
                "recording_id": a.recording_id,
                "model_name": a.model_name,
                "status": a.status,
                "total_beats": a.total_beats,
                "abnormal_beats": a.abnormal_beats,
                "flagged_for_review": a.flagged_for_review,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in analyses
        ]
    }


@router.get("/analyses/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
    """Get analysis by ID."""
    ensure_db()

    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Get beat predictions
    beats = db.query(BeatPrediction).filter(
        BeatPrediction.analysis_id == analysis_id
    ).order_by(BeatPrediction.beat_index).all()

    # Get abnormal segments
    segments = db.query(AbnormalSegment).filter(
        AbnormalSegment.analysis_id == analysis_id
    ).all()

    return AnalysisResponse(
        analysis_id=analysis.id,
        recording_id=analysis.recording_id,
        status=AnalysisStatus.COMPLETED,
        summary={
            "total_beats": analysis.total_beats,
            "normal_beats": analysis.normal_beats,
            "abnormal_beats": analysis.abnormal_beats,
            "class_distribution": analysis.class_distribution or [],
            "abnormal_segments": [
                {
                    "start_time_sec": s.start_time_sec,
                    "end_time_sec": s.end_time_sec,
                    "duration_sec": s.duration_sec,
                    "abnormal_beat_indices": s.abnormal_beat_indices,
                    "dominant_class": s.dominant_class,
                    "avg_confidence": s.avg_confidence,
                    "num_beats": s.num_beats or len(s.abnormal_beat_indices),
                }
                for s in segments
            ],
            "overall_confidence": analysis.overall_confidence,
            "flagged_for_review": analysis.flagged_for_review,
        },
        beat_predictions=[
            {
                "beat_index": b.beat_index,
                "sample_index": b.sample_index,
                "timestamp_sec": b.timestamp_sec,
                "predicted_class": b.predicted_class,
                "confidence": b.confidence,
                "probabilities": b.probabilities,
            }
            for b in beats[:100]  # Limit for response
        ],
        created_at=analysis.created_at or datetime.utcnow(),
        processing_time_sec=analysis.processing_time_sec,
    )


@router.get("/analyses/{analysis_id}/beats")
async def get_beat_predictions(
    analysis_id: str,
    offset: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """Get beat predictions with pagination."""
    ensure_db()

    beats = db.query(BeatPrediction).filter(
        BeatPrediction.analysis_id == analysis_id
    ).order_by(BeatPrediction.beat_index).offset(offset).limit(limit).all()

    total = db.query(BeatPrediction).filter(
        BeatPrediction.analysis_id == analysis_id
    ).count()

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "beats": [
            {
                "beat_index": b.beat_index,
                "sample_index": b.sample_index,
                "timestamp_sec": b.timestamp_sec,
                "predicted_class": b.predicted_class,
                "confidence": b.confidence,
                "probabilities": b.probabilities,
            }
            for b in beats
        ],
    }


@router.delete("/analyses/{analysis_id}")
async def delete_analysis(analysis_id: str, db: Session = Depends(get_db)):
    """Delete an analysis and its results."""
    ensure_db()

    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    db.delete(analysis)
    db.commit()

    return {"message": "Analysis deleted", "analysis_id": analysis_id}
