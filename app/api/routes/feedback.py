"""
POST /feedback — Record user quality signals on answers.

Feedback is written as JSON-lines to the feedback directory so it can be
consumed for fine-tuning, analytics, or prompt iteration without a database
dependency. Each record includes a UUID, the session context, the
rating (thumbs_up / thumbs_down), and an optional comment.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.api.schemas import FeedbackRequest, FeedbackResponse
from app.core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse, summary="Submit answer feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Store a thumbs-up / thumbs-down signal alongside the Q&A pair.
    Feedback is appended to a JSONL file named by date (feedback_YYYY-MM-DD.jsonl).
    """
    settings = get_settings()
    feedback_dir = Path(settings.feedback_dir)
    feedback_dir.mkdir(parents=True, exist_ok=True)

    feedback_id = str(uuid.uuid4())
    record = {
        "feedback_id": feedback_id,
        "session_id": request.session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": request.question,
        "answer": request.answer,
        "rating": request.rating,
        "comment": request.comment,
    }

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    feedback_file = feedback_dir / f"feedback_{date_str}.jsonl"

    try:
        with open(feedback_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.info("Feedback %s saved: %s", feedback_id, request.rating)
    except Exception as exc:
        logger.exception("Failed to write feedback: %s", exc)
        raise HTTPException(status_code=500, detail="Could not save feedback.")

    return FeedbackResponse(
        message="Feedback recorded. Thank you!",
        feedback_id=feedback_id,
    )
