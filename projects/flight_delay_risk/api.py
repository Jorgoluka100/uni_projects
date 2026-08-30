"""FastAPI service for the verified flight-delay risk model."""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.inference import load_release_artifacts, score_records

app = FastAPI(
    title="Flight Delay Risk API",
    version="1.0.0",
    description="Schedule-time risk scoring for the verified 2026 flight-delay portfolio model.",
)


class FlightRequest(BaseModel):
    flight_date: str
    carrier: str = Field(min_length=1, max_length=8)
    origin: str = Field(min_length=3, max_length=4)
    dest: str = Field(min_length=3, max_length=4)
    crs_dep_minutes: Annotated[int, Field(ge=0, le=1439)]
    crs_arr_minutes: Annotated[int, Field(ge=0, le=1439)]
    crs_elapsed_minutes: Annotated[float, Field(gt=0, le=1500)]
    distance_miles: Annotated[float, Field(gt=0, le=12000)]

    @field_validator("carrier", "origin", "dest")
    @classmethod
    def normalize_codes(cls, value: str) -> str:
        return value.strip().upper()


class BatchRequest(BaseModel):
    flights: list[FlightRequest] = Field(min_length=1, max_length=1000)


@lru_cache(maxsize=1)
def _release_state():
    return load_release_artifacts()


def _require_release():
    try:
        return _release_state()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Verified model release is not available: {exc}",
        ) from exc


@app.get("/health")
def health() -> dict[str, object]:
    try:
        _, metadata = _release_state()
    except Exception as exc:
        return {"status": "degraded", "model_loaded": False, "detail": str(exc)}

    return {
        "status": "ok",
        "model_loaded": True,
        "verification_pass": metadata.get("verification_pass"),
        "data_year": metadata.get("data_year"),
        "task": metadata.get("task"),
    }


@app.post("/predict")
def predict(request: FlightRequest) -> dict[str, object]:
    model, metadata = _require_release()
    record = request.model_dump()
    score = float(score_records(model, [record])[0])
    threshold = metadata.get("validation_threshold")
    return {
        "risk_score": score,
        "review_threshold": threshold,
        "flag_for_review": bool(threshold is not None and score >= float(threshold)),
        "model_task": metadata.get("task"),
        "data_year": metadata.get("data_year"),
    }


@app.post("/predict-batch")
def predict_batch(request: BatchRequest) -> dict[str, object]:
    model, metadata = _require_release()
    records = [flight.model_dump() for flight in request.flights]
    scores = score_records(model, records)
    threshold = metadata.get("validation_threshold")
    items = [
        {
            "risk_score": float(score),
            "flag_for_review": bool(
                threshold is not None and float(score) >= float(threshold)
            ),
        }
        for score in scores
    ]
    return {
        "count": len(items),
        "review_threshold": threshold,
        "predictions": items,
    }
