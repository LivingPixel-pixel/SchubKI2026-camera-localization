"""Validated API input settings. These defaults are development choices."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    feature_backend: Literal["sift", "orb", "xfeat_mnn", "xfeat_lighterglue"] = "sift"
    learned_match_threshold: float = Field(default=0.1, ge=0, le=1)
    motion_model: Literal["yaw", "so3"] = "yaw"
    max_features: int = Field(default=1800, ge=100, le=5000)
    ratio_test: float = Field(default=0.75, ge=0.4, le=0.95)
    ransac_px: float = Field(default=4.0, ge=0.5, le=12)
    min_inliers: int = Field(default=8, ge=6, le=100)
    relocalize_every: int = Field(default=4, ge=1, le=30)
    max_propagation_s: float = Field(default=0.45, ge=0, le=2)
    forecast_horizon_s: float = Field(default=0.1, ge=0.01, le=1)
    forecast_mode: Literal["next_frame", "fixed_horizon"] = "next_frame"
    method: Literal["retrieval", "geometry", "tracking"] = "tracking"
    position_tolerance_m: float = Field(default=1.0, gt=0, le=20)
    heading_tolerance_deg: float = Field(default=2.0, gt=0, le=45)
    rotation_tolerance_deg: float = Field(default=2.0, gt=0, le=45)


class MapRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference_session_ids: list[str] = Field(min_length=1, max_length=5)
    config: Settings = Field(default_factory=Settings)


class CandidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=80)
    map_id: str
    config: Settings = Field(default_factory=Settings)
    model_id: str | None = None


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    candidate_id: str


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    map_id: str
    session_id: str
    config: Settings = Field(default_factory=Settings)
    count: int = Field(default=3, ge=2, le=6)


class TrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    session_id: str


class AuditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str


class BenchmarkCompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feature_backends: list[Literal["sift", "orb", "xfeat_mnn", "xfeat_lighterglue"]] = Field(
        default=["sift", "xfeat_mnn", "xfeat_lighterglue"], min_length=2, max_length=4)
    config: Settings = Field(default_factory=lambda: Settings(method="geometry", max_features=512))
