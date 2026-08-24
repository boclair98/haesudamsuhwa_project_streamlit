from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "ro_history.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MANIFEST_PATH = MODEL_DIR / "manifest.json"

NUMERIC_COLUMNS = [
    "temperature_c",
    "ph",
    "pressure_stage1_bar",
    "pressure_stage2_bar",
    "tds_stage1_mg_l",
    "tds_stage2_mg_l",
    "turbidity_ntu",
    "cod_mg_l",
    "total_n_mg_l",
    "total_p_mg_l",
    "sec_kwh_m3",
]

REQUIRED_COLUMNS = ["timestamp", *NUMERIC_COLUMNS]


class ResourceError(RuntimeError):
    """Raised when a trusted dashboard resource cannot be prepared safely."""


@dataclass(frozen=True)
class DataProfile:
    source_rows: int
    usable_rows: int
    invalid_timestamp_rows: int
    incomplete_rows: int
    duplicate_rows_removed: int
    exact_hour_rows: int
    zero_counts: dict[str, int]

    @property
    def exact_hour_share(self) -> float:
        return self.exact_hour_rows / self.usable_rows if self.usable_rows else 0.0


@dataclass(frozen=True)
class ModelBundle:
    pressure: Any
    sec: Any
    manifest: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_history(path: Path = DATA_PATH) -> tuple[pd.DataFrame, DataProfile]:
    if not path.is_file():
        raise ResourceError(f"정제 공정 데이터가 없습니다: {path.name}")

    try:
        frame = pd.read_csv(path, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - pandas supplies the detail
        raise ResourceError(f"{path.name}을 읽지 못했습니다: {exc}") from exc

    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ResourceError(f"필수 데이터 열이 없습니다: {', '.join(missing)}")

    source_rows = len(frame)
    frame = frame[REQUIRED_COLUMNS].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    invalid_timestamp_rows = int(frame["timestamp"].isna().sum())

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    incomplete_mask = frame[NUMERIC_COLUMNS].isna().any(axis=1)
    incomplete_rows = int((incomplete_mask & frame["timestamp"].notna()).sum())
    frame = frame.loc[frame["timestamp"].notna() & ~incomplete_mask].copy()
    frame.sort_values("timestamp", inplace=True)

    duplicate_rows_removed = int(frame.duplicated("timestamp", keep="last").sum())
    frame.drop_duplicates("timestamp", keep="last", inplace=True)
    frame.reset_index(drop=True, inplace=True)

    if frame.empty:
        raise ResourceError("사용 가능한 완전한 공정 기록이 없습니다.")

    exact_hour = (
        frame["timestamp"].dt.minute.eq(0)
        & frame["timestamp"].dt.second.eq(0)
    )
    zero_counts = {
        column: int(frame[column].eq(0).sum())
        for column in ["turbidity_ntu", "cod_mg_l", "total_n_mg_l", "total_p_mg_l"]
    }
    profile = DataProfile(
        source_rows=source_rows,
        usable_rows=len(frame),
        invalid_timestamp_rows=invalid_timestamp_rows,
        incomplete_rows=incomplete_rows,
        duplicate_rows_removed=duplicate_rows_removed,
        exact_hour_rows=int(exact_hour.sum()),
        zero_counts=zero_counts,
    )
    return frame, profile


def _load_model(spec: dict[str, Any], manifest_path: Path) -> Any:
    filename = spec.get("file")
    if not isinstance(filename, str) or not filename:
        raise ResourceError("모델 매니페스트의 파일명이 올바르지 않습니다.")

    path = manifest_path.parent / filename
    if not path.is_file():
        raise ResourceError(f"모델 파일이 없습니다: {filename}")

    expected_hash = spec.get("sha256")
    actual_hash = _sha256(path)
    if expected_hash != actual_hash:
        raise ResourceError(f"모델 무결성 검증에 실패했습니다: {filename}")

    try:
        model = joblib.load(path)
    except Exception as exc:  # pragma: no cover - joblib supplies the detail
        raise ResourceError(f"{filename}을 불러오지 못했습니다: {exc}") from exc

    if not callable(getattr(model, "predict", None)):
        raise ResourceError(f"{filename}은 예측 모델이 아닙니다.")

    expected_features = spec.get("features", [])
    feature_count = int(getattr(model, "n_features_in_", len(expected_features)))
    if feature_count != len(expected_features):
        raise ResourceError(
            f"{filename} 입력 열 수가 맞지 않습니다: {feature_count} != {len(expected_features)}"
        )
    actual_features = list(getattr(model, "feature_names_in_", expected_features))
    if actual_features != expected_features:
        raise ResourceError(f"{filename} 입력 열 순서가 매니페스트와 다릅니다.")
    return model


def load_models(manifest_path: Path = MANIFEST_PATH) -> ModelBundle:
    if not manifest_path.is_file():
        raise ResourceError("모델 매니페스트가 없습니다.")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResourceError(f"모델 매니페스트를 읽지 못했습니다: {exc}") from exc

    models = manifest.get("models", {})
    if not {"pressure", "sec"}.issubset(models):
        raise ResourceError("모델 매니페스트에 pressure/sec 정의가 필요합니다.")

    return ModelBundle(
        pressure=_load_model(models["pressure"], manifest_path),
        sec=_load_model(models["sec"], manifest_path),
        manifest=manifest,
    )
