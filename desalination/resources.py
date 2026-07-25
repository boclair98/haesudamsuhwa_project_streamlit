from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILES = {
    "seawater": ("해양환경공단_해양수질자동측정망_천수만(2021).csv", "관측일자"),
    "ro": ("RO공정데이터_0621.csv", "일시"),
    "quality": ("수질만데이터.csv", "관측일자"),
    "ro_monthly": ("RO공정데이터.csv", "관측일자"),
    "seawater_quality": ("해수수질데이터.csv", "관측일자"),
}


class ResourceError(RuntimeError):
    """대시보드 리소스를 안전하게 준비할 수 없을 때 발생한다."""


def _read_csv(filename: str, date_column: str) -> pd.DataFrame:
    path = PROJECT_ROOT / filename
    if not path.is_file():
        raise ResourceError(f"데이터 파일을 찾을 수 없습니다: {filename}")

    try:
        frame = pd.read_csv(path, encoding="cp949")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        raise ResourceError(f"{filename}을 읽는 중 오류가 발생했습니다: {exc}") from exc

    if date_column not in frame.columns:
        raise ResourceError(f"{filename}에 필수 열 '{date_column}'이 없습니다.")

    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    invalid_dates = int(frame[date_column].isna().sum())
    if invalid_dates:
        raise ResourceError(f"{filename}에 해석할 수 없는 날짜가 {invalid_dates}건 있습니다.")
    return frame


def load_all_data() -> tuple[pd.DataFrame, ...]:
    return tuple(_read_csv(*spec) for spec in DATA_FILES.values())


def load_prediction_models():
    models = []
    for filename, expected_features in (("LR_pressure.pkl", 2), ("RF_elec.pkl", 5)):
        path = PROJECT_ROOT / filename
        if not path.is_file():
            raise ResourceError(f"모델 파일을 찾을 수 없습니다: {filename}")
        try:
            model = joblib.load(path)
        except Exception as exc:
            raise ResourceError(f"{filename}을 불러올 수 없습니다: {exc}") from exc
        if not callable(getattr(model, "predict", None)):
            raise ResourceError(f"{filename}은 예측 가능한 모델이 아닙니다.")
        feature_count = getattr(model, "n_features_in_", expected_features)
        if feature_count != expected_features:
            raise ResourceError(
                f"{filename} 입력 열 수가 예상과 다릅니다: {feature_count} != {expected_features}"
            )
        models.append(model)
    return tuple(models)

