from math import sqrt

import pandas as pd


PRESSURE_FEATURES = ["수온", "수소이온농도"]
ENERGY_FEATURES = ["총인", "화학적산소요구량", "총질소", "탁도", "1차 인입압력"]


def _regression_metrics(actual: pd.Series, predicted) -> dict[str, float]:
    predicted_series = pd.Series(predicted, index=actual.index, dtype=float)
    errors = actual.astype(float) - predicted_series
    mae = float(errors.abs().mean())
    rmse = sqrt(float(errors.pow(2).mean()))
    denominator = float((actual - actual.mean()).pow(2).sum())
    r2 = 1 - float(errors.pow(2).sum()) / denominator if denominator else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def evaluate_models(pressure_model, energy_model, ro_data: pd.DataFrame) -> dict:
    """저장된 모델이 과거 공정 데이터에 얼마나 잘 맞는지 진단한다.

    이는 별도 검증 세트가 아닌 저장소의 과거 데이터 적합도이므로,
    배포 화면에서도 그 한계를 함께 표시해야 한다.
    """
    pressure_data = ro_data.dropna(subset=PRESSURE_FEATURES + ["1차 인입압력"])
    energy_data = ro_data.dropna(subset=ENERGY_FEATURES + ["전체 전력량"])
    if pressure_data.empty or energy_data.empty:
        raise ValueError("모델 진단에 사용할 완전한 공정 데이터가 없습니다.")

    pressure_prediction = pressure_model.predict(pressure_data[PRESSURE_FEATURES])
    energy_prediction = energy_model.predict(energy_data[ENERGY_FEATURES])
    return {
        "pressure": _regression_metrics(
            pressure_data["1차 인입압력"], pressure_prediction
        ),
        "energy": _regression_metrics(energy_data["전체 전력량"], energy_prediction),
        "pressure_samples": len(pressure_data),
        "energy_samples": len(energy_data),
    }

