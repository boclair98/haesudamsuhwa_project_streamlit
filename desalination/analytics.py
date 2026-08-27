from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


PRESSURE_FEATURES = ["수온", "수소이온농도"]
SEC_FEATURES = ["총인", "화학적산소요구량", "총질소", "탁도", "1차 인입압력"]


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float
    samples: int


def _pressure_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "수온": frame["temperature_c"].to_numpy(),
            "수소이온농도": frame["ph"].to_numpy(),
        },
        index=frame.index,
    )


def _sec_frame(frame: pd.DataFrame, pressure) -> pd.DataFrame:
    pressure_values = np.asarray(pressure, dtype=float)
    return pd.DataFrame(
        {
            "총인": frame["total_p_mg_l"].to_numpy(),
            "화학적산소요구량": frame["cod_mg_l"].to_numpy(),
            "총질소": frame["total_n_mg_l"].to_numpy(),
            "탁도": frame["turbidity_ntu"].to_numpy(),
            "1차 인입압력": pressure_values,
        },
        index=frame.index,
    )


def predict_pressure(model, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(_pressure_frame(frame)), dtype=float)


def predict_sec(model, frame: pd.DataFrame, pressure) -> np.ndarray:
    return np.asarray(model.predict(_sec_frame(frame, pressure)), dtype=float)


def enrich_history(history: pd.DataFrame, pressure_model, sec_model) -> pd.DataFrame:
    enriched = history.copy()
    enriched["model_pressure_bar"] = predict_pressure(pressure_model, enriched)
    enriched["model_sec_kwh_m3"] = predict_sec(
        sec_model, enriched, enriched["model_pressure_bar"]
    )
    enriched["pressure_error_bar"] = (
        enriched["model_pressure_bar"] - enriched["pressure_stage1_bar"]
    )
    enriched["sec_error_kwh_m3"] = (
        enriched["model_sec_kwh_m3"] - enriched["sec_kwh_m3"]
    )
    return enriched


def regression_metrics(actual, predicted) -> RegressionMetrics:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    finite = np.isfinite(actual_values) & np.isfinite(predicted_values)
    actual_values = actual_values[finite]
    predicted_values = predicted_values[finite]
    if not len(actual_values):
        raise ValueError("회귀 지표를 계산할 유효 표본이 없습니다.")

    errors = actual_values - predicted_values
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    denominator = float(np.sum(np.square(actual_values - np.mean(actual_values))))
    r2 = 1 - float(np.sum(np.square(errors))) / denominator if denominator else 0.0
    return RegressionMetrics(mae=mae, rmse=rmse, r2=r2, samples=len(actual_values))


def diagnostics(history: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    pressure = regression_metrics(
        history["pressure_stage1_bar"], history["model_pressure_bar"]
    )
    sec = regression_metrics(history["sec_kwh_m3"], history["model_sec_kwh_m3"])
    return {"pressure": asdict(pressure), "sec_chain": asdict(sec)}


def model_explainability(pressure_model, sec_model) -> dict[str, pd.DataFrame]:
    """Expose model signals without presenting them as causal effects.

    The pressure model is a linear regressor, so its signed coefficients are
    useful as directional signals. The SEC model is a tree ensemble, so its
    feature_importances_ values are relative split-based importances. Both
    outputs are deliberately returned as small tables for transparent display
    and export; missing estimator metadata results in an empty table.
    """

    pressure_names = list(
        getattr(pressure_model, "feature_names_in_", PRESSURE_FEATURES)
    )
    pressure_values = np.asarray(
        getattr(pressure_model, "coef_", []), dtype=float
    ).reshape(-1)
    pressure_size = min(len(pressure_names), len(pressure_values))
    pressure = pd.DataFrame(
        {
            "feature": pressure_names[:pressure_size],
            "value": pressure_values[:pressure_size],
        }
    )

    sec_names = list(getattr(sec_model, "feature_names_in_", SEC_FEATURES))
    sec_values = np.asarray(
        getattr(sec_model, "feature_importances_", []), dtype=float
    ).reshape(-1)
    sec_size = min(len(sec_names), len(sec_values))
    sec = pd.DataFrame(
        {
            "feature": sec_names[:sec_size],
            "value": sec_values[:sec_size],
        }
    )
    return {"pressure": pressure, "sec": sec}


def anomaly_watchlist(
    history: pd.DataFrame,
    quantile: float = 0.95,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Return statistically unusual model/record gaps for human review.

    This is a distribution-based review queue, not a safety or quality alarm.
    The thresholds are calculated from the supplied history so the result stays
    tied to the visible data rather than an undocumented engineering limit.
    """
    if not 0.5 <= float(quantile) < 1.0:
        raise ValueError("검토 분위수는 0.5 이상 1.0 미만이어야 합니다.")

    frame = history.copy()
    required = {
        "timestamp",
        "pressure_error_bar",
        "sec_error_kwh_m3",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"이상 구간 계산에 필요한 열이 없습니다: {', '.join(missing)}")

    frame["abs_pressure_error_bar"] = frame["pressure_error_bar"].abs()
    frame["abs_sec_error_kwh_m3"] = frame["sec_error_kwh_m3"].abs()
    pressure_cutoff = float(frame["abs_pressure_error_bar"].quantile(quantile))
    sec_cutoff = float(frame["abs_sec_error_kwh_m3"].quantile(quantile))
    frame["watch_flag"] = (
        frame["abs_pressure_error_bar"].ge(pressure_cutoff)
        | frame["abs_sec_error_kwh_m3"].ge(sec_cutoff)
    )

    def reason(row: pd.Series) -> str:
        reasons = []
        if row["abs_pressure_error_bar"] >= pressure_cutoff:
            reasons.append("압력 차이")
        if row["abs_sec_error_kwh_m3"] >= sec_cutoff:
            reasons.append("SEC 차이")
        return " · ".join(reasons)

    frame["watch_reason"] = frame.apply(reason, axis=1)
    watchlist = (
        frame.loc[frame["watch_flag"]]
        .sort_values(
            ["abs_sec_error_kwh_m3", "abs_pressure_error_bar", "timestamp"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    thresholds: dict[str, float | int] = {
        "quantile": float(quantile),
        "pressure_cutoff": pressure_cutoff,
        "sec_cutoff": sec_cutoff,
        "watch_count": int(len(watchlist)),
        "watch_share_pct": float(len(watchlist) / len(frame) * 100) if len(frame) else 0.0,
    }
    return watchlist, thresholds


def percentile_rank(values, value: float) -> float:
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if not len(sample):
        raise ValueError("백분위 위치를 계산할 표본이 없습니다.")
    return float(np.mean(sample <= float(value)) * 100)


def monthly_summary(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.copy()
    frame["month"] = frame["timestamp"].dt.to_period("M")
    frame["is_exact_hour"] = (
        frame["timestamp"].dt.minute.eq(0)
        & frame["timestamp"].dt.second.eq(0)
    )
    monthly = (
        frame.groupby("month", observed=True)
        .agg(
            samples=("timestamp", "size"),
            exact_hour_samples=("is_exact_hour", "sum"),
            pressure_stage1_bar=("pressure_stage1_bar", "mean"),
            pressure_stage2_bar=("pressure_stage2_bar", "mean"),
            model_pressure_bar=("model_pressure_bar", "mean"),
            sec_kwh_m3=("sec_kwh_m3", "mean"),
            model_sec_kwh_m3=("model_sec_kwh_m3", "mean"),
            tds_stage1_mg_l=("tds_stage1_mg_l", "mean"),
            tds_stage2_mg_l=("tds_stage2_mg_l", "mean"),
            turbidity_ntu=("turbidity_ntu", "mean"),
            cod_mg_l=("cod_mg_l", "mean"),
            total_n_mg_l=("total_n_mg_l", "mean"),
            total_p_mg_l=("total_p_mg_l", "mean"),
        )
        .reset_index()
    )
    monthly["expected_hours"] = monthly["month"].map(
        lambda period: period.days_in_month * 24
    )
    monthly["exact_hour_density_pct"] = (
        monthly["exact_hour_samples"] / monthly["expected_hours"] * 100
    ).clip(upper=100)
    monthly["month_label"] = monthly["month"].astype(str)
    monthly["month_start"] = monthly["month"].dt.to_timestamp()
    return monthly


def support_ranges(history: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "temperature_c": ("수온", "°C"),
        "ph": ("pH", ""),
        "turbidity_ntu": ("탁도", "NTU"),
        "cod_mg_l": ("COD", "mg/L"),
        "total_n_mg_l": ("총질소", "mg/L"),
        "total_p_mg_l": ("총인", "mg/L"),
        "pressure_stage1_bar": ("1단 인입압력", "bar"),
    }
    rows = []
    for column, (label, unit) in labels.items():
        values = history[column].astype(float)
        rows.append(
            {
                "column": column,
                "label": label,
                "unit": unit,
                "min": float(values.min()),
                "p05": float(values.quantile(0.05)),
                "median": float(values.median()),
                "p95": float(values.quantile(0.95)),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def predict_scenario(
    pressure_model,
    sec_model,
    *,
    temperature_c: float,
    ph: float,
    turbidity_ntu: float,
    cod_mg_l: float,
    total_n_mg_l: float,
    total_p_mg_l: float,
) -> tuple[float, float]:
    frame = pd.DataFrame(
        [
            {
                "temperature_c": temperature_c,
                "ph": ph,
                "turbidity_ntu": turbidity_ntu,
                "cod_mg_l": cod_mg_l,
                "total_n_mg_l": total_n_mg_l,
                "total_p_mg_l": total_p_mg_l,
            }
        ]
    )
    pressure = float(predict_pressure(pressure_model, frame)[0])
    sec = float(predict_sec(sec_model, frame, [pressure])[0])
    return pressure, sec

