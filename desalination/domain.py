from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class EnergyStatus:
    level: str
    label: str
    message: str
    image_name: str


def classify_energy(value: float) -> EnergyStatus:
    """예측 에너지 사용량을 운영 단계로 변환한다."""
    value = float(value)
    if not isfinite(value):
        raise ValueError("전력량은 유한한 숫자여야 합니다.")
    if value < 3.5:
        return EnergyStatus(
            "normal",
            "정상",
            "정상 범위입니다. 현재 운전 조건을 유지하세요.",
            "대시보드 구성도_정상_w.png",
        )
    if value <= 3.7:
        return EnergyStatus(
            "warning",
            "주의",
            "주의 단계입니다. Partial two-pass 전환을 검토하세요.",
            "대시보드 구성도_주의_w.png",
        )
    return EnergyStatus(
        "danger",
        "경고",
        "경고 단계입니다. Split partial two-pass 전환과 원수 상태 점검이 필요합니다.",
        "대시보드 구성도_이상_w.png",
    )


def quality_achievement(inflow: float, reduction: float, standard: float) -> float:
    """필요 제거량 대비 실제 제거량을 0~1 범위의 달성률로 계산한다.

    원수가 이미 기준을 만족하면 추가 제거 없이 100%로 본다.
    """
    inflow = float(inflow)
    reduction = float(reduction)
    standard = float(standard)
    if not all(isfinite(value) for value in (inflow, reduction, standard)):
        raise ValueError("수질 값은 유한한 숫자여야 합니다.")

    required_reduction = inflow - standard
    if required_reduction <= 0:
        return 1.0
    return min(max(reduction / required_reduction, 0.0), 1.0)


def production_progress(
    hour: int,
    minute: int,
    hourly_rate_m3: float = 83.33 * 60,
    daily_capacity_m3: float = 120_000,
) -> float:
    """자정부터 현재 시각까지의 생산 진척률을 0~100으로 계산한다."""
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("올바른 시각을 입력하세요.")
    if hourly_rate_m3 < 0 or daily_capacity_m3 <= 0:
        raise ValueError("생산량과 설비 용량은 양수여야 합니다.")

    elapsed_hours = hour + minute / 60
    return min(max(elapsed_hours * hourly_rate_m3 / daily_capacity_m3 * 100, 0.0), 100.0)

