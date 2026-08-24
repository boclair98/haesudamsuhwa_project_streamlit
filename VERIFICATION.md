# Verification report

검증일: 2026-08-25 (Asia/Seoul)

## Automated checks

- `python -m unittest discover -s tests -v`: 8/8 passed
- `python -m py_compile`: application and all domain modules passed
- `GET /_stcore/health`: HTTP 200, body `ok`
- exact runtime pins exercised locally:
  - Python 3.11
  - Streamlit 1.62.0
  - Pandas 2.2.3
  - NumPy 1.26.4
  - scikit-learn 1.2.2
  - Plotly 6.9.0

## Browser checks

- initial page renders meaningful content with no Streamlit exception
- browser console contains no error/warning in a stable fresh session
- all five views render:
  - 운전 스냅샷
  - 기간 성과
  - 예측 실험실
  - 운영 인사이트
  - 데이터·모델 카드
- 기간 성과에서 임의의 두 달을 선택해 차이를 비교할 수 있음
- 운영 인사이트에서 오차 95백분위 검토 후보, 시간순 오차 차트, CSV 다운로드를 확인
- historical date/time selectors use real recorded timestamps
- monthly selector and sparse-month rendering work
- scenario slider change reruns the chained prediction without error
- 390 × 844 mobile viewport has no horizontal overflow
- 1280 × 720 desktop viewport has no horizontal overflow
- Streamlit development toolbar is hidden in the product UI

## Data/model checks

- 5,687 source rows become 5,686 unique usable timestamps
- one duplicate timestamp is removed deterministically by keeping the last row
- pressure and SEC model files are verified against SHA-256 hashes before loading
- 24-row real-resource integration sample produces finite chained predictions
- displayed model diagnostics use predicted pressure in the SEC chain

## Environment limitation

Docker is not installed in the local Codex workspace, so a local `docker build`
could not be executed. The container contract is verified through the same
Streamlit command, the health endpoint, and the coders.kr deployment build.
