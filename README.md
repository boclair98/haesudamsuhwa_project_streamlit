# RO Lens

2021년 천수만 원수 관측자료와 RO 공정 기록을 연결해
1단 인입압력과 비에너지소비량(SEC)을 탐색하는 Streamlit 데이터 분석 프로젝트입니다.

[공개 데모 바로가기 →](https://haesudamsuhwa.coders.kr/)

> 이 프로젝트는 실제 플랜트 제어 시스템이 아닌 연구·시연용 과거 데이터 데모입니다.
> 센서, PLC, SCADA, 제어 밸브와 연결되지 않으며 안전·식수 적합성·규제 준수·운전값을
> 판정하거나 권고하지 않습니다.

## 📌 프로젝트 소개

해수담수화 공정에서는 원수 수질과 운전 조건에 따라 필요한 압력과 에너지 사용량이 달라집니다.
RO Lens는 흩어진 원수·RO 기록을 하나의 분석 흐름으로 묶어 다음 질문에 답할 수 있도록 만들었습니다.

- 특정 기록 시점에서 모델 추정값과 실제 기록값은 얼마나 다른가?
- 월별 압력·SEC·TDS 흐름은 어떻게 변하는가?
- 기록 범위 안에서 원수 조건을 바꾸면 추정 결과가 어떻게 달라지는가?
- 모델과 기록의 차이가 큰 시점을 어떤 순서로 검토할 것인가?

## 🎯 프로젝트 목표

- 기록에 없는 실시간 상태를 만들어내지 않고 실제 시점만 탐색하기
- 모델 추정값과 관측 기록을 화면과 데이터 구조에서 분리하기
- 예측 결과의 적용 범위와 한계를 함께 공개하기
- 대시보드·분석 로직·리소스 로딩을 분리해 재실행 가능한 구조 만들기
- 테스트와 배포 설정을 저장소에 포함해 다른 환경에서도 같은 앱 실행하기

## 🚀 주요 기능

기능 | 설명
--- | ---
운전 스냅샷 | 실제 존재하는 날짜·시각을 선택해 압력, SEC, TDS의 기록값과 모델 추정값 비교
기간 성과 | 월별 압력·SEC·TDS·표본 수·정시 관측 밀도 확인 및 임의의 두 달 비교
예측 실험실 | 수온, pH, 탁도, COD, 총질소, 총인을 기록 범위 안에서 조절
연쇄 예측 | 수온·pH로 압력을 먼저 추정하고 그 결과를 SEC 모델 입력으로 연결
운영 인사이트 | 압력·SEC 오차의 95백분위 이상 시점을 통계적 검토 후보로 정렬
CSV 내보내기 | 월간 요약, 정제 기록, 검토 후보를 후속 분석용 CSV로 다운로드
데이터·모델 카드 | 데이터 출처, 결측·0값, 모델 입력 범위, 성능과 누락 변수를 공개
무결성 검증 | `manifest.json`의 SHA-256 해시가 일치하는 모델만 로드

## 🛠 기술 스택

- **Language**: Python 3.11
- **Dashboard**: Streamlit 1.62
- **Data**: Pandas, NumPy
- **Machine Learning**: scikit-learn, Joblib
- **Visualization**: Plotly
- **Testing**: `unittest`, `py_compile`
- **Deployment**: Docker, coders.kr standalone

## 🧩 도메인 구조

```text
원수·RO 기록
    │
    ▼
resources.py
  ├─ CSV 정제
  ├─ 필수 열·결측 검증
  └─ 모델 SHA-256 검증
    │
    ▼
analytics.py
  ├─ 압력 예측
  ├─ chained SEC 예측
  ├─ 월간 집계
  └─ 오차·검토 후보 계산
    │
    ▼
app.py + charts.py
  ├─ 운전 스냅샷
  ├─ 기간 성과
  ├─ 예측 실험실
  ├─ 운영 인사이트
  └─ 데이터·모델 카드
```

### 모델 입력 구조

모델 | 입력 | 출력
--- | --- | ---
압력 선형회귀 | 수온, pH | 1단 인입압력
SEC 랜덤포레스트 | 총인, COD, 총질소, 탁도, **추정된** 1단 인입압력 | SEC(kWh/m³)

SEC 모델은 실제 압력 기록이 아니라 압력 모델의 추정값을 입력으로 사용합니다.
따라서 화면의 모델 비교 지표는 압력 추정 오차가 SEC 추정에도 이어지는 chained prediction 기준입니다.

## 📋 문서

문서 | 설명
--- | ---
[`VERIFICATION.md`](VERIFICATION.md) | 자동 테스트, 브라우저 검증, 헬스 체크와 환경 한계
[`coders.yaml`](coders.yaml) | coders.kr standalone 배포 설정
[`Dockerfile`](Dockerfile) | 동일한 실행 환경을 만드는 컨테이너 정의
[공개 데모](https://haesudamsuhwa.coders.kr/) | 배포된 현재 앱 확인

## 📂 프로젝트 구조

```text
.
├── app.py
├── desalination/
│   ├── analytics.py              # 예측·집계·오차 분석
│   ├── charts.py                 # Plotly 차트
│   └── resources.py              # 데이터·모델 로딩과 검증
├── data/
│   └── ro_history.csv            # 정제된 5,686개 고유 기록
├── models/
│   ├── pressure.joblib
│   ├── sec.joblib
│   └── manifest.json              # 모델 SHA-256 매니페스트
├── tests/
│   ├── test_analytics.py
│   ├── test_integration.py
│   └── test_resources.py
├── .streamlit/config.toml
├── Dockerfile
├── coders.yaml
├── requirements.txt
└── VERIFICATION.md
```

## 🏁 시작하기

```bash
git clone https://github.com/boclair98/haesudamsuhwa_project_streamlit.git
cd haesudamsuhwa_project_streamlit

python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python -m streamlit run app.py
```

브라우저에서 `http://localhost:8501`을 엽니다.
실행 상태는 `http://localhost:8501/_stcore/health`에서 확인할 수 있습니다.

## ✅ 테스트

```bash
python -m unittest discover -s tests -v
python -m py_compile app.py desalination/*.py
```

테스트 범위:

- 데이터 필수 열·결측·중복 정제
- 모델 매니페스트 검증
- 실제 리소스 24행 chained prediction 통합 테스트
- 월간 표본 수와 정시 관측 밀도 집계
- 회귀 지표와 백분위 계산
- 95백분위 기반 검토 후보 생성

현재 검증 기준은 `unittest 8/8 passed`이며, 상세 내용은 [`VERIFICATION.md`](VERIFICATION.md)에 기록되어 있습니다.

## 📊 데이터 기준

항목 | 기준
--- | ---
원본 행 | 5,687건
사용 가능 행 | 5,686건
기록 기간 | 2021-01-01 — 2021-12-16
원수 변수 | 수온, pH, 탁도, COD, 총질소, 총인
RO 변수 | 1·2단 압력, 1·2단 TDS, SEC
정제 규칙 | 날짜·수치 변환 → 필수값 결측 제거 → 동일 시각 중복은 마지막 기록 유지

원수 변수는 저장소 파일명 기준 해양환경공단 해양수질자동측정망 천수만(2021) 자료입니다.
RO 압력·TDS·SEC는 측정 설비, 보정 방법, 생산량 가중치가 충분히 문서화되지 않은
샘플/가공 공정 기록으로 취급합니다.

## ⚠️ 한계 및 운영 경계

- 저장소에 모델 학습 파이프라인과 독립 시간순 검증 결과가 없습니다.
- 화면의 MAE·RMSE·R²는 전체 기록에 다시 적용한 적합도이지 일반화 성능이 아닙니다.
- 염분·전기전도도, 원수/생산수 유량, 회수율, 막 상태, 차압, 약품 주입량이 모델에 없습니다.
- 생산량·회수율·식수 적합성·규제 준수·막 세정·운전 모드 전환을 계산하거나 판정하지 않습니다.
- 데이터의 0값이 실제 측정값인지 결측 대체값인지 원자료만으로 확정할 수 없습니다.
- 운영 인사이트의 검토 후보는 통계적 점검 순서이며 알람이나 운전 권고가 아닙니다.

## 🚢 배포

`coders.yaml`은 Streamlit WebSocket을 보존하는 `standalone` 모드와 공개 포트 8501을 사용합니다.
Docker 컨테이너는 플랫폼이 주입하는 `PORT`에 `0.0.0.0`으로 바인딩합니다.

현재 배포 주소: [haesudamsuhwa.coders.kr](https://haesudamsuhwa.coders.kr/)

## 📅 진행 상황

완료 | 내용
--- | ---
✅ | 레거시 대시보드를 기록 기반 RO Lens 구조로 정리
✅ | 압력·SEC chained prediction 및 모델-기록 차이 표시
✅ | 월간 비교, 운영 인사이트, CSV 내보내기 추가
✅ | 데이터·모델 무결성 검증과 8개 자동 테스트 구성
✅ | Docker/coders.kr 배포 및 공개 헬스 체크 검증

다음 개선 후보:

- 모델 학습 파이프라인과 시간순 holdout 검증 추가
- 실측 유량·회수율·염분 데이터 계약 확장
- 운영 로그와 검토 후보를 연결하는 별도 분석 저장소 구성
- 실제 센서 연동 전 인증·권한·감사 로그 설계

## 📄 License

별도 라이선스를 지정하지 않았습니다. 데이터와 모델 아티팩트의 재배포 범위는 원 출처와 저장소 정책을 먼저 확인해야 합니다.
