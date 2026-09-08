# Hybrid Music Recommendation Pipeline

음원·사용자 행동 데이터를 전처리하고 ALS 협업 필터링과 콘텐츠 유사도를 결합해 Top-N 추천을 생성하는 Python 파이프라인입니다.

## 주요 기능

- 시간 순서 기반 train / latest DB / test 분리
- 학습 데이터에만 맞춘 결측치 대체, 윈저라이징, robust scaling
- implicit ALS 기반 개인화 추천
- 오디오 특성 기반 cosine nearest-neighbor 추천
- ALS와 콘텐츠 점수를 결합한 하이브리드 랭킹
- 대용량 EDA 및 시각화

## 사용 기술

### Language & Data Processing

- **Python 3.10+**: 전처리, 모델 학습, 추천 생성, EDA 파이프라인을 구성합니다. 각 단계는 독립 실행 가능한 스크립트로 분리했습니다.
- **Pandas**: CSV 입출력, 사용자·트랙별 재생 횟수 집계, 시간 기준 데이터 분할, 결측치 처리와 추천 결과 병합에 사용합니다.
- **NumPy**: 오디오 특성 행렬 변환, L2 정규화, percentile 기반 윈저라이징과 수치 연산을 벡터화합니다.
- **SciPy Sparse Matrix**: 사용자-아이템 상호작용을 `csr_matrix`로 저장해 관측되지 않은 대부분의 조합에 메모리를 쓰지 않습니다.

### Recommendation Models

- **Implicit ALS (`implicit`)**: 명시적 평점 대신 재생 횟수를 암시적 피드백으로 사용합니다. `1 + alpha × play_count`로 신뢰도를 부여하고 사용자·트랙을 64차원 잠재요인으로 학습합니다.
- **Content-Based Filtering (`scikit-learn`)**: danceability, energy, valence 등 9개 오디오 특성을 정규화한 뒤 cosine distance 기반 `NearestNeighbors`로 유사곡을 탐색합니다.
- **Hybrid Ranking**: ALS 개인화 점수와 콘텐츠 유사도를 각각 정규화하고 기본 가중치 `0.7 : 0.3`으로 결합해 사용자별 최종 Top-10을 만듭니다.

### Preprocessing & Validation

- **시간 기반 분할**: 최신 연도의 1~10월을 학습, 11월을 최신 DB, 12월을 테스트로 분리해 미래 정보가 학습에 섞이는 것을 방지합니다.
- **Robust Scaling**: 중앙값과 IQR을 학습 데이터에서만 계산하여 이상치 영향을 줄이고 데이터 누수를 방지합니다.
- **Winsorization**: 연속형 변수의 1·99 percentile 밖 값을 제한해 극단값의 영향을 완화합니다.
- **Reproducibility**: 모든 샘플링과 모델 학습에 고정 random seed를 사용합니다.

### Analysis & Visualization

- **Matplotlib**: 특성별 histogram, boxplot, 월별·시간대별 이벤트 분포를 생성합니다.
- **tqdm**: 유사곡 및 추천 배치 생성 진행률을 표시합니다.

### 성능 설계

- 콘텐츠 유사도는 `n × n` 행렬 전체를 생성하지 않아 메모리 사용량을 크게 줄입니다.
- ALS 추천을 1,024명 단위로 호출해 Python 반복 호출 비용을 줄입니다.
- 희소 행렬을 사용해 사용자 수와 트랙 수가 증가해도 관측된 상호작용만 저장합니다.
- 전처리 결과를 저장한 직후 메모리상의 DataFrame을 재사용하여 불필요한 CSV 재로딩을 제거합니다.

## 빠른 시작

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python preprocessing.py
python Modeling_model_based.py
python Modeling_content_based.py
python Modeling_hybrid_filtering.py
```

루트에 `context_content_features.csv`를 두거나 각 파일 상단의 경로 설정을 데이터 위치에 맞게 변경하세요. 입력에는 최소한 `user_id`, `track_id`, 타임스탬프(`created_at`, `timestamp`, `time`, `ts` 중 하나)가 필요합니다. 콘텐츠 모델은 `danceability`, `energy`, `valence`, `acousticness`, `instrumentalness`, `tempo`, `loudness`, `liveness`, `speechiness`도 사용합니다.

## 출력

- `ml_datasets/`: 전처리 데이터와 암시적 피드백
- `results/als_top10_per_user.csv`: ALS 추천
- `results/top10_per_track_final.csv`: 트랙별 유사곡
- `results/hybrid_top10_per_user.csv`: 최종 하이브리드 추천

생성 데이터와 결과 파일은 용량 및 개인정보 보호를 위해 Git에서 제외됩니다.
