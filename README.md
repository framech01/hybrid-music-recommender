# Hybrid Music Recommendation Pipeline

음원·사용자 행동 데이터를 전처리하고 ALS 협업 필터링과 콘텐츠 유사도를 결합해 Top-N 추천을 생성하는 Python 파이프라인입니다.

## 주요 기능

- 시간 순서 기반 train / latest DB / test 분리
- 학습 데이터에만 맞춘 결측치 대체, 윈저라이징, robust scaling
- implicit ALS 기반 개인화 추천
- 오디오 특성 기반 cosine nearest-neighbor 추천
- ALS와 콘텐츠 점수를 결합한 하이브리드 랭킹
- 대용량 EDA 및 시각화

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

## 성능 개선

- 콘텐츠 모델은 전체 `n x n` 유사도 행렬을 만들지 않고 nearest-neighbor 검색만 수행합니다.
- ALS 추천은 사용자 1명씩 호출하지 않고 1,024명 단위로 벡터화합니다.
- 전처리 split 저장 후 CSV를 다시 읽던 불필요한 I/O를 제거했습니다.

## 출력

- `ml_datasets/`: 전처리 데이터와 암시적 피드백
- `results/als_top10_per_user.csv`: ALS 추천
- `results/top10_per_track_final.csv`: 트랙별 유사곡
- `results/hybrid_top10_per_user.csv`: 최종 하이브리드 추천

생성 데이터와 결과 파일은 용량 및 개인정보 보호를 위해 Git에서 제외됩니다.
