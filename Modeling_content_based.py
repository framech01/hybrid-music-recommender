import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# ==============================================
DATA_PATH = "./ml_datasets/train_2014_1~10 after_preprocessing.csv"
OUT_PATH  = "./results/top10_per_track_final.csv"
TOP_K = 10

ID_COL = "track_id"
FEATURE_COLS = [
    "danceability","energy","valence","acousticness","instrumentalness",
    "tempo","loudness","liveness","speechiness"
]
# ==============================================


def normalize_rows(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return X / norms


def main():
    # Load
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURE_COLS)
    ids = df[ID_COL].astype(str).tolist()
    X = normalize_rows(df[FEATURE_COLS].to_numpy(float))
    n = len(df)
    print(f"[LOAD] {n:,} tracks loaded")

    # Brute-force cosine neighbors are evaluated in bounded batches by sklearn.
    # This avoids materializing the O(n^2) similarity matrix.
    print("[SIM] fitting nearest-neighbor index ...")
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
    model.fit(X)
    distances, indices = model.kneighbors(X, n_neighbors=min(TOP_K + 1, n))

    records = []
    for i, (row_distances, row_indices) in enumerate(
        tqdm(zip(distances, indices), total=n, desc=f"Top-{TOP_K} per track")
    ):
        rank = 0
        for distance, j in zip(row_distances, row_indices):
            if ids[i] == ids[j]:
                continue
            score = 1.0 - distance
            if not np.isfinite(score) or score <= 0:
                continue
            rank += 1
            records.append({
                "track_A": ids[i],
                "track_B": ids[j],
                "rank_in_A": rank,
                "similarity": float(score)
            })
            if rank >= TOP_K:
                break

    # Save
    df_topk = pd.DataFrame(records)
    df_topk = df_topk.sort_values(["track_A", "rank_in_A"])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_topk.to_csv(OUT_PATH, index=False)
    print(f"[SAVE] {OUT_PATH} | rows={len(df_topk):,}")

    # Validation
    n_self = (df_topk["track_A"] == df_topk["track_B"]).sum()
    print(f"[CHECK] self-pairs found: {n_self}")
    print("\n[PREVIEW]")
    print(df_topk.head(15).to_string(index=False))

    # 6 Rank check summary
    max_rank = df_topk.groupby("track_A")["rank_in_A"].max()
    print(f"[STATS] rank_in_A min={max_rank.min()} median={max_rank.median()} max={max_rank.max()}")

if __name__ == "__main__":
    main()
