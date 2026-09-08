import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm

# ---------------- Configuration ----------------
DATA_PATH = "./ml_datasets/user_track_interactions.csv"   # user_id, track_id, play_count
OUT_DIR = "./results"
TOP_K = 10
FACTOR_DIM = 64         # Latent dimension
REGULARIZATION = 0.1    # Regularization term (lambda)
ALPHA = 40              # Confidence scaling factor (alpha)
N_ITER = 15
RANDOM_SEED = 42
# ------------------------------------------------


def main():
    # Load implicit feedback data
    print(f"[LOAD] Reading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] shape={df.shape}, columns={list(df.columns)}")

    # Expected columns check
    required_cols = {"user_id", "track_id", "play_count"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain {required_cols}")

    # Encode IDs to integer indices
    user_codes, user_uniques = pd.factorize(df["user_id"])
    item_codes, item_uniques = pd.factorize(df["track_id"])
    n_users, n_items = len(user_uniques), len(item_uniques)
    print(f"[INFO] unique users={n_users:,}, unique items={n_items:,}")

    # Build sparse user-item matrix (implicit feedback)
    data = df["play_count"].astype(float)
    user_item_csr = csr_matrix((data, (user_codes, item_codes)), shape=(n_users, n_items))

    # Apply confidence weighting for implicit feedback: C_ui = 1 + alpha * R_ui
    user_item_conf = user_item_csr.copy()
    user_item_conf *= ALPHA
    user_item_conf.data += 1.0  # add base confidence
    user_item_conf = user_item_conf.tocsr()  # ensure CSR format
    print(f"[BUILD] CSR matrix shape={user_item_conf.shape}, nnz={user_item_conf.nnz:,}")

    # Train ALS model
    model = AlternatingLeastSquares(
        factors=FACTOR_DIM,
        regularization=REGULARIZATION,
        iterations=N_ITER,
        random_state=RANDOM_SEED
    )

    print("[TRAIN] Fitting ALS model ...")
    model.fit(user_item_conf)
    print("[TRAIN] Done ")

    # Sanity check
    assert model.user_factors.shape[0] == n_users, "User factor dimension mismatch"
    assert model.item_factors.shape[0] == n_items, "Item factor dimension mismatch"

    # Generate Top-N recommendations per user
    records = []
    batch_size = 1_024
    for start in tqdm(range(0, n_users, batch_size), desc=f"Top-{TOP_K} batches"):
        stop = min(start + batch_size, n_users)
        user_indices = np.arange(start, stop)
        rec_items, rec_scores = model.recommend(
            user_indices, user_item_conf[start:stop], N=TOP_K, filter_already_liked_items=True
        )
        for offset, u in enumerate(user_indices):
            user_id = user_uniques[u]
            for rank, (item_idx, score) in enumerate(
                zip(rec_items[offset], rec_scores[offset]), start=1
            ):
                records.append({
                    "user_id": user_id,
                    "track_id": item_uniques[item_idx],
                    "rank_in_user": rank,
                    "score": float(score)
                })

    # Save results
    df_topk = pd.DataFrame(records)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "als_top10_per_user.csv")
    df_topk.to_csv(out_path, index=False)
    print(f"[SAVE] {out_path} | rows={len(df_topk):,}")


if __name__ == "__main__":
    main()
