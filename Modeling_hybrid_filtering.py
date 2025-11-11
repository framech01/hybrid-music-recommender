import os
import numpy as np
import pandas as pd

# ---------------- Configuration ----------------
ALS_PATH = "./results/als_top10_per_user.csv"
CONTENT_PATH = "./results/top10_per_track_final.csv"
OUT_PATH = "./results/hybrid_top10_per_user.csv"

# weight configuration (adjustable)
W_ALS = 0.7
W_CONTENT = 0.3
TOP_K_FINAL = 10
# ------------------------------------------------


def main():
    print("[STEP 1] Loading input files")

    if not os.path.exists(ALS_PATH):
        raise FileNotFoundError(f"ALS result file not found: {ALS_PATH}")
    if not os.path.exists(CONTENT_PATH):
        raise FileNotFoundError(f"Content similarity file not found: {CONTENT_PATH}")

    df_als = pd.read_csv(ALS_PATH)
    df_content = pd.read_csv(CONTENT_PATH)

    print(f"[ALS] rows={len(df_als):,}, [Content] rows={len(df_content):,}")

    df_content.rename(columns={"track_A": "track_ref", "track_B": "track_id"}, inplace=True)

    df_hybrid = pd.merge(
        df_als,
        df_content[["track_id", "similarity"]],
        on="track_id",
        how="left"
    )

    df_hybrid["similarity"] = df_hybrid["similarity"].fillna(0)
    df_hybrid["score"] = df_hybrid["score"].fillna(0)

    for col in ["score", "similarity"]:
        maxv = df_hybrid[col].max()
        if maxv > 0:
            df_hybrid[col] /= maxv

    df_hybrid["final_score"] = (
        W_ALS * df_hybrid["score"] +
        W_CONTENT * df_hybrid["similarity"]
    )

    df_hybrid = (
        df_hybrid
        .sort_values(["user_id", "final_score"], ascending=[True, False])
        .groupby("user_id")
        .head(TOP_K_FINAL)
        .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_hybrid.to_csv(OUT_PATH, index=False)
    print(f"[SAVE] {OUT_PATH} | rows={len(df_hybrid):,}")

    print("\n[PREVIEW]")
    print(df_hybrid.head(15).to_string(index=False))

    print("\n[STATS]")
    print(df_hybrid.groupby("user_id")["final_score"].max().describe().round(3))


if __name__ == "__main__":
    main()
