import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List

# ----------------- Configuration -----------------
INPUT_CSV = "context_content_features.csv"  # Input raw dataset file
OUT_DIR = "./ml_datasets"                   # Output directory for processed files
SAMPLE_N = 50_000                           # 샘플링 크기 (ALS 및 전체 전처리에 사용)
TARGET_N = 10_000                           # Train set size
TARGET_M = 2_000                            # DB/Test set size
RANDOM_SEED = 42

TS_CANDIDATES = ["created_at", "timestamp", "time", "ts"]
ID_COLS = ["id", "user_id", "track_id", "artist_id"]
DROP_THRESHOLD = 0.50
WINSOR_PCT = (1.0, 99.0)
# -------------------------------------------------

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", None)

# ---------- I/O Functions ----------
def read_csv_robust(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"not found: {path}")
    try:
        return pd.read_csv(path, nrows=nrows, low_memory=False, on_bad_lines="skip", encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, nrows=nrows, low_memory=False, on_bad_lines="skip", encoding="latin-1")


def write_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[SAVE] {path} (rows={len(df):,}, cols={df.shape[1]})")


# ---------- Date/Split Functions ----------
def detect_timestamp(df: pd.DataFrame) -> str:
    for c in TS_CANDIDATES:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=False)
            return c
    raise KeyError(f"timestamp column not found. One of {TS_CANDIDATES} is required.")


def month_filter(df: pd.DataFrame, ts_col: str, year: int, months: Tuple[int, ...]) -> pd.DataFrame:
    s = df[ts_col]
    return df[(s.dt.year == year) & (s.dt.month.isin(months))]


def sample_n(df: pd.DataFrame, n: int, label: str) -> pd.DataFrame:
    if len(df) == 0:
        print(f"[WARN] {label}: no rows after filter.")
        return df
    if len(df) < n:
        print(f"[WARN] {label}: only {len(df):,} rows (< {n:,}). Using all rows.")
        return df.sample(n=len(df), random_state=RANDOM_SEED)
    return df.sample(n=n, random_state=RANDOM_SEED)


# ---------- Preprocessing Utilities ----------
def split_numeric(df: pd.DataFrame, exclude: List[str]) -> Tuple[List[str], List[str]]:
    nums_all = df.select_dtypes(include=[np.number]).columns.tolist()
    nums = [c for c in nums_all if c not in exclude]
    num01, num_other = [], []
    for c in nums:
        s = df[c].dropna()
        if s.empty:
            num_other.append(c)
            continue
        mn, mx = s.min(), s.max()
        if mn >= -1e-6 and mx <= 1.0 + 1e-6:
            num01.append(c)
        else:
            num_other.append(c)
    return num01, num_other


def most_frequent(series: pd.Series):
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else "UNK"


def robust_params_fit(train_df: pd.DataFrame, cols: List[str]) -> Dict[str, Tuple[float, float]]:
    params: Dict[str, Tuple[float, float]] = {}
    for c in cols:
        s = train_df[c].replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            params[c] = (0.0, 1.0)
            continue
        med = float(np.nanmedian(s))
        q1, q3 = np.nanpercentile(s, 25.0), np.nanpercentile(s, 75.0)
        iqr = q3 - q1
        scale = float(iqr) if iqr != 0 else 1.0
        params[c] = (med, scale)
    return params


def robust_apply_inplace(df: pd.DataFrame, cols: List[str], params: Dict[str, Tuple[float, float]]):
    for c in cols:
        med, scale = params.get(c, (0.0, 1.0))
        x = df[c].to_numpy(copy=False).astype(float)
        df[c] = (x - med) / scale


def clip01_inplace(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        df[c] = df[c].clip(lower=0.0, upper=1.0)


def winsor_apply_inplace(df: pd.DataFrame, cols: List[str], limits: Dict[str, Tuple[Optional[float], Optional[float]]]):
    for c in cols:
        lo, hi = limits.get(c, (None, None))
        if lo is None or hi is None or c not in df.columns:
            continue
        x = df[c].to_numpy(copy=False).astype(float)
        m = np.isfinite(x)
        x[m & (x < lo)] = lo
        x[m & (x > hi)] = hi
        df[c] = x


# ---------- Preprocessing Meta Fit/Apply ----------
def fit_meta_on_train(df_train: pd.DataFrame) -> Dict:
    meta: Dict = {
        "timestamp_col": None,
        "drop_cols": [],
        "numeric_01": [],
        "numeric_scaled": [],
        "categorical": [],
        "impute_numeric": {},
        "impute_categorical": {},
        "winsor_limits": {},
        "robust_params": {}
    }

    ts_col = detect_timestamp(df_train)
    meta["timestamp_col"] = ts_col

    miss_rate = df_train.isna().mean()
    drop_cols = miss_rate[miss_rate >= DROP_THRESHOLD].index.tolist()
    drop_cols = [c for c in drop_cols if c not in ID_COLS + TS_CANDIDATES]
    meta["drop_cols"] = drop_cols

    base = df_train.drop(columns=drop_cols, errors="ignore").copy()
    cat_cols = base.select_dtypes(exclude=[np.number, "datetime64[ns]"]).columns.tolist()
    exclude = ID_COLS + TS_CANDIDATES + cat_cols
    num01, num_other = split_numeric(base, exclude=exclude)
    meta["numeric_01"] = num01
    meta["numeric_scaled"] = num_other
    meta["categorical"] = cat_cols

    for c in num01 + num_other:
        med = base[c].median(skipna=True)
        meta["impute_numeric"][c] = None if np.isnan(med) else float(med)
    for c in cat_cols:
        meta["impute_categorical"][c] = most_frequent(base[c])

    limits: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for c in num_other:
        s = base[c].replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            limits[c] = (None, None)
        else:
            lo = float(np.nanpercentile(s, WINSOR_PCT[0]))
            hi = float(np.nanpercentile(s, WINSOR_PCT[1]))
            limits[c] = (lo, hi)
    meta["winsor_limits"] = limits
    meta["robust_params"] = robust_params_fit(base, num_other)

    print(f"[META] drop_cols: {meta['drop_cols']}")
    print(f"[META] numeric_01={len(num01)}, numeric_scaled={len(num_other)}, categorical={len(cat_cols)}")
    return meta


def apply_meta(df: pd.DataFrame, meta: Dict) -> pd.DataFrame:
    out = df.copy()
    ts_col = meta.get("timestamp_col")
    if ts_col and ts_col in out.columns:
        out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce", utc=False)
    out = out.drop(columns=meta.get("drop_cols", []), errors="ignore")
    num01 = [c for c in meta.get("numeric_01", []) if c in out.columns]
    num_other = [c for c in meta.get("numeric_scaled", []) if c in out.columns]
    cat_cols = [c for c in meta.get("categorical", []) if c in out.columns]
    for c, med in meta.get("impute_numeric", {}).items():
        if c in out.columns and med is not None:
            out[c] = out[c].fillna(med)
    for c, mf in meta.get("impute_categorical", {}).items():
        if c in out.columns:
            out[c] = out[c].fillna(mf)
    winsor_apply_inplace(out, num_other, meta.get("winsor_limits", {}))
    clip01_inplace(out, num01)
    robust_apply_inplace(out, num_other, meta.get("robust_params", {}))
    return out


# ---------- Main Execution ----------
def main():
    df = read_csv_robust(INPUT_CSV)
    print(f"[LOAD] Full dataset shape = {df.shape}")

    df = sample_n(df, SAMPLE_N, "Full Dataset Sampling")
    print(f"[SAMPLE] Randomly sampled {len(df):,} rows for further processing")

    if {"user_id", "track_id"}.issubset(df.columns):
        print("\n[ALS] Building user–track implicit feedback dataset")
        df_inter = (
            df.groupby(["user_id", "track_id"])
            .size()
            .reset_index(name="play_count")
        )
        df_inter["play_count"] = df_inter["play_count"].clip(upper=20)
        inter_path = os.path.join(OUT_DIR, "user_track_interactions.csv")
        write_csv(df_inter, inter_path)
        print(f"[ALS] user_track_interactions.csv generated ({len(df_inter):,} rows)")
    else:
        print("[ALS] Skipped: 'user_id' or 'track_id' missing in dataset.")

    df = df.drop_duplicates(subset=["track_id"]).reset_index(drop=True)
    print(f"[INIT] Removed duplicate track_id rows before split → {len(df):,} remaining.")

    ts_col = detect_timestamp(df)
    latest_year = int(df[ts_col].dt.year.max())
    print(f"[INFO] using latest year = {latest_year} (ts={ts_col})")

    os.makedirs(OUT_DIR, exist_ok=True)
    df_train_src = month_filter(df, ts_col, latest_year, tuple(range(1, 11)))
    df_db_src = month_filter(df, ts_col, latest_year, (11,))
    df_test_src = month_filter(df, ts_col, latest_year, (12,))

    df_train = sample_n(df_train_src, TARGET_N, f"TRAIN Jan-Oct {latest_year}")
    df_db = sample_n(df_db_src, TARGET_M, f"DB-LATEST Nov {latest_year}")
    df_test = sample_n(df_test_src, TARGET_M, f"TEST Dec {latest_year}")

    # Save split datasets
    train_path = os.path.join(OUT_DIR, f"train_{latest_year}_01~10_sample.csv")
    db_path = os.path.join(OUT_DIR, f"db_{latest_year}_11_sample.csv")
    test_path = os.path.join(OUT_DIR, f"test_{latest_year}_12_sample.csv")
    write_csv(df_train, train_path)
    write_csv(df_db, db_path)
    write_csv(df_test, test_path)

    df_train = read_csv_robust(train_path)
    df_db = read_csv_robust(db_path)
    df_test = read_csv_robust(test_path)
    print(f"[RELOAD] train={df_train.shape}, db_latest={df_db.shape}, test={df_test.shape}")

    meta = fit_meta_on_train(df_train)
    train_clean = apply_meta(df_train, meta)
    db_clean = apply_meta(df_db, meta)
    test_clean = apply_meta(df_test, meta)

    # Save final cleaned datasets
    train_out = os.path.join(OUT_DIR, f"train_{latest_year}_1~10 after_preprocessing.csv")
    db_out = os.path.join(OUT_DIR, f"db_latest_{latest_year}_11 after_preprocessing.csv")
    test_out = os.path.join(OUT_DIR, f"test_{latest_year}_12 after_preprocessing.csv")
    write_csv(train_clean, train_out)
    write_csv(db_clean, db_out)
    write_csv(test_clean, test_out)

    # Summary
    def brief(tag, dframe):
        cols = dframe.columns
        uid = dframe["user_id"].nunique() if "user_id" in cols else np.nan
        tid = dframe["track_id"].nunique() if "track_id" in cols else np.nan
        print(f"[SUMMARY] {tag:<10} | rows={len(dframe):>8,} | cols={len(cols):>3} | users={uid:,} | tracks={tid:,}")

    brief(f"TRAIN {latest_year}", train_clean)
    brief(f"DB {latest_year}-11", db_clean)
    brief(f"TEST {latest_year}-12", test_clean)


if __name__ == "__main__":
    main()
