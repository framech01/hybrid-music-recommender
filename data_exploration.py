import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================================
# Default dataset path (co-located CSV file)
CONTEXT_FEATURES_PATH = "context_content_features.csv"
# =========================================================================

plt.rcParams["axes.unicode_minus"] = False  # ensure minus sign renders properly


def read_csv_robust(path: str, nrows: int | None = None) -> pd.DataFrame | None:

    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return None
    try:
        return pd.read_csv(
            path,
            nrows=nrows,
            low_memory=False,
            on_bad_lines="skip",
            encoding="utf-8",
        )
    except UnicodeDecodeError:
        print(f"[WARN] UTF-8 failed {os.path.basename(path)}")
        return pd.read_csv(
            path,
            nrows=nrows,
            low_memory=False,
            on_bad_lines="skip",
            encoding="latin-1",
        )


def sample_df(df: pd.DataFrame, frac: float | None = None) -> pd.DataFrame:
    if frac is None or not (0 < frac < 1):
        return df
    return df.sample(frac=frac, random_state=42)


def detect_timestamp(df: pd.DataFrame) -> str | None:
    for col in ["created_at", "timestamp", "time", "ts"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
            return col
    return None


def print_table(
    df: pd.DataFrame,
    title: str,
    max_rows: int = 30,
    round_digits: int | None = 4,
) -> None:
    pd.set_option("display.max_rows", max_rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)

    print(f"\n {title}")
    if round_digits is not None:
        with pd.option_context("display.float_format", lambda x: f"{x:.{round_digits}f}"):
            print(df.head(max_rows).to_string(index=True))
    else:
        print(df.head(max_rows).to_string(index=True))


# ------------------------------ Core EDA ---------------------------------
def analyze_all_context_features(
    file_path: str,
    *,
    sample_frac: float | None = None,
    max_rows: int | None = None,
    show_plots: bool = True,
    save_plots: bool = False,
    topn: int = 20,
) -> pd.DataFrame | None:
    """
    Run the EDA pipeline on a given CSV file, excluding correlation heatmap/output.

    Parameters
    ----------
    file_path : str
        Path to the CSV file to analyze.
    sample_frac : float | None
        Optional fraction (0,1) for random sampling to speed up EDA on large data.
    max_rows : int | None
        Optional hard cap on the number of rows to read from disk.
    show_plots : bool
        Whether to display figures interactively.
    save_plots : bool
        Whether to write figures as PNGs into './eda_outputs'.
    topn : int
        Row cap for printed tables; also controls the outlier ranking length.

    Returns
    -------
    pd.DataFrame | None
        The DataFrame used in the analysis (useful for interactive follow-ups).
        Returns None if the file could not be read.
    """
    file_name = os.path.basename(file_path)
    print(f"[EDA] Context Exploration start  | file='{file_name}'")
    print("=" * 90)

    # 1) Load
    df = read_csv_robust(file_path, nrows=max_rows)
    if df is None:
        return None

    # 2) Optional sampling
    if sample_frac:
        df = sample_df(df, sample_frac)
        print(f"[INFO] sampled dataframe shape={df.shape}")
    else:
        print(f"[INFO] dataframe shape={df.shape}")

    # 3) Timestamp detection
    ts_col = detect_timestamp(df)
    if ts_col:
        print(f"[INFO] timestamp detected: {ts_col}")

    # 4) Structure / Missing / Unique
    missing_cnt = df.isna().sum().sort_values(ascending=False)
    missing_rate = (missing_cnt / max(len(df), 1) * 100).round(3)

    struct = (
        pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "missing_cnt": missing_cnt,
                "missing_rate_%": missing_rate,
                "unique_cnt": df.nunique(dropna=True),
            }
        )
        .sort_values("missing_cnt", ascending=False)
    )
    print_table(struct, "Structure / Missing / Unique (sorted by missing)", max_rows=topn)

    # 5) Numeric describe (+skew/kurtosis), excluding ID-like columns
    exclude_cols = {"user_id", "id"}
    all_num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in all_num_cols if c not in exclude_cols]

    if num_cols:
        desc = df[num_cols].describe().T
        extra = pd.DataFrame(
            {
                "missing_rate_%": (df[num_cols].isna().sum() / len(df) * 100).round(3),
                "skew": df[num_cols].skew(numeric_only=True),
                "kurtosis": df[num_cols].kurtosis(numeric_only=True),
            }
        )
        desc = desc.join(extra, how="left")
        print_table(desc, "Numeric describe (+missing_rate, skew, kurtosis)", max_rows=topn)
    else:
        print("[Info] no numeric columns found; skip numeric describe.")

    # 6) IQR-based outlier table (user_id/id already excluded via num_cols)
    if num_cols:
        rows = []
        for c in num_cols:
            s = df[c].replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                rows.append((c, 0, 0.0, np.nan, np.nan))
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            # Why 1.5*IQR: standard boxplot rule balancing robustness/sensitivity.
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            cnt = int(((s < lo) | (s > hi)).sum())
            rate = (cnt / len(s) * 100.0) if len(s) else 0.0
            rows.append((c, cnt, round(rate, 3), lo, hi))

        iqr_tbl = (
            pd.DataFrame(
                rows,
                columns=["column", "outlier_cnt", "outlier_rate_%", "lower_bound", "upper_bound"],
            )
            .sort_values("outlier_rate_%", ascending=False)
        )
        print_table(iqr_tbl.head(topn), f"IQR-based outliers", max_rows=topn)
    else:
        iqr_tbl = pd.DataFrame()

    # 7) Plots (histograms + boxplot only; correlation heatmap removed)
    outdir = "./eda_outputs"
    if save_plots:
        os.makedirs(outdir, exist_ok=True)

    # 7-1) Histograms (per numeric column)
    if num_cols:
        for c in num_cols:
            s = df[c].replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                continue
            fig = plt.figure()
            plt.hist(s.values, bins=50)
            plt.title(f"context: Distribution - {c}")
            plt.xlabel(c)
            plt.ylabel("count")
            if save_plots:
                fig.savefig(os.path.join(outdir, f"context_hist_{c}.png"), bbox_inches="tight", dpi=150)
            if show_plots:
                plt.show()
            plt.close(fig)

        # 7-2) Boxplot for all numeric columns
        fig = plt.figure(figsize=(max(10, len(num_cols) * 0.8), 6))
        df[num_cols].boxplot(rot=45)
        plt.title("context: numeric feature boxplot")
        if save_plots:
            fig.savefig(os.path.join(outdir, "boxplot_context.png"), bbox_inches="tight", dpi=150)
        if show_plots:
            plt.show()
        plt.close(fig)
    else:
        print("[Info] no numeric columns; skip hist/boxplot.")

    # 8) Time patterns (if a timestamp column exists and is non-empty)
    if ts_col:
        ts = df[ts_col].dropna()
        if not ts.empty:
            by_month = ts.dt.to_period("M").value_counts().sort_index()
            print_table(by_month.rename("count").to_frame(), "Events by month (context)", max_rows=topn)

            hours = ts.dt.hour.value_counts().sort_index()
            print_table(hours.rename("count").to_frame(), "Hourly distribution (context)", max_rows=topn)

            # Monthly bar chart
            fig = plt.figure(figsize=(max(8, len(by_month) * 0.2), 4))
            by_month.index = by_month.index.astype(str)  # PeriodIndex → string for plotting
            by_month.plot(kind="bar")
            plt.title("context: events by month")
            plt.xlabel("month")
            plt.ylabel("count")
            if save_plots:
                fig.savefig(os.path.join(outdir, "context_by_month.png"), bbox_inches="tight", dpi=150)
            if show_plots:
                plt.show()
            plt.close(fig)

            # Hourly bar chart
            fig = plt.figure()
            hours.plot(kind="bar")
            plt.title("context: hourly distribution (0-23)")
            plt.xlabel("hour")
            plt.ylabel("count")
            if save_plots:
                fig.savefig(os.path.join(outdir, "context_hourly.png"), bbox_inches="tight", dpi=150)
            if show_plots:
                plt.show()
            plt.close(fig)

    return df

if __name__ == "__main__":
    analyze_all_context_features(
        CONTEXT_FEATURES_PATH,
        sample_frac=None,      
        max_rows=None,          
        show_plots=True,    
        save_plots=True,        
        topn=30,
    )
