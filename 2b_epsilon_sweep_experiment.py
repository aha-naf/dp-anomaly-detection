from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

# Import the user's pipeline module
import 2b_IF_PIM as ux


EPSILON_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
PERCENT = 5.0
OUTPUT_DIR = Path("priv_exp_outputs/epsilon_sweep_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def overlap_at_k_sets(merged: pd.DataFrame, k: int):
    k_eff = min(k, len(merged))
    if k_eff == 0:
        return 0.0, 0.0

    raw_top = set(merged.nlargest(k_eff, "unsup_score_raw")["trip_key"])
    noisy_top = set(merged.nlargest(k_eff, "unsup_score_noisy")["trip_key"])

    overlap = len(raw_top & noisy_top) / k_eff
    union = raw_top | noisy_top
    jaccard = 0.0 if not union else len(raw_top & noisy_top) / len(union)
    return overlap, jaccard

def precision_recall_fixed_raw_ground_truth(merged: pd.DataFrame, percent: float = 5.0):
    if "trip_key" not in merged.columns:
        merged = merged.copy()
        merged["trip_key"] = merged["user_id"].astype(str) + "_" + merged["traj_id"].astype(str)

    k = max(1, int(round(len(merged) * percent / 100.0)))

    raw_top = set(merged.nlargest(k, "unsup_score_raw")["trip_key"])
    noisy_top = set(merged.nlargest(k, "unsup_score_noisy")["trip_key"])

    tp = len(raw_top & noisy_top)
    fp = len(noisy_top - raw_top)
    fn = len(raw_top - noisy_top)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "raw_anomalies": len(raw_top),
        "noisy_anomalies": len(noisy_top),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def precision_recall_raw_threshold(merged, percent=5.0):
    if "trip_key" not in merged.columns:
        merged = merged.copy()
        merged["trip_key"] = merged["user_id"].astype(str) + "_" + merged["traj_id"].astype(str)

    # threshold from raw
    raw_thresh = merged["unsup_score_raw"].quantile(1 - percent / 100.0)

    # raw ground truth
    raw_top = set(merged[merged["unsup_score_raw"] >= raw_thresh]["trip_key"])

    # noisy predictions 
    noisy_top = set(merged[merged["unsup_score_noisy"] >= raw_thresh]["trip_key"])

    tp = len(raw_top & noisy_top)
    fp = len(noisy_top - raw_top)
    fn = len(raw_top - noisy_top)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "raw_threshold": raw_thresh,
    }

def precision_recall_percentile(merged: pd.DataFrame, percent: float = 5.0) -> Dict[str, float]:
    if "trip_key" not in merged.columns:
        merged = merged.copy()
        merged["trip_key"] = merged["user_id"].astype(str) + "_" + merged["traj_id"].astype(str)

    raw_thresh = merged["unsup_score_raw"].quantile(1 - percent / 100.0)
    noisy_thresh = merged["unsup_score_noisy"].quantile(1 - percent / 100.0)

    raw_top = set(merged[merged["unsup_score_raw"] >= raw_thresh]["trip_key"])
    noisy_top = set(merged[merged["unsup_score_noisy"] >= noisy_thresh]["trip_key"])

    tp = len(raw_top & noisy_top)
    fp = len(noisy_top - raw_top)
    fn = len(raw_top - noisy_top)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "raw_threshold": float(raw_thresh),
        "noisy_threshold": float(noisy_thresh),
        "raw_anomalies": float(len(raw_top)),
        "noisy_anomalies": float(len(noisy_top)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def run_one_epsilon(epsilon: float) -> Dict[str, float]:
    print(f"\n=== Running epsilon = {epsilon} ===")
    start = time.time()

    ux.CENTER_CACHE.clear()
    ux.PIM_CACHE.clear()

    trips = ux.load_geolife_trips(ux.ROOT, max_users=ux.MAX_USERS)
    if len(trips) == 0:
        raise RuntimeError("No trips found. Check ROOT in unsupervised_xiao_fast.py")

    model = ux.build_markov_model(trips, ux.GRID_SIZE)

    raw_feat = ux.build_feature_dataframe(trips, ux.GRID_SIZE, preprocess=True)
    raw_scores = ux.fit_iforest_scores(raw_feat).rename(columns={"unsup_score": "unsup_score_raw"})

    noisy_trips, meta_df = ux.privatize_all_trips_pim(
        trips=trips,
        model=model,
        epsilon=epsilon,
        delta_locset=ux.DELTA_LOCSET,
        grid_size=ux.GRID_SIZE,
        seed=ux.RANDOM_STATE,
    )

    noisy_feat = ux.build_feature_dataframe(noisy_trips, ux.GRID_SIZE, preprocess=False)
    noisy_scores = ux.fit_iforest_scores(noisy_feat).rename(columns={"unsup_score": "unsup_score_noisy"})

    merged = ux.compare_raw_vs_noisy(raw_scores, noisy_scores)
    pr = precision_recall_raw_threshold(merged, percent=PERCENT)

    row = {
        "epsilon": float(epsilon),
        "n_trips": float(len(merged)),
        "spearman_rank_corr": float(merged["unsup_score_raw"].corr(merged["unsup_score_noisy"], method="spearman")),
        "mean_abs_score_diff": float((merged["unsup_score_raw"] - merged["unsup_score_noisy"]).abs().mean()),
        "top_10_overlap": float(overlap_at_k_sets(merged, 10)[0]),
        "top_20_overlap": float(overlap_at_k_sets(merged, 20)[0]),
        "top_50_overlap": float(overlap_at_k_sets(merged, 50)[0]),
        "jaccard_10": float(overlap_at_k_sets(merged, 10)[1]),
        "jaccard_20": float(overlap_at_k_sets(merged, 20)[1]),
        "jaccard_50": float(overlap_at_k_sets(merged, 50)[1]),
        "avg_locset_size": float(meta_df["avg_locset_size"].mean()) if not meta_df.empty else 0.0,
        "avg_drift_ratio": float(meta_df["drift_ratio"].mean()) if not meta_df.empty else 0.0,
        "avg_points_after_preprocess": float(meta_df["points_after_preprocess"].mean()) if not meta_df.empty else 0.0,
        "runtime_sec": float(time.time() - start),
        **pr,
    }

    per_eps_dir = OUTPUT_DIR / f"epsilon_{str(epsilon).replace('.', '_')}"
    per_eps_dir.mkdir(exist_ok=True)
    merged.to_csv(per_eps_dir / "merged_scores.csv", index=False)
    raw_feat.to_csv(per_eps_dir / "raw_features.csv", index=False)
    noisy_feat.to_csv(per_eps_dir / "noisy_features.csv", index=False)
    meta_df.to_csv(per_eps_dir / "meta.csv", index=False)

    return row


def make_plot(df: pd.DataFrame, y_cols: List[str], title: str, ylabel: str, filename: str) -> None:
    plt.figure(figsize=(7, 4.5))
    for col in y_cols:
        plt.plot(df["epsilon"], df[col], marker="o", label=col)
    plt.xscale("log", base=2)
    plt.xlabel("Epsilon")
    plt.ylabel(ylabel)
    plt.title(title)
    if len(y_cols) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()


def main() -> None:
    rows = []
    for eps in EPSILON_VALUES:
        rows.append(run_one_epsilon(eps))

    results_df = pd.DataFrame(rows).sort_values("epsilon").reset_index(drop=True)
    results_csv = OUTPUT_DIR / "epsilon_sweep_summary.csv"
    results_df.to_csv(results_csv, index=False)

    make_plot(
        results_df,
        ["spearman_rank_corr"],
        "Epsilon vs Spearman Correlation",
        "Spearman correlation",
        "epsilon_vs_spearman.png",
    )
    make_plot(
        results_df,
        ["top_10_overlap", "top_20_overlap", "top_50_overlap"],
        "Epsilon vs Top-k Overlap",
        "Overlap",
        "epsilon_vs_topk_overlap.png",
    )
    make_plot(
        results_df,
        ["f1", "precision", "recall"],
        "Epsilon vs Percentile-based Metrics",
        "Score",
        "epsilon_vs_f1_precision_recall.png",
    )
    make_plot(
        results_df,
        ["mean_abs_score_diff"],
        "Epsilon vs Mean Absolute Score Difference",
        "Mean absolute score difference",
        "epsilon_vs_mad.png",
    )

    print("\nSaved summary:")
    print(results_csv)
    print(OUTPUT_DIR / "epsilon_vs_spearman.png")
    print(OUTPUT_DIR / "epsilon_vs_topk_overlap.png")
    print(OUTPUT_DIR / "epsilon_vs_f1_precision_recall.png")
    print(OUTPUT_DIR / "epsilon_vs_mad.png")


if __name__ == "__main__":
    main()
