from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

GEOLIFE_ROOT = Path(r"/Users/ahnafhassan/Desktop/python_projects/Privacy Course/Geolife Trajectories 1.3/Data")

MIN_POINTS_PER_TRIP = 10
GRID_SIZE = 0.05
ALPHA = 1.0
MAX_TRANSITIONS_PER_TRIP = 50
RANDOM_SEED = 42

EPSILON_VALUES = [0.25,0.5,1.0,2.0,4.0,8.0]

OUTPUT_DIR = Path("priv_exp_output_markov dp")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Trip:
    user_id: str
    trip_id: str
    file_path: Path
    states: List[str]


# ============================================================
# IO + DISCRETIZATION
# ============================================================

def read_plt_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        file_path,
        skiprows=6,
        header=None,
        names=["lat", "lon", "unused", "altitude", "days", "date", "time"],
    )
    df = df.dropna(subset=["lat", "lon"])
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return df


def gps_to_grid_state(lat: float, lon: float, grid_size: float = GRID_SIZE) -> str:
    lat_bin = math.floor(lat / grid_size)
    lon_bin = math.floor(lon / grid_size)
    return f"{lat_bin}_{lon_bin}"


def trip_df_to_states(df: pd.DataFrame, grid_size: float = GRID_SIZE) -> List[str]:
    raw_states = [gps_to_grid_state(lat, lon, grid_size) for lat, lon in zip(df["lat"], df["lon"])]

    collapsed_states: List[str] = []
    for s in raw_states:
        if not collapsed_states or collapsed_states[-1] != s:
            collapsed_states.append(s)

    return collapsed_states


def load_geolife_trips(root: Path) -> List[Trip]:
    trips: List[Trip] = []

    user_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    for user_dir in user_dirs:
        traj_dir = user_dir / "Trajectory"
        if not traj_dir.exists():
            continue

        for plt_file in sorted(traj_dir.glob("*.plt")):
            try:
                df = read_plt_file(plt_file)
                states = trip_df_to_states(df, grid_size=GRID_SIZE)

                if len(states) < MIN_POINTS_PER_TRIP:
                    continue

                trips.append(
                    Trip(
                        user_id=user_dir.name,
                        trip_id=plt_file.stem,
                        file_path=plt_file,
                        states=states,
                    )
                )
            except Exception as e:
                print(f"Skipping {plt_file} due to error: {e}")

    return trips


# ============================================================
# MODEL BUILDING
# ============================================================

def build_global_markov_model(
    trips: List[Trip],
) -> Tuple[Dict[str, Counter], List[str], Dict[str, int]]:
    transition_counts: Dict[str, Counter] = defaultdict(Counter)
    state_vocab_set = set()

    for trip in trips:
        states = trip.states
        state_vocab_set.update(states)
        for a, b in zip(states[:-1], states[1:]):
            transition_counts[a][b] += 1

    state_vocab = sorted(state_vocab_set)
    state_to_idx = {s: i for i, s in enumerate(state_vocab)}
    return transition_counts, state_vocab, state_to_idx


def transition_probability(
    current_state: str,
    next_state: str,
    transition_counts: Dict[str, Counter],
    vocab_size: int,
    alpha: float = ALPHA,
) -> float:
    outgoing = transition_counts.get(current_state, Counter())
    total = sum(outgoing.values())
    count = outgoing.get(next_state, 0)
    return (count + alpha) / (total + alpha * vocab_size)


def build_bounded_count_matrix_from_trips(
    trips: List[Trip],
    state_to_idx: Dict[str, int],
    max_transitions_per_trip: int = MAX_TRANSITIONS_PER_TRIP,
) -> np.ndarray:
    n = len(state_to_idx)
    C = np.zeros((n, n), dtype=float)

    for trip in trips:
        transitions = list(zip(trip.states[:-1], trip.states[1:]))[:max_transitions_per_trip]
        for a, b in transitions:
            if a not in state_to_idx or b not in state_to_idx:
                continue
            i = state_to_idx[a]
            j = state_to_idx[b]
            C[i, j] += 1.0

    return C


def privatize_count_matrix(
    count_matrix: np.ndarray,
    epsilon: float,
    sensitivity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")

    scale = sensitivity / epsilon
    noise = rng.laplace(loc=0.0, scale=scale, size=count_matrix.shape)
    noisy = count_matrix + noise
    noisy = np.maximum(noisy, 0.0)
    return noisy


def build_transition_matrix_from_count_matrix(
    count_matrix: np.ndarray,
    alpha: float = ALPHA,
) -> np.ndarray:
    n = count_matrix.shape[0]
    M = np.zeros_like(count_matrix, dtype=float)

    for i in range(n):
        row_sum = float(count_matrix[i].sum())
        M[i, :] = (count_matrix[i, :] + alpha) / (row_sum + alpha * n)

    return M


# ============================================================
# SCORING
# ============================================================

def score_trip_average_negative_log_likelihood(
    trip: Trip,
    transition_counts: Dict[str, Counter],
    state_vocab: List[str],
    alpha: float = ALPHA,
) -> float:
    states = trip.states
    if len(states) < 2:
        return float("inf")

    vocab_size = len(state_vocab)
    log_probs: List[float] = []

    for a, b in zip(states[:-1], states[1:]):
        p = transition_probability(a, b, transition_counts, vocab_size, alpha)
        log_probs.append(math.log(p))

    return -sum(log_probs) / len(log_probs)


def score_trip_with_matrix(
    trip: Trip,
    transition_matrix: np.ndarray,
    state_to_idx: Dict[str, int],
) -> float:
    states = trip.states
    if len(states) < 2:
        return float("inf")

    n = transition_matrix.shape[0]
    fallback_p = 1.0 / n
    log_probs: List[float] = []

    for a, b in zip(states[:-1], states[1:]):
        if a not in state_to_idx or b not in state_to_idx:
            p = fallback_p
        else:
            i = state_to_idx[a]
            j = state_to_idx[b]
            p = float(transition_matrix[i, j])

        p = max(p, 1e-12)
        log_probs.append(math.log(p))

    return -sum(log_probs) / len(log_probs)


def score_all_trips_raw_model(
    trips: List[Trip],
    transition_counts: Dict[str, Counter],
    state_vocab: List[str],
) -> pd.DataFrame:
    rows = []

    for trip in trips:
        anomaly_score = score_trip_average_negative_log_likelihood(
            trip, transition_counts, state_vocab, alpha=ALPHA
        )
        rows.append(
            {
                "user_id": trip.user_id,
                "trip_id": trip.trip_id,
                "file_path": str(trip.file_path),
                "num_states": len(trip.states),
                "anomaly_score": anomaly_score,
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)


def score_all_trips_matrix_model(
    trips: List[Trip],
    transition_matrix: np.ndarray,
    state_to_idx: Dict[str, int],
) -> pd.DataFrame:
    rows = []

    for trip in trips:
        anomaly_score = score_trip_with_matrix(trip, transition_matrix, state_to_idx)
        rows.append(
            {
                "user_id": trip.user_id,
                "trip_id": trip.trip_id,
                "file_path": str(trip.file_path),
                "num_states": len(trip.states),
                "anomaly_score": anomaly_score,
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)


# ============================================================
# EVALUATION
# ============================================================
def mean_absolute_difference(raw_scores: np.ndarray, dp_scores: np.ndarray) -> float:
    raw_scores = np.asarray(raw_scores, dtype=float)
    dp_scores = np.asarray(dp_scores, dtype=float)
    return float(np.mean(np.abs(raw_scores - dp_scores)))

def rankdata_average(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)

    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata_average(np.asarray(x, dtype=float))
    ry = rankdata_average(np.asarray(y, dtype=float))
    return pearson_corr(rx, ry)


def overlap_at_k(raw_df: pd.DataFrame, dp_df: pd.DataFrame, k: int) -> float:
    raw_top = set(raw_df.head(k).apply(lambda r: (r["user_id"], r["trip_id"]), axis=1))
    dp_top = set(dp_df.head(k).apply(lambda r: (r["user_id"], r["trip_id"]), axis=1))
    return len(raw_top & dp_top) / max(k, 1)


def compare_score_tables(raw_df: pd.DataFrame, dp_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df.merge(
        dp_df,
        on=["user_id", "trip_id", "file_path", "num_states"],
        suffixes=("_raw", "_dp"),
    )


def percentile_metrics(raw_df: pd.DataFrame, dp_df: pd.DataFrame, percentile: float = 95.0) -> Dict[str, float]:
    raw_threshold = np.percentile(raw_df["anomaly_score"], percentile)
    dp_threshold = np.percentile(dp_df["anomaly_score"], percentile)

    raw_anomalies = set(
        raw_df.loc[raw_df["anomaly_score"] >= raw_threshold, ["user_id", "trip_id"]].apply(tuple, axis=1)
    )
    dp_anomalies = set(
        dp_df.loc[dp_df["anomaly_score"] >= dp_threshold, ["user_id", "trip_id"]].apply(tuple, axis=1)
    )

    tp = len(raw_anomalies & dp_anomalies)
    fp = len(dp_anomalies - raw_anomalies)
    fn = len(raw_anomalies - dp_anomalies)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "raw_threshold": raw_threshold,
        "dp_threshold": dp_threshold,
        "raw_anomalies": len(raw_anomalies),
        "dp_anomalies": len(dp_anomalies),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# PLOTTING
# ============================================================

def make_plot(df: pd.DataFrame, xcol: str, ycol: str, title: str, ylabel: str, filename: str) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot(df[xcol], df[ycol], marker="o")
    plt.xscale("log")
    plt.xticks(df[xcol], df[xcol]) 
    plt.xlabel("Epsilon")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()

def make_multi_overlap_plot(
    df: pd.DataFrame,
    xcol: str,
    overlap_cols: List[str],
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    plt.figure(figsize=(7, 4.5))
    for col in overlap_cols:
        plt.plot(df[xcol], df[col], marker="o", label=col)

    plt.xscale("log")
    plt.xticks(df[xcol], df[xcol]) 
    plt.xlabel("Epsilon")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()
# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("Loading GeoLife trips...")
    trips = load_geolife_trips(GEOLIFE_ROOT)
    print(f"Loaded {len(trips)} trips")

    if not trips:
        print("No valid trips found. Check GEOLIFE_ROOT path.")
        return

    print("Building raw model...")
    transition_counts, state_vocab, state_to_idx = build_global_markov_model(trips)

    print("Scoring raw trips once...")
    raw_results_df = score_all_trips_raw_model(trips, transition_counts, state_vocab)
    raw_results_df.to_csv(OUTPUT_DIR / "raw_results.csv", index=False)

    print("Building bounded count matrix once...")
    bounded_count_matrix = build_bounded_count_matrix_from_trips(
        trips=trips,
        state_to_idx=state_to_idx,
        max_transitions_per_trip=MAX_TRANSITIONS_PER_TRIP,
    )

    sensitivity = float(MAX_TRANSITIONS_PER_TRIP)
    all_rows = []

    for eps in EPSILON_VALUES:
        print(f"\nRunning epsilon = {eps}")
        rng = np.random.default_rng(RANDOM_SEED)

        dp_count_matrix = privatize_count_matrix(
            count_matrix=bounded_count_matrix,
            epsilon=eps,
            sensitivity=sensitivity,
            rng=rng,
        )

        dp_transition_matrix = build_transition_matrix_from_count_matrix(
            dp_count_matrix,
            alpha=ALPHA,
        )

        dp_results_df = score_all_trips_matrix_model(
            trips=trips,
            transition_matrix=dp_transition_matrix,
            state_to_idx=state_to_idx,
        )

        dp_results_df.to_csv(OUTPUT_DIR / f"dp_results_eps_{str(eps).replace('.', '_')}.csv", index=False)

        merged = compare_score_tables(raw_results_df, dp_results_df)

        raw_scores = merged["anomaly_score_raw"].to_numpy()
        dp_scores = merged["anomaly_score_dp"].to_numpy()

        pearson = pearson_corr(raw_scores, dp_scores)
        spearman = spearman_corr(raw_scores, dp_scores)
        mad = mean_absolute_difference(raw_scores, dp_scores)

        pct = percentile_metrics(raw_results_df, dp_results_df, percentile=95)

        row = {
            "epsilon": eps,
            "pearson": pearson,
            "spearman": spearman,
            "mad":mad,
            "overlap@10": overlap_at_k(raw_results_df, dp_results_df, 10),
            "overlap@20": overlap_at_k(raw_results_df, dp_results_df, 20),
            "overlap@50": overlap_at_k(raw_results_df, dp_results_df, 50),
            "overlap@100": overlap_at_k(raw_results_df, dp_results_df, 100),
            "overlap@200": overlap_at_k(raw_results_df, dp_results_df, 200),
            "precision": pct["precision"],
            "recall": pct["recall"],
            "f1": pct["f1"],
            "tp": pct["tp"],
            "fp": pct["fp"],
            "fn": pct["fn"],
            "raw_threshold": pct["raw_threshold"],
            "dp_threshold": pct["dp_threshold"],
        }
        all_rows.append(row)

    results_df = pd.DataFrame(all_rows).sort_values("epsilon").reset_index(drop=True)
    results_df.to_csv(OUTPUT_DIR / "epsilon_sweep_summary.csv", index=False)

    print("\nFinal summary:")
    print(results_df)

    make_plot(results_df, "epsilon", "spearman", "Epsilon vs Spearman Correlation", "Spearman", "eps_vs_spearman.png")
    make_plot(results_df, "epsilon", "pearson", "Epsilon vs Pearson Correlation", "Pearson", "eps_vs_pearson.png")
    make_plot(results_df, "epsilon", "mad", "Epsilon vs Mean Absolute Difference", "MAD", "eps_vs_mad.png")
    make_plot(results_df, "epsilon", "f1", "Epsilon vs F1 Score", "F1", "eps_vs_f1.png")
    make_plot(results_df, "epsilon", "precision", "Epsilon vs Precision", "Precision", "eps_vs_precision.png")
    make_plot(results_df, "epsilon", "recall", "Epsilon vs Recall", "Recall", "eps_vs_recall.png")

    make_multi_overlap_plot(
        results_df,
        "epsilon",
        [ "overlap@50", "overlap@100", "overlap@200"],
        "Epsilon vs Top-k Overlap",
        "Overlap Rate",
        "eps_vs_all_overlaps.png",
    )
    print(f"\nSaved everything in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()