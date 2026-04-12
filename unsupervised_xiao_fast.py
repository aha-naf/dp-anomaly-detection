from __future__ import annotations

import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================
ROOT = Path("/Users/ahnafhassan/Desktop/python_projects/Privacy Course/Geolife Trajectories 1.3/Data")
GRID_SIZE = 0.05
MIN_POINTS_PER_TRIP = 10
MAX_USERS = 3
RANDOM_STATE = 42

# Speed / scale controls
DOWNSAMPLE_EVERY_K_POINTS = 5      # 1 = no downsampling
COLLAPSE_SAME_STATE_RUNS = False   # True = privatize one point per repeated grid-cell run

# Xiao-style release params
EPSILON = 8
DELTA_LOCSET = 0.05
INIT_PRIOR_TOPK = 20
POSTERIOR_TOPK = 30
MIN_LOCSET_SIZE = 2
ISOTROPIC_SAMPLES = 20
MAX_CANDIDATES_PER_STEP = 15

# IF params
N_ESTIMATORS = 200


# ============================================================
# GLOBAL CACHES
# ============================================================
CENTER_CACHE: Dict[Tuple[Tuple[int, int], float], Tuple[float, float]] = {}
PIM_CACHE: Dict[Tuple[Tuple[int, int], ...], Dict] = {}


# ============================================================
# BASIC HELPERS
# ============================================================
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def grid_cell(lat: float, lon: float, grid_size: float) -> Tuple[int, int]:
    return (int(math.floor(lat / grid_size)), int(math.floor(lon / grid_size)))


def cell_center(cell: Tuple[int, int], grid_size: float) -> Tuple[float, float]:
    key = (cell, grid_size)
    if key not in CENTER_CACHE:
        i, j = cell
        CENTER_CACHE[key] = ((i + 0.5) * grid_size, (j + 0.5) * grid_size)
    return CENTER_CACHE[key]


def read_plt_file(filepath: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            filepath,
            skiprows=6,
            header=None,
            names=["lat", "lon", "unused", "altitude", "date_days", "date", "time"],
        )
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "timestamp"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_geolife_trips(root: Path, max_users: Optional[int] = None) -> List[Dict]:
    trips = []
    user_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if max_users is not None:
        user_dirs = user_dirs[:max_users]

    for user_dir in user_dirs:
        traj_dir = user_dir / "Trajectory"
        if not traj_dir.exists():
            continue

        for plt_file in sorted(traj_dir.glob("*.plt")):
            df = read_plt_file(plt_file)
            if len(df) < MIN_POINTS_PER_TRIP:
                continue

            trips.append({
                "user_id": user_dir.name,
                "traj_id": plt_file.stem,
                "df": df
            })

    return trips


def trip_to_states(df: pd.DataFrame, grid_size: float) -> List[Tuple[int, int]]:
    return [grid_cell(lat, lon, grid_size) for lat, lon in zip(df["lat"], df["lon"])]


def collapse_consecutive(states: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not states:
        return []
    out = [states[0]]
    for s in states[1:]:
        if s != out[-1]:
            out.append(s)
    return out


def preprocess_trip_for_privatization(df: pd.DataFrame, grid_size: float) -> pd.DataFrame:
    """
    Optional speedups:
    1) downsample dense timestamp streams
    2) optionally keep one representative point per repeated grid-cell run
    """
    out = df.sort_values("timestamp").reset_index(drop=True).copy()

    if DOWNSAMPLE_EVERY_K_POINTS > 1:
        out = out.iloc[::DOWNSAMPLE_EVERY_K_POINTS].reset_index(drop=True)

    if COLLAPSE_SAME_STATE_RUNS and len(out) > 1:
        states = trip_to_states(out, grid_size)
        keep_idx = [0]
        for i in range(1, len(states)):
            if states[i] != states[keep_idx[-1]]:
                keep_idx.append(i)
        out = out.iloc[keep_idx].reset_index(drop=True)

    return out


# ============================================================
# MARKOV MODEL
# ============================================================
def build_markov_model(trips: List[Dict], grid_size: float):
    transition_counts = Counter()
    outgoing_counts = Counter()
    state_counts = Counter()
    successors = defaultdict(Counter)

    for trip in trips:
        # Use the SAME preprocessing as the privatization pipeline
        proc_df = preprocess_trip_for_privatization(trip["df"], grid_size)
        if len(proc_df) < 2:
            continue

        states = collapse_consecutive(trip_to_states(proc_df, grid_size))

        for s in states:
            state_counts[s] += 1

        for a, b in zip(states[:-1], states[1:]):
            transition_counts[(a, b)] += 1
            outgoing_counts[a] += 1
            successors[a][b] += 1

    all_states = set(state_counts.keys())
    for (a, b) in transition_counts:
        all_states.add(a)
        all_states.add(b)

    global_total = sum(state_counts.values())
    global_prior = {}
    if global_total > 0:
        for s, c in state_counts.most_common(INIT_PRIOR_TOPK):
            global_prior[s] = c / global_total

    ssum = sum(global_prior.values())
    if ssum > 0:
        global_prior = {k: v / ssum for k, v in global_prior.items()}

    return {
        "transition_counts": transition_counts,
        "outgoing_counts": outgoing_counts,
        "state_counts": state_counts,
        "successors": successors,
        "all_states": all_states,
        "global_prior": global_prior,
    }


def propagate_prior_sparse(prev_post: Dict[Tuple[int, int], float], model: Dict) -> Dict[Tuple[int, int], float]:
    successors = model["successors"]
    out = defaultdict(float)

    for s, p in prev_post.items():
        succ = successors.get(s, {})
        total = sum(succ.values())
        if total > 0:
            for nxt, c in succ.items():
                out[nxt] += p * (c / total)

    if not out:
        return dict(model["global_prior"])

    total_prob = sum(out.values())
    if total_prob > 0:
        out = {k: v / total_prob for k, v in out.items()}
    return dict(out)


def build_delta_location_set(
    prior: Dict[Tuple[int, int], float],
    delta: float,
    min_size: int = 2,
    max_size: int = 50,
) -> List[Tuple[int, int]]:
    items = sorted(prior.items(), key=lambda kv: kv[1], reverse=True)
    locset = []
    cum = 0.0
    for s, p in items:
        locset.append(s)
        cum += p
        if cum >= (1.0 - delta) and len(locset) >= min_size:
            break
        if len(locset) >= max_size:
            break

    if len(locset) < min_size:
        for s, _ in items[len(locset):]:
            locset.append(s)
            if len(locset) >= min_size:
                break

    return locset if locset else list(prior.keys())[:min_size]


def nearest_surrogate(true_state: Tuple[int, int], locset: List[Tuple[int, int]], grid_size: float) -> Tuple[int, int]:
    if true_state in locset:
        return true_state

    tlat, tlon = cell_center(true_state, grid_size)
    best = None
    best_d = float("inf")
    for s in locset:
        lat, lon = cell_center(s, grid_size)
        d = (lat - tlat) ** 2 + (lon - tlon) ** 2
        if d < best_d:
            best_d = d
            best = s
    return best


# ============================================================
# CONVEX / POLYGON UTILS
# ============================================================
def unique_rows(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    return np.unique(points, axis=0)


def convex_hull_polygon(points: np.ndarray) -> np.ndarray:
    """
    Returns convex hull vertices in CCW order.
    Handles degenerate 0D/1D cases by creating a tiny thin polygon.
    """
    points = unique_rows(points)

    if len(points) == 0:
        raise ValueError("No points provided to convex_hull_polygon.")

    if len(points) == 1:
        p = points[0]
        eps = 1e-9
        return np.array([
            [p[0], p[1]],
            [p[0] + eps, p[1]],
            [p[0], p[1] + eps],
        ])

    if len(points) == 2:
        p1, p2 = points
        v = p2 - p1
        perp = np.array([-v[1], v[0]], dtype=float)
        nrm = np.linalg.norm(perp)

        if nrm < 1e-12:
            perp = np.array([1e-9, 0.0])
        else:
            perp = perp / nrm * 1e-9

        mid = (p1 + p2) / 2.0
        return np.array([p1, p2, mid + perp])

    base = points[1] - points[0]
    collinear = True
    for i in range(2, len(points)):
        v = points[i] - points[0]
        cross = base[0] * v[1] - base[1] * v[0]
        if abs(cross) > 1e-12:
            collinear = False
            break

    if collinear:
        direction = base.astype(float)
        norm = np.linalg.norm(direction)

        if norm < 1e-12:
            p = points[0]
            eps = 1e-9
            return np.array([
                [p[0], p[1]],
                [p[0] + eps, p[1]],
                [p[0], p[1] + eps],
            ])

        direction = direction / norm
        perp = np.array([-direction[1], direction[0]]) * 1e-9

        projections = points @ direction
        p_min = points[np.argmin(projections)]
        p_max = points[np.argmax(projections)]

        return np.array([
            p_min + perp,
            p_max + perp,
            p_max - perp,
            p_min - perp,
        ])

    hull = ConvexHull(points)
    return points[hull.vertices]


def polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))




def build_sensitivity_hull(locset: List[Tuple[int, int]], grid_size: float) -> np.ndarray:
    pts = np.array([cell_center(s, grid_size) for s in locset], dtype=float)
    k0 = convex_hull_polygon(pts)
    diffs = []
    for i in range(len(k0)):
        for j in range(len(k0)):
            diffs.append(k0[i] - k0[j])
    diffs = np.array(diffs, dtype=float)
    K = convex_hull_polygon(diffs)

    if polygon_area(K) < 1e-18:
        center = np.mean(K, axis=0)
        eps = 1e-9
        K = np.array([
            center + [ eps, 0.0],
            center + [0.0,  eps],
            center + [-eps, 0.0],
            center + [0.0, -eps],
        ])

    return K


def sample_uniform_in_triangle(a: np.ndarray, b: np.ndarray, c: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    u = rng.random()
    v = rng.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    return a + u * (b - a) + v * (c - a)


def sample_uniform_in_convex_polygon(poly: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(poly) == 3:
        return sample_uniform_in_triangle(poly[0], poly[1], poly[2], rng)

    a0 = poly[0]
    tris = []
    areas = []
    for i in range(1, len(poly) - 1):
        tri = (a0, poly[i], poly[i + 1])
        area = polygon_area(np.array(tri))
        tris.append(tri)
        areas.append(area)
    areas = np.array(areas, dtype=float)
    probs = areas / areas.sum()
    idx = rng.choice(len(tris), p=probs)
    a, b, c = tris[idx]
    return sample_uniform_in_triangle(a, b, c, rng)


def transform_polygon(poly: np.ndarray, T: np.ndarray) -> np.ndarray:
    return (T @ poly.T).T


def inverse_sqrt_matrix_2x2(M: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(M)
    vals = np.clip(vals, 1e-12, None)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(vals))
    return vecs @ D_inv_sqrt @ vecs.T


def isotropic_transform(
    poly: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = 50,
    tol: float = 5e-3,
    max_rounds: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Re-sample until T stabilizes in Frobenius norm.
    """
    prev_T = None
    curr_samples = n_samples

    for _ in range(max_rounds):
        samples = np.array(
            [sample_uniform_in_convex_polygon(poly, rng) for _ in range(curr_samples)],
            dtype=float,
        )

        cov = np.cov(samples.T)
        T = inverse_sqrt_matrix_2x2(cov)

        if prev_T is not None:
            diff = np.linalg.norm(T - prev_T, ord="fro")
            if diff < tol:
                KI = transform_polygon(poly, T)
                return T, KI

        prev_T = T
        curr_samples *= 2

    KI = transform_polygon(poly, prev_T)
    return prev_T, KI


def polygon_halfspaces(poly: np.ndarray) -> List[Tuple[np.ndarray, float]]:
    hs = []
    m = len(poly)
    for i in range(m):
        a = poly[i]
        b = poly[(i + 1) % m]
        edge = b - a
        n = np.array([edge[1], -edge[0]], dtype=float)
        c = float(np.dot(n, a))
        if c < 0:
            n = -n
            c = -c
        hs.append((n, c))
    return hs


def minkowski_norm_from_halfspaces(x: np.ndarray, hs: List[Tuple[np.ndarray, float]]) -> float:
    vals = [float(np.dot(n, x) / max(c, 1e-12)) for n, c in hs]
    return max(vals)


def get_pim_aux(
    locset: List[Tuple[int, int]],
    grid_size: float,
    rng: np.random.Generator,
) -> Dict:
    key = tuple(sorted(locset))
    cached = PIM_CACHE.get(key)
    if cached is not None:
        return cached

    K = build_sensitivity_hull(locset, grid_size)
    T, KI = isotropic_transform(K, rng, n_samples=ISOTROPIC_SAMPLES)
    hs_KI = polygon_halfspaces(KI)

    aux = {
        "K": K,
        "T": T,
        "KI": KI,
        "hs_KI": hs_KI,
    }
    PIM_CACHE[key] = aux
    return aux


# ============================================================
# PIM RELEASE + EMISSION
# ============================================================
def pim_release(
    surrogate_state: Tuple[int, int],
    locset: List[Tuple[int, int]],
    epsilon: float,
    grid_size: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict]:
    x_star = np.array(cell_center(surrogate_state, grid_size), dtype=float)
    
    aux = get_pim_aux(locset, grid_size, rng)
    T = aux["T"]
    KI = aux["KI"]

    z0 = sample_uniform_in_convex_polygon(KI, rng)
    r = rng.gamma(shape=3.0, scale=1.0 / epsilon)
    noise = np.linalg.solve(T, r * z0)
    z = x_star + noise
    return z, aux



def build_posterior_candidate_states(
    prior: Dict[Tuple[int, int], float],
    model: Dict,
    locset: Optional[List[Tuple[int, int]]] = None,
    max_candidates: int = 1000,
    global_topk: int = 200,
) -> List[Tuple[int, int]]:
    """
    Build a broader candidate state set for posterior updating.

    Includes:
    - current prior support
    - current locset
    - one-hop successors from prior states
    - one-hop predecessors into prior states
    - a few globally common states as fallback
    """
    candidates = set(prior.keys())

    if locset is not None:
        candidates.update(locset)

    successors = model["successors"]
    transition_counts = model["transition_counts"]

    # one-hop successors
    for s in prior.keys():
        for nxt in successors.get(s, {}).keys():
            candidates.add(nxt)

    # one-hop predecessors
    for (a, b), _ in transition_counts.items():
        if b in prior:
            candidates.add(a)

    # common global states as safety net
    for s, _ in model["state_counts"].most_common(global_topk):
        candidates.add(s)

    # rank candidates: prior states first, then globally frequent states
    ranked = sorted(
        candidates,
        key=lambda s: (
            prior.get(s, 0.0),
            model["state_counts"].get(s, 0),
        ),
        reverse=True,
    )

    return ranked[:max_candidates]

def update_posterior(
    prior: Dict[Tuple[int, int], float],
    released_z: np.ndarray,
    epsilon: float,
    grid_size: float,
    aux: Dict,
    model: Dict,
    locset: Optional[List[Tuple[int, int]]] = None,
    topk=None,
    max_candidates: int = 2000,
) -> Dict[Tuple[int, int], float]:
    candidate_states = build_posterior_candidate_states(
        prior=prior,
        model=model,
        locset=locset,
        max_candidates=max_candidates,
    )

    if not candidate_states:
        return dict(prior)

    fallback = 1e-12
    prior_vals = np.array([prior.get(s, fallback) for s in candidate_states], dtype=float)
    prior_vals /= prior_vals.sum()

    centers = np.array([cell_center(s, grid_size) for s in candidate_states], dtype=float)

    T = aux["T"]
    hs_KI = aux["hs_KI"]

    # log prior
    log_prior = np.log(prior_vals + 1e-300)

    # log emission: proportional to exp(-epsilon * norm)
    log_emit = np.empty(len(candidate_states), dtype=float)
    for i, c in enumerate(centers):
        diff = T @ (released_z - c)
        norm_val = minkowski_norm_from_halfspaces(diff, hs_KI)
        log_emit[i] = -epsilon * norm_val

    log_post = log_prior + log_emit

    # stable normalize with log-sum-exp trick
    m = float(np.max(log_post))
    post_vals = np.exp(log_post - m)
    total = float(post_vals.sum())
    if total <= 0 or not np.isfinite(total):
        return dict(prior)

    post_vals /= total

    out = {
        candidate_states[i]: float(post_vals[i])
        for i in range(len(candidate_states))
        if post_vals[i] > 1e-15
    }

    s = sum(out.values())
    if s > 0:
        out = {k: v / s for k, v in out.items()}
    return out


# ============================================================
# FULL XIAO-STYLE PRIVATIZATION
# ============================================================
def privatize_trip_pim_xiao_style(
    df: pd.DataFrame,
    model: Dict,
    epsilon: float,
    delta_locset: float,
    grid_size: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = preprocess_trip_for_privatization(df, grid_size)
    n = len(df)
    if n == 0:
        return df, {"avg_locset_size": 0.0, "drift_ratio": 0.0, "points_after_preprocess": 0.0}

    true_states = trip_to_states(df, grid_size)

    prior = dict(model["global_prior"])
    if not prior:
        prior = {true_states[0]: 1.0}

    noisy_points = []
    locset_sizes = []
    drifts = 0

    for t in range(n):
        if t > 0:
            prior = propagate_prior_sparse(prior, model)
            if not prior:
                prior = dict(model["global_prior"])
                if not prior:
                    prior = {true_states[t - 1]: 1.0}

        locset = build_delta_location_set(
            prior=prior,
            delta=delta_locset,
            min_size=MIN_LOCSET_SIZE,
            max_size=MAX_CANDIDATES_PER_STEP,
        )

        true_state = true_states[t]
        surrogate = nearest_surrogate(true_state, locset, grid_size)
        if surrogate != true_state:
            drifts += 1

        z, aux = pim_release(
            surrogate_state=surrogate,
            locset=locset,
            epsilon=epsilon,
            grid_size=grid_size,
            rng=rng,
        )
        noisy_points.append(z)
        locset_sizes.append(len(locset))

        prior = update_posterior(
    prior=prior,
    released_z=z,
    epsilon=epsilon,
    grid_size=grid_size,
    aux=aux,
    model=model,
    locset=locset,
    topk=POSTERIOR_TOPK,
    max_candidates=300,
)

    out = df.copy()
    noisy_points = np.array(noisy_points)
    out["lat"] = noisy_points[:, 0]
    out["lon"] = noisy_points[:, 1]

    meta = {
        "avg_locset_size": float(np.mean(locset_sizes)),
        "drift_ratio": float(drifts / max(n, 1)),
        "points_after_preprocess": float(n),
    }
    return out, meta


def privatize_all_trips_pim(
    trips: List[Dict],
    model: Dict,
    epsilon: float,
    delta_locset: float,
    grid_size: float,
    seed: int = 42,
) -> Tuple[List[Dict], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    noisy_trips = []
    meta_rows = []

    for idx, trip in enumerate(trips, start=1):
        print(f"Privatizing trip {idx}/{len(trips)}: {trip['user_id']} / {trip['traj_id']}")
        noisy_df, meta = privatize_trip_pim_xiao_style(
            df=trip["df"],
            model=model,
            epsilon=epsilon,
            delta_locset=delta_locset,
            grid_size=grid_size,
            rng=rng,
        )
        noisy_trips.append({
            "user_id": trip["user_id"],
            "traj_id": trip["traj_id"],
            "df": noisy_df,
        })
        meta_rows.append({
            "user_id": trip["user_id"],
            "traj_id": trip["traj_id"],
            **meta,
        })

    return noisy_trips, pd.DataFrame(meta_rows)


# ============================================================
# FEATURE EXTRACTION + UNSUPERVISED
# ============================================================
def extract_trip_features(df: pd.DataFrame, grid_size: float) -> Dict[str, float]:
    df = df.sort_values("timestamp").reset_index(drop=True)

    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    ts = df["timestamp"].astype("int64").to_numpy() / 1e9

    if len(df) < 2:
        return {}

    dists = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    delta_t = np.diff(ts)
    valid = delta_t > 0

    dists = dists[valid]
    delta_t = delta_t[valid]
    if len(dists) == 0:
        return {}

    speeds = dists / delta_t

# acceleration from consecutive speeds
    if len(speeds) >= 2:
        accels = np.diff(speeds)
        avg_accel = float(np.mean(accels))
        std_accel = float(np.std(accels))
        max_abs_accel = float(np.max(np.abs(accels)))
    else:
        avg_accel = 0.0
        std_accel = 0.0
        max_abs_accel = 0.0

    # rolling-smoothed speeds to reduce jitter
    if len(speeds) >= 3:
        smooth_speeds = pd.Series(speeds).rolling(window=3, min_periods=1).mean().to_numpy()
    else:
        smooth_speeds = speeds

    avg_smooth_speed = float(np.mean(smooth_speeds))
    std_smooth_speed = float(np.std(smooth_speeds))
    max_smooth_speed = float(np.max(smooth_speeds)) if len(smooth_speeds) > 0 else 0.0

    total_distance = float(np.sum(dists))
    total_duration = float(ts[-1] - ts[0])
    avg_speed = float(np.mean(speeds))
    max_speed = float(np.max(speeds))
    std_speed = float(np.std(speeds))
    stop_ratio = float(np.mean(speeds < 0.5))
    fast_ratio = float(np.mean(speeds > 20.0))

    states = collapse_consecutive(trip_to_states(df, grid_size))
    unique_state_ratio = len(set(states)) / max(len(states), 1)

    dx = lon[1:] - lon[:-1]
    dy = lat[1:] - lat[:-1]
    headings = np.arctan2(dy, dx)
    heading_changes = np.abs(np.diff(headings))
    heading_changes = np.where(heading_changes > np.pi, 2 * np.pi - heading_changes, heading_changes)

    avg_heading_change = float(np.mean(heading_changes)) if len(heading_changes) > 0 else 0.0
    std_heading_change = float(np.std(heading_changes)) if len(heading_changes) > 0 else 0.0

    bbox_lat = float(np.max(lat) - np.min(lat))
    bbox_lon = float(np.max(lon) - np.min(lon))

    return {
        "num_points": float(len(df)),
        "num_states": float(len(states)),
        "trip_duration_sec": total_duration,
        "trip_distance_m": total_distance,
        "avg_speed_mps": avg_speed,
        "max_speed_mps": max_speed,
        "std_speed_mps": std_speed,
        "stop_ratio": stop_ratio,
        "fast_ratio": fast_ratio,
        "unique_state_ratio": float(unique_state_ratio),
        "avg_heading_change": avg_heading_change,
        "std_heading_change": std_heading_change,
        "bbox_lat": bbox_lat,
        "bbox_lon": bbox_lon,
        "avg_accel": avg_accel,
        "std_accel": std_accel,
        "max_abs_accel": max_abs_accel,
        "avg_smooth_speed": avg_smooth_speed,
        "std_smooth_speed": std_smooth_speed,
        "max_smooth_speed": max_smooth_speed,
    }


def build_feature_dataframe(trips: List[Dict], grid_size: float, preprocess: bool = False) -> pd.DataFrame:
    rows = []
    for trip in trips:
        df = trip["df"]
        if preprocess:
            df = preprocess_trip_for_privatization(df, grid_size)

        feats = extract_trip_features(df, grid_size)
        if not feats:
            continue
        feats["user_id"] = trip["user_id"]
        feats["traj_id"] = trip["traj_id"]
        rows.append(feats)
    return pd.DataFrame(rows)


def fit_iforest_scores(feature_df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = ["user_id", "traj_id"]
    feat_cols = [c for c in feature_df.columns if c not in meta_cols]

    X = feature_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(Xs)
    scores = -iso.decision_function(Xs)

    out = feature_df[meta_cols].copy()
    out["unsup_score"] = scores
    return out


# ============================================================
# COMPARISON
# ============================================================
def overlap_at_k(df: pd.DataFrame, col1: str, col2: str, k: int) -> float:
    top1 = set(df.nlargest(k, col1)["trip_key"])
    top2 = set(df.nlargest(k, col2)["trip_key"])
    return len(top1 & top2) / max(k, 1)


def jaccard_at_k(df: pd.DataFrame, col1: str, col2: str, k: int) -> float:
    top1 = set(df.nlargest(k, col1)["trip_key"])
    top2 = set(df.nlargest(k, col2)["trip_key"])
    union = top1 | top2
    return 0.0 if not union else len(top1 & top2) / len(union)


def compare_raw_vs_noisy(raw_scores: pd.DataFrame, noisy_scores: pd.DataFrame) -> pd.DataFrame:
    merged = raw_scores.merge(noisy_scores, on=["user_id", "traj_id"], how="inner").copy()
    merged["trip_key"] = merged["user_id"] + "_" + merged["traj_id"]

    corr, _ = spearmanr(merged["unsup_score_raw"], merged["unsup_score_noisy"])
    mad = float(np.mean(np.abs(merged["unsup_score_raw"] - merged["unsup_score_noisy"])))

    print("\n=== Raw vs Noisy Unsupervised Comparison ===")
    print(f"Trips compared: {len(merged)}")
    print(f"spearman_rank_corr: {corr:.6f}")
    print(f"mean_abs_score_diff: {mad:.6f}")
    print(f"top_10_overlap: {overlap_at_k(merged, 'unsup_score_raw', 'unsup_score_noisy', 10):.6f}")
    print(f"top_20_overlap: {overlap_at_k(merged, 'unsup_score_raw', 'unsup_score_noisy', 20):.6f}")
    print(f"top_50_overlap: {overlap_at_k(merged, 'unsup_score_raw', 'unsup_score_noisy', 50):.6f}")
    print(f"jaccard_10: {jaccard_at_k(merged, 'unsup_score_raw', 'unsup_score_noisy', 10):.6f}")
    print(f"jaccard_20: {jaccard_at_k(merged, 'unsup_score_raw', 'unsup_score_noisy', 20):.6f}")
    print(f"jaccard_50: {jaccard_at_k(merged, 'unsup_score_raw', 'unsup_score_noisy', 50):.6f}")

    return merged

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


def precision_recall_percentile(merged: pd.DataFrame, percent: float = 5.0):
    """
    Percentile-based thresholding computed independently for raw and noisy scores.
    """
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

    print(f"\n=== Percentile-based ({percent}%) Precision/Recall ===")
    print(f"Raw threshold: {raw_thresh:.6f}")
    print(f"Noisy threshold: {noisy_thresh:.6f}")
    print(f"Raw anomalies: {len(raw_top)}")
    print(f"Noisy anomalies: {len(noisy_top)}")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1-score: {f1:.6f}")

    print("\n--- Top-k Overlaps ---")
    for k in [10, 20, 50, 100]:
        overlap, jaccard = overlap_at_k_sets(merged, k)
        print(f"top_{k}_overlap: {overlap:.6f} | jaccard_{k}: {jaccard:.6f}")

    return {
        "raw_threshold": raw_thresh,
        "noisy_threshold": noisy_thresh,
        "raw_anomalies": len(raw_top),
        "noisy_anomalies": len(noisy_top),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading GeoLife trips...")
    trips = load_geolife_trips(ROOT, max_users=MAX_USERS)
    print(f"Loaded trips: {len(trips)}")
    if len(trips) == 0:
        print("No trips found. Check ROOT.")
        return

    print("\nBuilding Markov model...")
    model = build_markov_model(trips, GRID_SIZE)
    print(f"Unique states: {len(model['all_states'])}")

    print("\nScoring raw trips...")
    raw_feat = build_feature_dataframe(trips, GRID_SIZE, preprocess=True)
    raw_scores = fit_iforest_scores(raw_feat).rename(columns={"unsup_score": "unsup_score_raw"})

    print("\nApplying faster Xiao-style PIM privatization...")
    noisy_trips, meta_df = privatize_all_trips_pim(
        trips=trips,
        model=model,
        epsilon=EPSILON,
        delta_locset=DELTA_LOCSET,
        grid_size=GRID_SIZE,
        seed=RANDOM_STATE,
    )

    if not meta_df.empty:
        print("\nMechanism stats:")
        print(f"avg_locset_size: {meta_df['avg_locset_size'].mean():.4f}")
        print(f"avg_drift_ratio: {meta_df['drift_ratio'].mean():.4f}")
        print(f"avg_points_after_preprocess: {meta_df['points_after_preprocess'].mean():.2f}")
        print(f"PIM cache size: {len(PIM_CACHE)}")

    print("\nScoring noisy trips...")
    noisy_feat = build_feature_dataframe(noisy_trips, GRID_SIZE, preprocess=False)
    noisy_scores = fit_iforest_scores(noisy_feat).rename(columns={"unsup_score": "unsup_score_noisy"})

    merged = compare_raw_vs_noisy(raw_scores, noisy_scores)
    eval_results = precision_recall_percentile(merged, percent=5.0)

    pd.DataFrame([eval_results]).to_csv("geolife_eval_summary.csv", index=False)
    


    merged.to_csv("geolife_raw_vs_xiao_pim_fast.csv", index=False)
    raw_feat.to_csv("geolife_raw_features.csv", index=False)
    noisy_feat.to_csv("geolife_xiao_pim_fast_features.csv", index=False)
    meta_df.to_csv("geolife_xiao_pim_fast_meta.csv", index=False)

    print("\nSaved:")
    print("  geolife_raw_vs_xiao_pim_fast.csv")
    print("  geolife_eval_summary.csv")
  #  print("  geolife_raw_features.csv")
   # print("  geolife_xiao_pim_fast_features.csv")
    #print("  geolife_xiao_pim_fast_meta.csv")


if __name__ == "__main__":
    main()
