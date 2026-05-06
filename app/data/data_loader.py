"""
Data Loading Module
Covers Issues 1-4: Content, Platform Activity, Historical Engagement, Creator Data

Missing value strategy (all use averages, never zeros):
  - M_hist cells    → column mean (same arm across all creators)
  - activity scores → per-platform mean for missing slots
  - creator profile → mean base_engagement / mean cooldown_hours across all creators
  - new creator row → mean M_score row for that content_type
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class ContentItem:
    content_id: int
    creator_id: int
    content_type: str          # "SHORT" | "LONG"
    created_timestamp: int     # hour 0-23
    time_sensitivity: str      # "Low" | "Medium" | "High"


@dataclass
class CreatorProfile:
    creator_id: int
    base_engagement: float
    cooldown_hours: int


@dataclass
class DataStore:
    # Issue 1
    content: dict = field(default_factory=dict)

    # Issue 2 — (platform, time_slot) → score
    activity: dict = field(default_factory=dict)

    # Issue 3 — engagement matrix
    M_hist: Optional[np.ndarray] = None
    row_index: dict = field(default_factory=dict)   # (creator_id, ctype) → row
    arm_index: dict = field(default_factory=dict)   # (platform, slot)   → col
    arms: list = field(default_factory=list)

    # Issue 4
    creators: dict = field(default_factory=dict)

    # Derived
    M_score: Optional[np.ndarray] = None


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

VALID_CONTENT_TYPES    = {"SHORT", "LONG"}
VALID_TIME_SENSITIVITY = {"Low", "Medium", "High"}
VALID_PLATFORMS        = {"Instagram", "YouTube"}


# ─────────────────────────────────────────────
# Issue 1: Load Content Submissions
# ─────────────────────────────────────────────

def load_content(path: str, store: DataStore) -> None:
    logger.info(f"[Issue 1] Loading content from {path}")
    df = pd.read_csv(path)
    _check_columns(df, {"content_id","creator_id","content_type",
                        "created_timestamp","time_sensitivity"}, path)

    loaded, skipped = 0, 0
    for _, row in df.iterrows():
        try:
            ct   = str(row["content_type"]).strip().upper()
            ts   = str(row["time_sensitivity"]).strip().capitalize()
            slot = int(row["created_timestamp"])

            if ct not in VALID_CONTENT_TYPES:
                raise ValueError(f"Invalid content_type: {ct}")
            if ts not in VALID_TIME_SENSITIVITY:
                raise ValueError(f"Invalid time_sensitivity: {ts}")
            if not (0 <= slot <= 23):
                raise ValueError(f"created_timestamp out of range: {slot}")

            item = ContentItem(
                content_id        = int(row["content_id"]),
                creator_id        = int(row["creator_id"]),
                content_type      = ct,
                created_timestamp = slot,
                time_sensitivity  = ts,
            )
            store.content[item.content_id] = item
            loaded += 1
        except Exception as e:
            logger.warning(f"  Skipping content row {row.get('content_id','?')}: {e}")
            skipped += 1

    logger.info(f"  Loaded {loaded} content items, skipped {skipped}")


# ─────────────────────────────────────────────
# Issue 2: Load Platform Activity Data
# ─────────────────────────────────────────────

def load_platform_activity(path: str, store: DataStore) -> None:
    logger.info(f"[Issue 2] Loading platform activity from {path}")
    df = pd.read_csv(path)
    _check_columns(df, {"platform","time_slot","activity_score"}, path)

    platforms  = sorted(VALID_PLATFORMS)
    store.arms = [(p, t) for p in platforms for t in range(24)]  # 48 arms
    store.arm_index = {arm: i for i, arm in enumerate(store.arms)}

    loaded, skipped = 0, 0
    for _, row in df.iterrows():
        try:
            platform = str(row["platform"]).strip()
            slot     = int(row["time_slot"])
            score    = float(row["activity_score"])

            if platform not in VALID_PLATFORMS:
                raise ValueError(f"Unknown platform: {platform}")
            if not (0 <= slot <= 23):
                raise ValueError(f"time_slot out of range: {slot}")

            store.activity[(platform, slot)] = score
            loaded += 1
        except Exception as e:
            logger.warning(f"  Skipping activity row: {e}")
            skipped += 1

    # ── Fill missing (platform, slot) combos with platform average ──────────
    for p in VALID_PLATFORMS:
        existing = [v for (pl, _), v in store.activity.items() if pl == p]
        avg = float(np.mean(existing)) if existing else 0.6
        for t in range(24):
            if (p, t) not in store.activity:
                store.activity[(p, t)] = avg
                logger.warning(f"  [Impute] Missing activity ({p}, {t}) → {avg:.3f}")

    logger.info(f"  Loaded {loaded} activity records, skipped {skipped}")
    logger.info(f"  Arms defined: {len(store.arms)}")


# ─────────────────────────────────────────────
# Issue 3: Load Historical Engagement → Matrix
# ─────────────────────────────────────────────

def load_historical_engagement(path: str, store: DataStore) -> None:
    logger.info(f"[Issue 3] Loading historical engagement from {path}")

    if not store.arm_index:
        raise RuntimeError("Load platform activity before historical engagement")

    df = pd.read_csv(path)
    _check_columns(df, {"creator_id","platform","content_type",
                        "time_slot","avg_engagement"}, path)

    contexts = sorted(
        df[["creator_id","content_type"]].drop_duplicates()
          .apply(lambda r: (int(r["creator_id"]),
                            str(r["content_type"]).strip().upper()), axis=1)
          .tolist()
    )
    store.row_index = {ctx: i for i, ctx in enumerate(contexts)}

    n_rows = len(contexts)
    n_cols = len(store.arms)
    M = np.zeros((n_rows, n_cols), dtype=np.float32)

    loaded, skipped = 0, 0
    for _, row in df.iterrows():
        try:
            ctx = (int(row["creator_id"]),
                   str(row["content_type"]).strip().upper())
            arm = (str(row["platform"]).strip(), int(row["time_slot"]))

            if ctx not in store.row_index:
                raise ValueError(f"Unknown context: {ctx}")
            if arm not in store.arm_index:
                raise ValueError(f"Unknown arm: {arm}")

            M[store.row_index[ctx], store.arm_index[arm]] = float(row["avg_engagement"])
            loaded += 1
        except Exception as e:
            logger.warning(f"  Skipping engagement row: {e}")
            skipped += 1

    # ── Impute missing cells with column mean ────────────────────────────────
    # column = one (platform, time_slot) arm across all creators
    # if a creator has no data for that arm → use average of others who do
    global_mean = float(M[M > 0].mean()) if (M > 0).any() else 0.5
    imputed = 0
    for col in range(n_cols):
        col_vals = M[:, col]
        nonzero  = col_vals[col_vals > 0]
        fill     = float(nonzero.mean()) if len(nonzero) > 0 else global_mean
        mask     = col_vals == 0
        if mask.any():
            M[mask, col] = fill
            imputed += mask.sum()

    store.M_hist = M
    logger.info(f"  Matrix shape: {M.shape} | Loaded {loaded}, skipped {skipped}, imputed {imputed} cells")

    _build_score_matrix(store)


def _build_score_matrix(store: DataStore) -> None:
    """M_score = M_hist * activity_vector  (shape: n_contexts × 48)"""
    act_vec = np.array(
        [store.activity[arm] for arm in store.arms],   # all slots guaranteed filled
        dtype=np.float32
    )
    store.M_score = store.M_hist * act_vec
    logger.info(f"[Matrix] M_score built {store.M_score.shape} | "
                f"range [{store.M_score.min():.3f}, {store.M_score.max():.3f}]")


# ─────────────────────────────────────────────
# Issue 4: Load Creator Base Engagement Data
# ─────────────────────────────────────────────

def load_creators(path: str, store: DataStore) -> None:
    logger.info(f"[Issue 4] Loading creators from {path}")
    df = pd.read_csv(path)
    _check_columns(df, {"creator_id","base_engagement","cooldown_hours"}, path)

    loaded, skipped = 0, 0
    for _, row in df.iterrows():
        try:
            base = float(row["base_engagement"])
            cool = int(row["cooldown_hours"])

            if not (0.0 < base <= 5.0):
                raise ValueError(f"base_engagement out of range: {base}")
            if not (0 <= cool <= 24):
                raise ValueError(f"cooldown_hours out of range: {cool}")

            store.creators[int(row["creator_id"])] = CreatorProfile(
                creator_id      = int(row["creator_id"]),
                base_engagement = base,
                cooldown_hours  = cool,
            )
            loaded += 1
        except Exception as e:
            logger.warning(f"  Skipping creator {row.get('creator_id','?')}: {e}")
            skipped += 1

    # ── Compute averages for unknown creator fallback ────────────────────────
    all_base = [c.base_engagement for c in store.creators.values()]
    all_cool = [c.cooldown_hours  for c in store.creators.values()]
    store._avg_base_engagement = float(np.mean(all_base)) if all_base else 1.0
    store._avg_cooldown_hours  = int(round(np.mean(all_cool))) if all_cool else 4

    logger.info(f"  Loaded {loaded} creators, skipped {skipped}")
    logger.info(f"  Fallback averages → base={store._avg_base_engagement:.3f}, "
                f"cooldown={store._avg_cooldown_hours}h")


def get_creator(creator_id: int, store: DataStore) -> CreatorProfile:
    """
    Safe lookup. Unknown creators get average base_engagement and cooldown
    computed from all loaded creators — not a hardcoded guess.
    """
    if creator_id in store.creators:
        return store.creators[creator_id]

    logger.warning(f"  Unknown creator {creator_id} → using dataset averages")
    return CreatorProfile(
        creator_id      = creator_id,
        base_engagement = getattr(store, "_avg_base_engagement", 1.0),
        cooldown_hours  = getattr(store, "_avg_cooldown_hours",  4),
    )


def get_score_row(creator_id: int, content_type: str, store: DataStore) -> np.ndarray:
    """
    Return the M_score row for a given (creator, content_type).
    Unknown creators get the mean row across all creators of that content_type
    — not zeros, not a hardcoded fallback.
    """
    ctx = (creator_id, content_type)
    if ctx in store.row_index:
        return store.M_score[store.row_index[ctx]].copy()

    # Cold start: average over all known rows with same content_type
    logger.warning(f"  No history for {ctx} → using content_type mean row")
    same_type_rows = [
        store.M_score[i]
        for (cid, ct), i in store.row_index.items()
        if ct == content_type
    ]
    if same_type_rows:
        return np.mean(same_type_rows, axis=0).astype(np.float32)

    # Last resort: global mean row
    return store.M_score.mean(axis=0).astype(np.float32)


# ─────────────────────────────────────────────
# Master Loader
# ─────────────────────────────────────────────

def load_all(
    content_path:    str = "data/raw/content.csv",
    activity_path:   str = "data/raw/platform_activity.csv",
    engagement_path: str = "data/raw/historical_engagement.csv",
    creators_path:   str = "data/raw/creators.csv",
) -> DataStore:
    store = DataStore()
    load_platform_activity(activity_path, store)       # must be first
    load_historical_engagement(engagement_path, store) # needs arm_index
    load_creators(creators_path, store)
    load_content(content_path, store)
    _log_summary(store)
    return store


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _check_columns(df: pd.DataFrame, required: set, path: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")


def _log_summary(store: DataStore) -> None:
    logger.info("\n=== DataStore Summary ===")
    logger.info(f"  Content items : {len(store.content)}")
    logger.info(f"  Creators      : {len(store.creators)}")
    logger.info(f"  Activity slots: {len(store.activity)}")
    logger.info(f"  Arms (cols)   : {len(store.arms)}")
    logger.info(f"  Contexts(rows): {len(store.row_index)}")
    logger.info(f"  M_hist        : {store.M_hist.shape}")
    logger.info(f"  M_score       : {store.M_score.shape}")
    logger.info("=========================\n")


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("data/raw", exist_ok=True)

    # Generate minimal test CSVs
    import random
    random.seed(42)
    pd.DataFrame({
        "creator_id": range(1, 4),
        "base_engagement": [1.11, 0.62, 0.82],
        "cooldown_hours":  [4, 6, 6]
    }).to_csv("data/raw/creators.csv", index=False)

    rows = []
    for p in ["Instagram","YouTube"]:
        for t in range(24):
            score = 0.4 + (0.6 if (p=="Instagram" and 18<=t<=22)
                           or (p=="YouTube" and 20<=t<=23) else 0.2)
            rows.append([p, t, round(score, 2)])
    pd.DataFrame(rows, columns=["platform","time_slot","activity_score"]
                 ).to_csv("data/raw/platform_activity.csv", index=False)

    hist = []
    for c in range(1, 4):
        for p in ["Instagram","YouTube"]:
            for ct in ["SHORT","LONG"]:
                for t in range(24):
                    base = random.uniform(0.3, 1.0)
                    base *= 1.25 if (p=="Instagram" and ct=="SHORT") \
                                 or (p=="YouTube"   and ct=="LONG")  else 0.85
                    hist.append([c, p, ct, t, round(base, 3)])
    pd.DataFrame(hist, columns=["creator_id","platform","content_type",
                                "time_slot","avg_engagement"]
                 ).to_csv("data/raw/historical_engagement.csv", index=False)

    pd.DataFrame({
        "content_id":        [1, 2, 3],
        "creator_id":        [1, 99, 2],   # creator 99 is unknown → tests cold start
        "content_type":      ["SHORT","LONG","SHORT"],
        "created_timestamp": [6, 22, 19],
        "time_sensitivity":  ["Medium","High","Low"]
    }).to_csv("data/raw/content.csv", index=False)

    store = load_all()

    print("\n-- Spot checks --")
    for cid, ct in [(1,"SHORT"), (99,"LONG")]:
        row  = get_score_row(cid, ct, store)
        best = int(np.argmax(row))
        p, t = store.arms[best]
        prof = get_creator(cid, store)
        print(f"  creator={cid} {ct}: best arm=({p}, slot {t})  "
              f"score={row[best]:.4f}  base={prof.base_engagement:.2f}")