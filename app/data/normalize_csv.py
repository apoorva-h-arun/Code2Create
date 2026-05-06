import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.data.data_loader import (
    VALID_CONTENT_TYPES,
    VALID_PLATFORMS,
    VALID_TIME_SENSITIVITY,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PLATFORM_MAP = {
    'instagram': 'Instagram',
    'insta': 'Instagram',
    'youtube': 'YouTube',
    'yt': 'YouTube',
}

CONTENT_TYPE_MAP = {
    'short': 'SHORT',
    'long': 'LONG',
}

TIME_SENSITIVITY_MAP = {
    'low': 'Low',
    'medium': 'Medium',
    'high': 'High',
}


def _normalize_platform(value: str) -> str:
    if pd.isna(value):
        return None
    key = str(value).strip().lower()
    return PLATFORM_MAP.get(key, str(value).strip().title())


def _normalize_content_type(value: str) -> str:
    if pd.isna(value):
        return None
    return CONTENT_TYPE_MAP.get(str(value).strip().lower(), str(value).strip().upper())


def _normalize_time_sensitivity(value: str) -> str:
    if pd.isna(value):
        return None
    return TIME_SENSITIVITY_MAP.get(str(value).strip().lower(), str(value).strip().capitalize())


def _validate_range(value, min_value, max_value, fallback=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    if value < min_value or value > max_value:
        return fallback
    return value


def _safe_int(value, fallback=None):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def normalize_content(df: pd.DataFrame) -> pd.DataFrame:
    expected = {'content_id', 'creator_id', 'content_type', 'created_timestamp', 'time_sensitivity'}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in content CSV: {missing}")

    df = df.copy()
    df['content_type'] = df['content_type'].map(_normalize_content_type)
    df['time_sensitivity'] = df['time_sensitivity'].map(_normalize_time_sensitivity)
    df['created_timestamp'] = df['created_timestamp'].map(_safe_int)
    df['content_id'] = df['content_id'].map(_safe_int)
    df['creator_id'] = df['creator_id'].map(_safe_int)

    valid = (
        df['content_id'].notna() &
        df['creator_id'].notna() &
        df['content_type'].isin(VALID_CONTENT_TYPES) &
        df['time_sensitivity'].isin(VALID_TIME_SENSITIVITY) &
        df['created_timestamp'].between(0, 23)
    )
    invalid_rows = len(df) - int(valid.sum())
    if invalid_rows:
        logger.warning(f"Dropping {invalid_rows} invalid content rows")
    df = df.loc[valid].drop_duplicates(subset=['content_id'])
    df['created_timestamp'] = df['created_timestamp'].astype(int)
    df[['content_id', 'creator_id']] = df[['content_id', 'creator_id']].astype(int)
    return df


def normalize_platform_activity(df: pd.DataFrame) -> pd.DataFrame:
    expected = {'platform', 'time_slot', 'activity_score'}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in platform activity CSV: {missing}")

    df = df.copy()
    df['platform'] = df['platform'].map(_normalize_platform)
    df['time_slot'] = df['time_slot'].map(_safe_int)
    df['activity_score'] = pd.to_numeric(df['activity_score'], errors='coerce')

    df = df.loc[ df['platform'].isin(VALID_PLATFORMS) ]
    df = df.loc[ df['time_slot'].between(0, 23) ]
    df['activity_score'] = df['activity_score'].clip(0.0, 1.0)

    df = df.groupby(['platform', 'time_slot'], as_index=False)['activity_score'].mean()

    # Fill any missing platform/time_slot combinations with per-platform average
    full_index = pd.MultiIndex.from_product([sorted(VALID_PLATFORMS), range(24)], names=['platform', 'time_slot'])
    df = df.set_index(['platform', 'time_slot']).reindex(full_index).reset_index()
    df['activity_score'] = df.groupby('platform')['activity_score'].transform(lambda x: x.fillna(x.mean()))
    df['activity_score'] = df['activity_score'].fillna(0.5)
    return df


def normalize_historical_engagement(df: pd.DataFrame) -> pd.DataFrame:
    expected = {'creator_id', 'platform', 'content_type', 'time_slot', 'avg_engagement'}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in historical engagement CSV: {missing}")

    df = df.copy()
    df['platform'] = df['platform'].map(_normalize_platform)
    df['content_type'] = df['content_type'].map(_normalize_content_type)
    df['creator_id'] = df['creator_id'].map(_safe_int)
    df['time_slot'] = df['time_slot'].map(_safe_int)
    df['avg_engagement'] = pd.to_numeric(df['avg_engagement'], errors='coerce')

    valid = (
        df['creator_id'].notna() &
        df['platform'].isin(VALID_PLATFORMS) &
        df['content_type'].isin(VALID_CONTENT_TYPES) &
        df['time_slot'].between(0, 23) &
        df['avg_engagement'].notna()
    )
    if not valid.all():
        logger.warning(f"Dropping {len(df) - int(valid.sum())} invalid historical engagement rows")
    df = df.loc[valid]
    df = df.groupby(
        ['creator_id', 'platform', 'content_type', 'time_slot'],
        as_index=False
    )['avg_engagement'].mean()
    return df


def normalize_creators(df: pd.DataFrame) -> pd.DataFrame:
    expected = {'creator_id', 'base_engagement', 'cooldown_hours'}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in creators CSV: {missing}")

    df = df.copy()
    df['creator_id'] = df['creator_id'].map(_safe_int)
    df['base_engagement'] = pd.to_numeric(df['base_engagement'], errors='coerce')
    df['cooldown_hours'] = df['cooldown_hours'].map(_safe_int)

    valid = (
        df['creator_id'].notna() &
        df['base_engagement'].between(0.0, 10.0) &
        df['cooldown_hours'].between(0, 24)
    )
    if not valid.all():
        logger.warning(f"Dropping {len(df) - int(valid.sum())} invalid creator rows")
    df = df.loc[valid]
    df = df.groupby('creator_id', as_index=False).agg({
        'base_engagement': 'mean',
        'cooldown_hours': 'mean',
    })
    df['cooldown_hours'] = df['cooldown_hours'].round().astype(int)
    df['base_engagement'] = df['base_engagement'].astype(float)
    df['creator_id'] = df['creator_id'].astype(int)
    return df


def clean_all(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    content = pd.read_csv(input_dir / 'content.csv')
    content = normalize_content(content)
    content.to_csv(output_dir / 'content.csv', index=False)

    activity = pd.read_csv(input_dir / 'platform_activity.csv')
    activity = normalize_platform_activity(activity)
    activity.to_csv(output_dir / 'platform_activity.csv', index=False)

    engagement = pd.read_csv(input_dir / 'historical_engagement.csv')
    engagement = normalize_historical_engagement(engagement)
    engagement.to_csv(output_dir / 'historical_engagement.csv', index=False)

    creators = pd.read_csv(input_dir / 'creators.csv')
    creators = normalize_creators(creators)
    creators.to_csv(output_dir / 'creators.csv', index=False)

    logger.info(f"Cleaned CSV files written to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean and normalize raw CSV files.')
    parser.add_argument(
        '--input-dir', '-i',
        default='data/raw',
        help='Directory containing the raw CSV files.'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='data/cleaned',
        help='Directory to write cleaned CSV files.'
    )
    args = parser.parse_args()
    clean_all(Path(args.input_dir), Path(args.output_dir))
