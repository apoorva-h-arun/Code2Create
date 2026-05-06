"""
Quick reference guide and interactive demo for bandit + data loader
"""

import numpy as np
import pandas as pd
from app.core.bandit import ContentBandit
from app.data.data_loader import DataStore, load_content, load_platform_activity, load_historical_engagement, load_creators


def demo_basic_bandit():
    """Minimal example: bandit without data loader"""
    print("\n" + "="*70)
    print("DEMO 1: Basic Bandit (No Data Loader)")
    print("="*70)
    
    # Initialize
    bandit = ContentBandit(n_platforms=2, n_hours=24, seed=42)
    print(f"✓ Created bandit with {bandit.n_arms} arms")
    
    # Warm start with synthetic data
    synthetic_warmstart = pd.DataFrame({
        'platform': ['YouTube'] * 12 + ['Instagram'] * 12,
        'time_slot': list(range(12)) + list(range(12)),
        'avg_engagement': np.random.rand(24)
    })
    
    bandit.warm_start(synthetic_warmstart)
    print(f"✓ Warm-started with {len(synthetic_warmstart)} observations")
    
    # Get recommendation
    activity_scores = np.ones(48) * 0.5
    rec = bandit.recommend("High", activity_scores)
    print(f"✓ Recommendation: {rec['platform']} at {rec['recommended_hour']:02d}:00")


def demo_bandit_with_loader():
    """Full example: bandit + data loader"""
    print("\n" + "="*70)
    print("DEMO 2: Bandit with Data Loader")
    print("="*70)
    
    try:
        # Initialize store and load data
        store = DataStore()
        print("Loading data...")
        load_content("data/raw/content.csv", store)
        load_platform_activity("data/raw/platform_activity.csv", store)
        load_historical_engagement("data/raw/historical_engagement.csv", store)
        load_creators("data/raw/creators.csv", store)
        
        print(f"✓ Loaded {len(store.content)} content items")
        print(f"✓ Loaded {len(store.creators)} creators")
        print(f"✓ Engagement matrix: {store.M_hist.shape}")
        
        # Initialize and warm start bandit
        bandit = ContentBandit(n_platforms=2, n_hours=24, seed=42)
        
        warmstart_df = pd.DataFrame({
            'platform': [arm[0] for arm in store.arms],
            'time_slot': [arm[1] for arm in store.arms],
            'avg_engagement': store.M_hist.mean(axis=0)  # Average across creators
        })
        
        bandit.warm_start(warmstart_df)
        print(f"✓ Warm-started bandit with {len(store.arms)} arms")
        
        # Get recommendations
        activity_scores = np.array([store.activity[arm] for arm in store.arms])
        
        for sensitivity in ["High", "Low"]:
            rec = bandit.recommend(sensitivity, activity_scores)
            print(f"✓ [{sensitivity}] → {rec['platform']} at {rec['recommended_hour']:02d}:00")
    
    except FileNotFoundError as e:
        print(f"⚠ Data files not found: {e}")
        print("  (Make sure to run from project root with data/raw/ directory)")


def api_reference():
    """API reference guide"""
    print("\n" + "="*70)
    print("API REFERENCE")
    print("="*70)
    
    print("""
CONTENTBANDIT API:
─────────────────

1. Initialize:
   bandit = ContentBandit(n_platforms=2, n_hours=24, seed=42)

2. Warm Start (update beliefs with historical data):
   historical_df = pd.DataFrame({
       'platform': ['YouTube', 'Instagram', ...],
       'time_slot': [5, 10, ...],
       'avg_engagement': [0.8, 0.6, ...]
   })
   bandit.warm_start(historical_df)

3. Get Recommendation:
   activity_scores = np.ones(48) * 0.5  # 48 = 2 platforms × 24 hours
   recommendation = bandit.recommend(
       time_sensitivity="High",  # "High", "Low", or "Medium"
       platform_activity_scores=activity_scores
   )
   # Returns:
   # {
   #     "platform": "YouTube",
   #     "recommended_hour": 14,
   #     "decision": "DETERMINISTIC"
   # }

DATA LOADER API:
────────────────

store = DataStore()

# Load data sequentially (order matters!)
load_content("path/to/content.csv", store)
load_platform_activity("path/to/platform_activity.csv", store)
load_historical_engagement("path/to/historical_engagement.csv", store)
load_creators("path/to/creators.csv", store)

# Access loaded data:
print(store.content)           # Dict of ContentItem
print(store.activity)          # Dict of (platform, time_slot) → score
print(store.M_hist)            # Matrix: (creator, content_type) × arm
print(store.M_score)           # M_hist × activity scores
print(store.creators)          # Dict of CreatorProfile
print(store.arms)              # List of (platform, time_slot) tuples
print(store.row_index)         # Dict mapping (creator_id, content_type) → row
print(store.arm_index)         # Dict mapping (platform, time_slot) → col
    """)


def performance_notes():
    """Performance and usage notes"""
    print("\n" + "="*70)
    print("PERFORMANCE & USAGE NOTES")
    print("="*70)
    
    print("""
TIME SENSITIVITY MODES:
─────────────────────

HIGH Sensitivity (Exploitation):
  • Uses deterministic: mean * activity_score
  • Always picks same best arm in same state
  • Best for: Real-time bidding, critical decisions
  
MEDIUM/LOW Sensitivity (Exploration):
  • Uses Thompson Sampling: sample from Beta(α,β) × activity_score
  • Picks randomly based on uncertainty
  • Best for: Learning, A/B testing

WARM START BEST PRACTICES:
──────────────────────────
  • Always warm start before first recommendation
  • Use historical engagement data if available
  • Updates alpha = successes, beta = failures
  • Multiple warm starts accumulate beliefs

ACTIVITY SCORES:
────────────────
  • Array of 48 floats (2 platforms × 24 hours)
  • Order: YouTube [0-23], Instagram [24-47]
  • Range typically [0, 1] but can be any positive values
  • Missing slots are imputed with platform average

DATA LOADING ORDER:
──────────────────
  1. load_content()                  (Issue 1)
  2. load_platform_activity()        (Issue 2) 
  3. load_historical_engagement()    (Issue 3)
  4. load_creators()                 (Issue 4)
  
Loading out of order will raise RuntimeError!

MISSING DATA STRATEGY:
─────────────────────
  • All missing values use averages (never zeros)
  • Activity: per-platform mean for missing slots
  • Engagement: column mean (same arm across creators)
  • Creators: dataset mean base_engagement & cooldown_hours
    """)


if __name__ == "__main__":
    demo_basic_bandit()
    demo_bandit_with_loader()
    api_reference()
    performance_notes()
    
    print("\n" + "="*70)
    print("For complete tests: pytest test_bandit.py -v")
    print("For full workflow:  python run_bandit.py")
    print("="*70)
