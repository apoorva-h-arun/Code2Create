"""
Script to run bandit with data loader
Demonstrates full workflow: load data → initialize bandit → warm start → make recommendations
"""

import numpy as np
import pandas as pd
from app.core.bandit import ContentBandit
from app.data.data_loader import (
    DataStore, 
    load_content, 
    load_platform_activity, 
    load_historical_engagement, 
    load_creators,
    get_score_row
)


def main():
    """Main execution"""
    print("=" * 70)
    print("BANDIT + DATA LOADER WORKFLOW")
    print("=" * 70)
    
    # 1. Initialize data store
    store = DataStore()
    print("\n[1] Initializing DataStore...")
    
    # 2. Load all data
    print("[2] Loading data files...")
    load_content("data/raw/content.csv", store)
    load_platform_activity("data/raw/platform_activity.csv", store)
    load_historical_engagement("data/raw/historical_engagement.csv", store)
    load_creators("data/raw/creators.csv", store)
    
    print(f"\n    ✓ Content items: {len(store.content)}")
    print(f"    ✓ Arms (platform × hour): {len(store.arms)}")
    print(f"    ✓ Contexts (creator × content_type): {len(store.row_index)}")
    print(f"    ✓ Creators: {len(store.creators)}")
    print(f"    ✓ M_hist shape: {store.M_hist.shape}")
    print(f"    ✓ M_score shape: {store.M_score.shape}")
    
    # 3. Initialize bandit
    print("\n[3] Initializing ContentBandit...")
    bandit = ContentBandit(n_platforms=2, n_hours=24, seed=42)
    print(f"    ✓ Number of arms: {bandit.n_arms}")
    
    # 4. Warm start with historical data
    print("\n[4] Warm-starting bandit with historical engagement...")
    # Create warm-start dataframe from historical data
    warmstart_data = []
    for (platform, slot), score in store.activity.items():
        warmstart_data.append({
            'platform': platform,
            'time_slot': slot,
            'avg_engagement': store.M_hist.mean()  # Use average engagement
        })
    
    if warmstart_data:
        warmstart_df = pd.DataFrame(warmstart_data)
        bandit.warm_start(warmstart_df)
        print(f"    ✓ Warm-started with {len(warmstart_df)} arm observations")
        print(f"    ✓ Alpha range: [{bandit.alpha.min():.3f}, {bandit.alpha.max():.3f}]")
        print(f"    ✓ Beta range: [{bandit.beta.min():.3f}, {bandit.beta.max():.3f}]")
    
    # 5. Make recommendations for different scenarios
    print("\n[5] Making recommendations...")
    
    # Prepare platform activity scores
    activity_scores = np.array(
        [store.activity[arm] for arm in store.arms],
        dtype=np.float32
    )
    
    scenarios = [
        ("High", "High time sensitivity (exploitation)"),
        ("Low", "Low time sensitivity (exploration)"),
        ("Medium", "Medium time sensitivity"),
    ]
    
    for time_sensitivity, description in scenarios:
        print(f"\n    Scenario: {description}")
        rec = bandit.recommend(time_sensitivity, activity_scores)
        print(f"      Platform: {rec['platform']}")
        print(f"      Hour: {rec['recommended_hour']:02d}:00")
        print(f"      Decision: {rec['decision']}")
    
    print("\n" + "=" * 70)
    print("✓ EXECUTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
