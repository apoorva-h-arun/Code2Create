import numpy as np
import pandas as pd
from typing import Dict, Tuple

class ContentBandit:
    def __init__(self, n_platforms: int = 2, n_hours: int = 24, seed: int = 42):
        """
        Initializes for 48 arms (2 platforms * 24 hours).
        """
        self.n_arms = n_platforms * n_hours
        self.rng = np.random.default_rng(seed)
        
        # Bayesian parameters: alpha (successes), beta (failures)
        self.alpha = np.ones(self.n_arms) 
        self.beta = np.ones(self.n_arms)

    def _get_arm_index(self, platform: str, hour: int) -> int:
        """Maps platform and hour to index 0-47."""
        p_idx = 0 if platform.lower() == 'youtube' else 1
        return (p_idx * 24) + hour

    def _get_arm_details(self, index: int) -> Tuple[str, int]:
        """Maps index 0-47 back to Platform and Hour."""
        platform = 'YouTube' if index < 24 else 'Instagram'
        hour = index % 24
        return platform, hour

    def warm_start(self, historical_df: pd.DataFrame):
        """
        Issue 3: Initialize using Historical Data (image_a1a7d7.png).
        """
        for _, row in historical_df.iterrows():
            # Use data from image_a1a7d7.png
            idx = self._get_arm_index(row['platform'], row['time_slot'])
            # avg_engagement is the reward signal
            self.alpha[idx] += row['avg_engagement']
            self.beta[idx] += 1 

    def recommend(self, 
                  time_sensitivity: str, 
                  platform_activity_scores: np.ndarray) -> Dict:
        """
        Issue 5, 8, 11: Optimal decision based on current belief and activity scores.
        """
        # activity_score from image_a1a7b4.png
        # platform_activity_scores should be a flat array of length 48
        
        if time_sensitivity.lower() == 'high':
            # Deterministic Exploitation: Mean * Activity Score
            expected_rewards = (self.alpha / (self.alpha + self.beta)) * platform_activity_scores
            best_arm = int(np.argmax(expected_rewards))
        else:
            # Exploration: Thompson Sampling * Activity Score
            samples = self.rng.beta(self.alpha, self.beta) * platform_activity_scores
            best_arm = int(np.argmax(samples))

        platform, hour = self._get_arm_details(best_arm)
        
        return {
            "platform": platform,
            "recommended_hour": hour,
            "decision": "DETERMINISTIC" # Issue 11
        }