import numpy as np
from typing import Dict, Any

class ContentScheduler:
    def __init__(self, threshold: float = 0.15):
        """
        Initializes the scheduler.
        :param threshold: The minimum percentage increase required to justify scheduling for later.
        """
        self.threshold = threshold

    def decide_schedule(self, recommendation: Dict[str, Any], current_timestamp: int) -> Dict[str, Any]:
        """
        Issue 9, 11: Logic to decide between immediate posting and future scheduling.
        """
        # 1. Identify current context
        current_hour = int(current_timestamp) % 24
        rec_hour = recommendation["time_slot"]
        
        # 2. Extract scores (assuming the recommender passed the full score vector for comparison)
        # Note: In a production loop, we compare the score of the 'current_hour' arm
        # vs the score of the 'rec_hour' arm.
        rec_score = recommendation.get("expected_score", 0.0)
        
        # 3. Deterministic Decision Path (Issue 11)
        if current_hour == rec_hour:
            recommendation["decision"] = "POST_NOW"
            return recommendation

        # 4. Threshold Logic (Issue 9)
        # If the recommended time slot is significantly better than 'now', we schedule.
        # Otherwise, we prioritize immediate delivery to maintain content freshness.
        # Note: For this logic to work, the recommender must provide the score of the current hour.
        current_hour_score = recommendation.get("current_hour_score", 0.0)
        
        # Check if the gain from waiting exceeds the threshold (e.g., 15% better)
        if rec_score > (current_hour_score * (1 + self.threshold)):
            recommendation["decision"] = "SCHEDULE"
        else:
            # If the gain is marginal, post now to avoid unnecessary latency
            recommendation["decision"] = "POST_NOW"
            # If we switch to POST_NOW, we update the time_slot to match reality
            recommendation["time_slot"] = current_hour

        return recommendation

    def validate_burst_capacity(self, queue_size: int, limit: int = 100):
        """
        Issue 13: Monitor queue size to handle burst submissions.
        """
        if queue_size > limit:
            # Implement a lightweight strategy like skipping non-essential 
            # exploration when system load is high.
            return "CONSERVATIVE"
        return "OPTIMAL"