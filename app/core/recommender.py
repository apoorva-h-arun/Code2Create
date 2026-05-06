import numpy as np
from datetime import datetime

class RecommendationEngine:
    def __init__(self, store, bandit):
        self.store = store  # Your data_loader instance
        self.bandit = bandit

    def get_final_scores(self, content_item):
        """
        Implements the 'How They All Chain Together' logic.
        """
        c_id = content_item['creator_id']
        c_type = content_item['content_type']
        
        # 1. Slice M_score by context (Issue 10)
        row_idx = self.store.row_index.get((c_id, c_type))
        if row_idx is None:
            # Fallback for unknown contexts (Issue 15)
            scores = self.store.M_score.mean(axis=0).copy()
        else:
            scores = self.store.M_score[row_idx].copy()

        # 2. Build fit_vec (Issue 6)
        fit_vec = np.ones(48, dtype=np.float32)
        if c_type == "SHORT":
            fit_vec[24:] = 0.92  # YT Penalty
        else:
            fit_vec[:24] = 0.80  # IG Penalty

        # 3. Apply Cooldown Mask (Issue 17)
        # Assuming store.get_cooldown_mask returns a (48,) bool array
        mask = self.store.get_cooldown_mask(c_id)
        
        # 4. Final calculation (Issue 5)
        # base_engagement is treated as a scalar from creator profile
        base_eng = self.store.creator_data.get(c_id, {}).get('base_engagement', 1.0)
        
        final_scores = scores * fit_vec * mask * base_eng
        return final_scores

    def recommend(self, content_item):
        """
        Executes Issue 8 (Joint Optimization) and Issue 12 (Output Format).
        """
        final_scores = self.get_final_scores(content_item)
        
        # Issue 11: Deterministic Argmax
        best_arm_idx = int(np.argmax(final_scores))
        platform, time_slot = self.store.arms[best_arm_idx]
        
        # Issue 9: Scheduling Decision
        # Compare current submission time slot vs recommended slot
        current_hour = content_item['created_timestamp'] % 24
        decision = "POST_NOW" if current_hour == time_slot else "SCHEDULE"
        
        return {
            "content_id": content_item['content_id'],
            "platform": platform,
            "time_slot": time_slot,
            "decision": decision,
            "expected_score": float(final_scores[best_arm_idx])
        }