"""
Unit tests for ContentBandit implementation
Tests initialization, warm start, and recommendation logic
"""

import pytest
import numpy as np
import pandas as pd
from app.core.bandit import ContentBandit


class TestContentBandit:
    """Test suite for ContentBandit class"""
    
    @pytest.fixture
    def bandit(self):
        """Fixture to create a fresh bandit instance"""
        return ContentBandit(n_platforms=2, n_hours=24, seed=42)
    
    def test_initialization(self, bandit):
        """Test bandit initializes with correct number of arms"""
        assert bandit.n_arms == 48, "Should have 48 arms (2 platforms × 24 hours)"
        assert len(bandit.alpha) == 48
        assert len(bandit.beta) == 48
        assert np.all(bandit.alpha == 1.0), "Alpha should be initialized to 1"
        assert np.all(bandit.beta == 1.0), "Beta should be initialized to 1"
    
    def test_arm_index_mapping(self, bandit):
        """Test arm index calculation for platform/hour pairs"""
        # YouTube
        assert bandit._get_arm_index("youtube", 0) == 0
        assert bandit._get_arm_index("YouTube", 5) == 5
        assert bandit._get_arm_index("YOUTUBE", 23) == 23
        
        # Instagram
        assert bandit._get_arm_index("instagram", 0) == 24
        assert bandit._get_arm_index("Instagram", 5) == 29
        assert bandit._get_arm_index("INSTAGRAM", 23) == 47
    
    def test_arm_details_recovery(self, bandit):
        """Test reverse mapping from arm index to platform/hour"""
        # YouTube arm indices
        platform, hour = bandit._get_arm_details(0)
        assert platform == "YouTube" and hour == 0
        
        platform, hour = bandit._get_arm_details(12)
        assert platform == "YouTube" and hour == 12
        
        platform, hour = bandit._get_arm_details(23)
        assert platform == "YouTube" and hour == 23
        
        # Instagram arm indices
        platform, hour = bandit._get_arm_details(24)
        assert platform == "Instagram" and hour == 0
        
        platform, hour = bandit._get_arm_details(35)
        assert platform == "Instagram" and hour == 11
        
        platform, hour = bandit._get_arm_details(47)
        assert platform == "Instagram" and hour == 23
    
    def test_warm_start_updates_beliefs(self, bandit):
        """Test that warm_start correctly updates alpha/beta parameters"""
        warmstart_data = pd.DataFrame({
            'platform': ['YouTube', 'YouTube', 'Instagram', 'Instagram'],
            'time_slot': [5, 10, 5, 10],
            'avg_engagement': [0.8, 0.6, 0.7, 0.9]
        })
        
        bandit.warm_start(warmstart_data)
        
        # Check YouTube 5:00 (index 5)
        assert bandit.alpha[5] == 1 + 0.8, "Alpha should be updated with engagement"
        assert bandit.beta[5] == 2, "Beta should increment by 1"
        
        # Check YouTube 10:00 (index 10)
        assert bandit.alpha[10] == 1 + 0.6
        assert bandit.beta[10] == 2
        
        # Check Instagram 5:00 (index 29)
        assert bandit.alpha[29] == 1 + 0.7
        assert bandit.beta[29] == 2
        
        # Check Instagram 10:00 (index 34)
        assert bandit.alpha[34] == 1 + 0.9
        assert bandit.beta[34] == 2
    
    def test_recommend_high_time_sensitivity(self, bandit):
        """Test recommendation with high time sensitivity (exploitation)"""
        # Setup: strong signal for YouTube hour 12
        warmstart_data = pd.DataFrame({
            'platform': ['YouTube', 'Instagram'],
            'time_slot': [12, 12],
            'avg_engagement': [0.95, 0.1]
        })
        bandit.warm_start(warmstart_data)
        
        # Activity scores favor YouTube
        activity_scores = np.ones(48) * 0.5
        activity_scores[12] = 0.9  # YouTube 12:00 has high activity
        activity_scores[36] = 0.3  # Instagram 12:00 has low activity
        
        rec = bandit.recommend("High", activity_scores)
        
        assert rec["platform"] == "YouTube", "Should recommend high-activity arm"
        assert rec["recommended_hour"] == 12
        assert rec["decision"] == "DETERMINISTIC"
    
    def test_recommend_low_time_sensitivity(self, bandit):
        """Test recommendation with low time sensitivity (exploration)"""
        activity_scores = np.ones(48) * 0.5
        activity_scores[12] = 0.9
        
        # Thompson sampling is stochastic, but should still return valid arm
        rec = bandit.recommend("Low", activity_scores)
        
        assert rec["platform"] in ["YouTube", "Instagram"]
        assert 0 <= rec["recommended_hour"] < 24
        assert "decision" in rec
    
    def test_recommend_returns_valid_structure(self, bandit):
        """Test that recommendation always returns valid structure"""
        activity_scores = np.random.rand(48)
        
        for time_sensitivity in ["High", "Low", "Medium"]:
            rec = bandit.recommend(time_sensitivity, activity_scores)
            
            assert isinstance(rec, dict)
            assert "platform" in rec
            assert "recommended_hour" in rec
            assert "decision" in rec
            assert rec["platform"] in ["YouTube", "Instagram"]
            assert isinstance(rec["recommended_hour"], (int, np.integer))
            assert 0 <= rec["recommended_hour"] < 24
    
    def test_multiple_warm_starts_accumulate(self, bandit):
        """Test that multiple warm starts accumulate belief updates"""
        # First warm start
        ws1 = pd.DataFrame({
            'platform': ['YouTube'],
            'time_slot': [5],
            'avg_engagement': [0.5]
        })
        bandit.warm_start(ws1)
        alpha_after_first = bandit.alpha[5].copy()
        
        # Second warm start
        ws2 = pd.DataFrame({
            'platform': ['YouTube'],
            'time_slot': [5],
            'avg_engagement': [0.3]
        })
        bandit.warm_start(ws2)
        alpha_after_second = bandit.alpha[5]
        
        # Should accumulate
        assert alpha_after_second > alpha_after_first
        assert alpha_after_second == alpha_after_first + 0.3


class TestBanditEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_deterministic_exploitation_picks_max(self):
        """Ensure deterministic mode picks the actual maximum"""
        bandit = ContentBandit(seed=42)
        
        # Create strong signal
        activity_scores = np.zeros(48)
        activity_scores[15] = 1.0  # Maximum
        
        # Many trials should always pick the same arm
        picks = []
        for _ in range(10):
            rec = bandit.recommend("High", activity_scores)
            picks.append((rec["platform"], rec["recommended_hour"]))
        
        # All should be the same
        assert len(set(picks)) == 1, "Deterministic mode should always pick same arm"
        assert picks[0] == ("YouTube", 15)
    
    def test_thompson_sampling_explores(self):
        """Thompson sampling should explore multiple arms over time"""
        bandit = ContentBandit(seed=42)
        activity_scores = np.ones(48) * 0.5
        
        picks = []
        for _ in range(100):
            rec = bandit.recommend("Low", activity_scores)
            picks.append((rec["platform"], rec["recommended_hour"]))
        
        unique_picks = set(picks)
        assert len(unique_picks) > 1, "Thompson sampling should explore different arms"
    
    def test_with_zero_activity_scores(self):
        """Test handling of zero activity scores"""
        bandit = ContentBandit()
        activity_scores = np.zeros(48)
        
        # Should not crash, but recommendations might be arbitrary
        rec = bandit.recommend("High", activity_scores)
        assert rec["platform"] in ["YouTube", "Instagram"]
        assert 0 <= rec["recommended_hour"] < 24


if __name__ == "__main__":
    # Run with: pytest test_bandit.py -v
    pytest.main([__file__, "-v"])
