"""
Scheduling Engine
This module contains the deterministic logic for deciding when content 
should be posted based on calculated optimization scores.
"""
from app.schemas import DecisionEnum


def evaluate_scheduling(
    current_hour: int, 
    optimal_hour: int, 
    current_score: float, 
    optimal_score: float
) -> DecisionEnum:
    """
    Evaluates the trade-off between posting immediately versus waiting 
    for the optimal hour.

    The decision relies on a strict 15% threshold: if waiting for the 
    optimal time yields a score at least 15% better than posting now, 
    the system will schedule the post. Otherwise, it posts immediately.

    Args:
        current_hour (int): The current hour of the day (0-23).
        optimal_hour (int): The calculated best hour to post (0-23).
        current_score (float): The expected performance score if posted now.
        optimal_score (float): The expected performance score if posted at optimal_hour.

    Returns:
        DecisionEnum: 'SCHEDULE' if the optimal score justifies the wait, else 'POST_NOW'.
    """
    # Calculate the threshold: current score + 15% decay/improvement barrier
    improvement_threshold = current_score * 1.15
    
    # Check if the optimal score breaks the threshold
    if optimal_score > improvement_threshold:
        return DecisionEnum.schedule
        
    return DecisionEnum.post_now
