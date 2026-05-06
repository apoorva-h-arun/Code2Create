from app.schemas import DecisionEnum

def evaluate_scheduling(current_hour: int, optimal_hour: int, current_score: float, optimal_score: float) -> DecisionEnum:
    """
    Evaluates whether to post now or schedule for later based on a 15% decay threshold.
    Rule: If optimal_score > (current_score * 1.15), return DecisionEnum.schedule. Otherwise, return DecisionEnum.post_now.
    """
    if optimal_score > (current_score * 1.15):
        return DecisionEnum.schedule
    return DecisionEnum.post_now
