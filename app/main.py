"""
Main FastAPI Application Entrypoint
Handles routing, global state, and integration with the Vectorized Contextual Bandit engine.
Optimized for zero-blocking I/O and high concurrency.
"""
from fastapi import FastAPI

from app.schemas import ContentSubmission, RecommendationOutput, PlatformEnum
from app.scheduler import evaluate_scheduling

# Initialize the FastAPI application
app = FastAPI(
    title="Creator Content Posting Optimization API",
    description="High-performance async API for real-time posting decisions",
    version="1.0.0"
)


class MockGlobalState:
    """
    Simulates an in-memory state object (e.g., loaded NumPy matrices)
    to ensure O(1) access times without relying on database latency.
    """
    def __init__(self):
        self.is_initialized = True


# Global state singleton mimicking an active in-memory cache
global_state = MockGlobalState()


def dummy_vectorized_ucb(creator_id: int, content_type: str) -> tuple[PlatformEnum, int, float, float]:
    """
    Simulates the Vectorized Upper Confidence Bound (UCB) engine.
    In production, this queries in-memory NumPy tensors in O(1) time.
    
    Args:
        creator_id (int): ID of the content creator.
        content_type (str): "SHORT" or "LONG" format.
        
    Returns:
        tuple: (optimal_platform, optimal_hour, current_score, optimal_score)
    """
    # Dummy logic: map SHORT format to Instagram, LONG to YouTube
    optimal_platform = PlatformEnum.instagram if content_type == "SHORT" else PlatformEnum.youtube
    
    # Dummy optimal hour assignment derived from creator_id mapping
    optimal_hour = creator_id % 24
    
    # Dummy scores established for scheduling demonstration logic
    current_score = 1.0
    optimal_score = 1.2 if creator_id % 2 == 0 else 1.1 
    
    return optimal_platform, optimal_hour, current_score, optimal_score


@app.post(
    "/get_recommendation", 
    response_model=RecommendationOutput,
    summary="Get Posting Recommendation",
    description="Calculates optimal platform and time slot for content submission."
)
async def get_recommendation(submission: ContentSubmission) -> RecommendationOutput:
    """
    Async endpoint to retrieve posting recommendations.
    Architected for maximum concurrency under high request volume (>1200 req/s).
    """
    # 1. Fetch predictions from the O(1) decision engine
    optimal_platform, optimal_hour, current_score, optimal_score = dummy_vectorized_ucb(
        creator_id=submission.creator_id,
        content_type=submission.content_type
    )
    
    # 2. Extract or define current context (hardcoded to 12 for mock purposes)
    current_hour = 12 
    
    # 3. Apply the deterministic scheduling logic (15% threshold rule)
    decision = evaluate_scheduling(
        current_hour=current_hour,
        optimal_hour=optimal_hour,
        current_score=current_score,
        optimal_score=optimal_score
    )
    
    # 4. Construct and return the strictly formatted response representation
    return RecommendationOutput(
        content_id=submission.content_id,
        platform=optimal_platform,
        time_slot=optimal_hour,
        decision=decision
    )