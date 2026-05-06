from fastapi import FastAPI
from app.schemas import ContentSubmission, RecommendationOutput, PlatformEnum
from app.scheduler import evaluate_scheduling

app = FastAPI()

# Mock global state object to simulate the in-memory NumPy tensors
class MockGlobalState:
    def __init__(self):
        self.is_initialized = True

global_state = MockGlobalState()

def dummy_vectorized_ucb(creator_id: int, content_type: str):
    """
    Mock vectorized UCB math engine. 
    Instantly returns (optimal_platform, optimal_hour, current_score, optimal_score) in O(1) time.
    """
    optimal_platform = PlatformEnum.instagram if content_type == "SHORT" else PlatformEnum.youtube
    optimal_hour = (creator_id % 24)
    current_score = 1.0
    optimal_score = 1.2 if creator_id % 2 == 0 else 1.1 
    return optimal_platform, optimal_hour, current_score, optimal_score

@app.post("/get_recommendation", response_model=RecommendationOutput)
async def get_recommendation(submission: ContentSubmission) -> RecommendationOutput:
    optimal_platform, optimal_hour, current_score, optimal_score = dummy_vectorized_ucb(
        creator_id=submission.creator_id,
        content_type=submission.content_type
    )
    
    current_hour = 12 
    
    decision = evaluate_scheduling(
        current_hour=current_hour,
        optimal_hour=optimal_hour,
        current_score=current_score,
        optimal_score=optimal_score
    )
    
    return RecommendationOutput(
        content_id=submission.content_id,
        platform=optimal_platform,
        time_slot=optimal_hour,
        decision=decision
    )