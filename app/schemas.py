from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import Union

class PlatformEnum(str, Enum):
    instagram = "Instagram"
    youtube = "YouTube"

class DecisionEnum(str, Enum):
    post_now = "POST_NOW"
    schedule = "SCHEDULE"

class ContentSubmission(BaseModel):
    content_id: str
    creator_id: int
    content_type: str = Field(pattern="^(SHORT|LONG)$")
    created_timestamp: Union[datetime, int]

class RecommendationOutput(BaseModel):
    content_id: str
    platform: PlatformEnum
    time_slot: int = Field(ge=0, le=23)
    decision: DecisionEnum
