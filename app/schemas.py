"""
Data Validation Schemas
This module defines the strict Pydantic models used for input validation
and response formatting in the Creator Content Posting Optimization System.
"""
from datetime import datetime
from enum import Enum
from typing import Union

from pydantic import BaseModel, Field


class PlatformEnum(str, Enum):
    """Supported platforms for content posting."""
    instagram = "Instagram"
    youtube = "YouTube"


class DecisionEnum(str, Enum):
    """Possible scheduling decisions."""
    post_now = "POST_NOW"
    schedule = "SCHEDULE"


class ContentSubmission(BaseModel):
    """
    Schema for incoming content submission requests.
    Enforces strict typing and regex patterns to ensure valid data.
    """
    content_id: str = Field(
        ..., 
        description="Unique identifier for the content item"
    )
    creator_id: int = Field(
        ..., 
        description="Unique identifier for the creator"
    )
    content_type: str = Field(
        pattern="^(SHORT|LONG)$", 
        description="Must be either 'SHORT' or 'LONG'"
    )
    created_timestamp: Union[datetime, int] = Field(
        ..., 
        description="Unix timestamp or datetime string of creation time"
    )


class RecommendationOutput(BaseModel):
    """
    Schema for the outgoing recommendation response.
    Guarantees the output structure adheres to system requirements.
    """
    content_id: str = Field(
        ..., 
        description="Unique identifier for the content item"
    )
    platform: PlatformEnum = Field(
        ..., 
        description="Recommended platform for posting"
    )
    time_slot: int = Field(
        ge=0, 
        le=23, 
        description="Optimal hour of the day to post (0-23)"
    )
    decision: DecisionEnum = Field(
        ..., 
        description="Final decision whether to post now or schedule"
    )
