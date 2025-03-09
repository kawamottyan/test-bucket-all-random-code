from enum import Enum
from typing import Optional

from pydantic import BaseModel


class InteractionType(str, Enum):
    POSTER_VIEWED = "poster_viewed"
    DETAIL_VIEWED = "detail_viewed"
    PLAY_STARTED = "play_started"


class InteractionLog(BaseModel):
    uuid: Optional[str] = None
    interaction_type: InteractionType
    item_id: Optional[str] = None
    watch_time: Optional[str] = None
    query: Optional[str] = None
    index: Optional[str] = None
    created_at: str
    local_timestamp: str
    session_id: Optional[str] = None
    migrated_at: str


class PosterViewedLog(InteractionLog):
    interaction_type: InteractionType = InteractionType.POSTER_VIEWED
    item_id: str
    watch_time: str


class DetailViewedLog(InteractionLog):
    interaction_type: InteractionType = InteractionType.DETAIL_VIEWED
    item_id: str


class PlayStartedLog(InteractionLog):
    interaction_type: InteractionType = InteractionType.PLAY_STARTED
    item_id: str
    watch_time: str
