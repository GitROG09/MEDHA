from typing import Any, Dict, List, Optional
from pydantic import Field

from models.schemas import BaseSchema, MessageType
from utils.helpers import generate_id, utcnow_iso


class Message(BaseSchema):
    """Structured message exchanged between agents on the message bus."""

    message_id: str = Field(default_factory=lambda: generate_id("msg"))
    sender_id: str
    recipient_id: str = Field(default="ALL", description="Target agent ID or 'ALL' for broadcast")
    task_id: str = ""
    subtask_id: Optional[str] = None
    message_type: MessageType
    content: str
    evidence_references: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=utcnow_iso)

