from enum import Enum
from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    """Base Pydantic model for MEDHA schemas with standard config."""

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        populate_by_name=True,
    )


class TaskStatus(str, Enum):
    """Status lifecycle of a task or subtask."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class AgentStatus(str, Enum):
    """Status lifecycle of an agent."""

    INITIALIZED = "INITIALIZED"
    THINKING = "THINKING"
    EXECUTING = "EXECUTING"
    WAITING = "WAITING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class MessageType(str, Enum):
    """Types of inter-agent messages on the message bus."""

    FINDING = "FINDING"
    QUESTION = "QUESTION"
    CHALLENGE = "CHALLENGE"
    RESPONSE = "RESPONSE"
    EVIDENCE = "EVIDENCE"
    CORRECTION = "CORRECTION"
    STATUS = "STATUS"


class ClaimStatus(str, Enum):
    """Verification status of a claim in the evidence ledger."""

    PENDING = "PENDING"
    SUPPORTED = "SUPPORTED"
    REJECTED = "REJECTED"
    NEEDS_REVISION = "NEEDS_REVISION"
    UNVERIFIABLE = "UNVERIFIABLE"


class VerificationDecision(str, Enum):
    """Decision made by the adversarial reviewer for a claim."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    REQUEST_REVISION = "REQUEST_REVISION"
    UNVERIFIABLE = "UNVERIFIABLE"


class EvidenceSourceType(str, Enum):
    """Source classification for pieces of evidence."""

    USER_INPUT = "USER_INPUT"
    TOOL_RETRIEVED = "TOOL_RETRIEVED"
    REASONING_INFERENCE = "REASONING_INFERENCE"
    DOMAIN_KNOWLEDGE = "DOMAIN_KNOWLEDGE"
    BENCHMARK_DATA = "BENCHMARK_DATA"

