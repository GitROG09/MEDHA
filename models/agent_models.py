from typing import Any, Dict, List, Optional
from pydantic import Field

from models.schemas import AgentStatus, BaseSchema
from utils.helpers import generate_id


class AgentConfig(BaseSchema):
    """Configuration used to instantiate a dynamic agent."""

    agent_id: str = Field(default_factory=lambda: generate_id("agent"))
    role: str
    objective: str
    instructions: str = ""
    assigned_task_id: str = ""
    assigned_subtask_id: str = ""
    model_name: str = "gemini-3.8-flash"
    temperature: float = 0.2
    expected_output_schema: Optional[Dict[str, Any]] = None


class AgentResponse(BaseSchema):
    """Structured output returned by an agent upon task execution."""

    agent_id: str
    role: str
    task_id: str
    subtask_id: str
    content: str = Field(description="Comprehensive text output or deliverable")
    findings: List[str] = Field(default_factory=list, description="Core analytical discoveries or proposals")
    claims: List[str] = Field(default_factory=list, description="Specific factual assertions requiring verification")
    evidence_references: List[str] = Field(default_factory=list, description="References to supporting materials or logical justifications")
    assumptions: List[str] = Field(default_factory=list, description="Underlying assumptions made during analysis")
    unresolved_questions: List[str] = Field(default_factory=list, description="Known unknowns or edge cases left open")
    status: AgentStatus = AgentStatus.COMPLETED
    duration_seconds: float = 0.0
    token_usage: Dict[str, Any] = Field(default_factory=dict)

