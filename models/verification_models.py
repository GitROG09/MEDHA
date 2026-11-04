from typing import Any, Dict, List, Optional
from pydantic import Field

from models.schemas import BaseSchema, ClaimStatus, EvidenceSourceType, VerificationDecision
from utils.helpers import generate_id, utcnow_iso


class Evidence(BaseSchema):
    """An individual piece of supporting evidence stored in the Evidence Ledger."""

    evidence_id: str = Field(default_factory=lambda: generate_id("ev"))
    source_type: EvidenceSourceType = EvidenceSourceType.DOMAIN_KNOWLEDGE
    source_reference: str = Field(description="User prompt snippet, retrieval citation, or standard reference")
    supporting_content: str = Field(description="Content of the evidence")
    originating_agent: str = Field(description="ID or role of agent providing evidence")
    associated_claim_ids: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utcnow_iso)


class Claim(BaseSchema):
    """A factual or analytical assertion extracted from agent outputs for verification."""

    claim_id: str = Field(default_factory=lambda: generate_id("claim"))
    claim_text: str = Field(description="The concrete assertion being evaluated")
    originating_agent: str = Field(description="Agent ID that made the claim")
    originating_role: str = Field(default="", description="Role of the agent that made the claim")
    task_id: str = Field(default="")
    subtask_id: str = Field(default="")
    evidence_references: List[str] = Field(default_factory=list, description="IDs of evidence supporting this claim")
    supporting_reasoning: str = Field(default="", description="Logical rationale provided by the agent")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions this claim relies on")
    status: ClaimStatus = Field(default=ClaimStatus.PENDING)
    reviewer_feedback: Optional[str] = None
    verification_history: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=utcnow_iso)
    updated_at: str = Field(default_factory=utcnow_iso)


class ReviewFeedback(BaseSchema):
    """Structured evaluation returned by the Adversarial Reviewer for a claim."""

    review_id: str = Field(default_factory=lambda: generate_id("rev"))
    claim_id: str
    decision: VerificationDecision
    reasoning: str = Field(description="Adversarial evaluation rationale")
    critique: str = Field(default="", description="Specific flaws, fallacies, or unsubstantiated points identified")
    suggested_fix: Optional[str] = Field(default=None, description="Guidance for targeted repair")
    is_supported_by_evidence: bool = Field(default=False)
    is_evidence_relevant: bool = Field(default=False)
    has_contradictions: bool = Field(default=False)
    assumptions_presented_as_facts: bool = Field(default=False)
    logically_valid: bool = Field(default=False)
    timestamp: str = Field(default_factory=utcnow_iso)


class RepairAttempt(BaseSchema):
    """Record of a targeted repair cycle for a rejected claim."""

    repair_id: str = Field(default_factory=lambda: generate_id("rep"))
    attempt_number: int
    claim_id: str
    agent_id: str
    role: str
    original_claim_text: str
    rejection_reason: str
    revised_claim_text: Optional[str] = None
    additional_reasoning: str = ""
    new_evidence_references: List[str] = Field(default_factory=list)
    status: ClaimStatus = ClaimStatus.NEEDS_REVISION
    timestamp: str = Field(default_factory=utcnow_iso)


class VerificationReport(BaseSchema):
    """Comprehensive summary of the claim verification and repair lifecycle."""

    total_claims: int = 0
    accepted_claims: List[Claim] = Field(default_factory=list)
    rejected_claims: List[Claim] = Field(default_factory=list)
    needs_revision_claims: List[Claim] = Field(default_factory=list)
    unverifiable_claims: List[Claim] = Field(default_factory=list)
    review_decisions: List[ReviewFeedback] = Field(default_factory=list)
    repair_history: List[RepairAttempt] = Field(default_factory=list)
    support_rate: float = 0.0
    unsupported_rate: float = 0.0
    repair_success_rate: float = 0.0

