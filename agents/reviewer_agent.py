import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from models.agent_models import AgentConfig, AgentResponse
from models.schemas import AgentStatus, VerificationDecision
from models.verification_models import Claim, Evidence, ReviewFeedback
from utils.helpers import parse_json_safely
from utils.logger import get_logger

logger = get_logger("reviewer_agent")


class ClaimEvaluationSchema(BaseModel):
    """Pydantic schema for individual claim evaluation by Reviewer."""

    claim_id: str
    decision: VerificationDecision
    reasoning: str = Field(description="Adversarial scrutiny of the claim and supporting rationale")
    critique: str = Field(default="", description="Specific flaws, fallacies, or unsubstantiated extrapolations")
    suggested_fix: Optional[str] = Field(default=None, description="Actionable instructions for targeted repair")
    is_supported_by_evidence: bool = Field(description="Is the claim directly backed by valid evidence or logic?")
    is_evidence_relevant: bool = Field(description="Is the cited evidence directly applicable?")
    has_contradictions: bool = Field(description="Does this claim conflict with known facts or other verified claims?")
    assumptions_presented_as_facts: bool = Field(description="Are unverified assumptions framed as certain truth?")
    logically_valid: bool = Field(description="Is the internal deduction logically sound?")


class ReviewerAgent(BaseAgent):
    """Adversarial Reviewer Agent dedicated to independent, evidence-grounded claim scrutiny."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm_client: Optional[Any] = None,
    ):
        cfg = config or AgentConfig(
            role="Adversarial Claim Reviewer",
            objective="Independently interrogate and verify factual and architectural claims.",
            instructions=(
                "Reject claims that lack evidence, confuse assumptions with facts, "
                "or exceed the bounds of their premises. Never accept by consensus alone."
            ),
        )
        super().__init__(cfg, llm_client=llm_client)

    async def evaluate_claim(
        self,
        claim: Claim,
        evidence_items: List[Evidence],
        task_context: str = "",
    ) -> ReviewFeedback:
        """Evaluate a single claim against associated evidence and task context."""
        logger.info(f"Reviewer evaluating claim [{claim.claim_id}]: '{claim.claim_text[:60]}...'")

        # Format evidence items
        ev_text = "No explicit evidence items provided."
        if evidence_items:
            ev_text = "\n".join(
                f"- [Evidence {e.evidence_id} ({e.source_type})]: {e.supporting_content} (Source: {e.source_reference})"
                for e in evidence_items
            )

        system_instruction = (
            "You are an adversarial claim auditor in the MEDHA multi-agent intelligence system. "
            "Your duty is to uphold rigorous truthfulness, sound deduction, and empirical fidelity.\n"
            "DO NOT accept a claim simply because it sounds plausible or because an agent asserted it with high confidence.\n"
            "Scrutinize whether assumptions are masquerading as established facts, whether conclusions outpace the premises, "
            "and whether claims are verifiable.\n\n"
            "Evaluate along these 8 criteria:\n"
            "1. Is the claim supported by evidence?\n"
            "2. Is the evidence relevant?\n"
            "3. Does the evidence actually support the claim?\n"
            "4. Are there contradictions?\n"
            "5. Are assumptions being presented as facts?\n"
            "6. Is the reasoning logically valid?\n"
            "7. Is information missing?\n"
            "8. Is the conclusion stronger than the evidence allows?\n\n"
            "Decisions:\n"
            "- ACCEPT: The claim is strictly supported by evidence or sound deductive necessity.\n"
            "- REJECT: The claim is demonstrably false, deeply contradictory, or dangerous.\n"
            "- REQUEST_REVISION: The claim is promising but unsupported, over-generalized, or confusing assumption with fact.\n"
            "- UNVERIFIABLE: The claim relies on unknowable external facts or unprovable speculation."
        )

        prompt = (
            f"AUDIT THE FOLLOWING CLAIM:\n"
            f"Claim ID: {claim.claim_id}\n"
            f"Claim Text: {claim.claim_text}\n"
            f"Originating Agent: {claim.originating_role} ({claim.originating_agent})\n"
            f"Agent's Supporting Reasoning: {claim.supporting_reasoning or 'None provided.'}\n"
            f"Assumptions Stated: {', '.join(claim.assumptions) if claim.assumptions else 'None stated.'}\n"
            f"Available Evidence:\n{ev_text}\n\n"
            f"Global Task Context: {task_context[:500]}\n\n"
            f"Return a structured evaluation JSON with:\n"
            f"- 'claim_id': '{claim.claim_id}'\n"
            f"- 'decision': one of 'ACCEPT', 'REJECT', 'REQUEST_REVISION', 'UNVERIFIABLE'\n"
            f"- 'reasoning': detailed explanation of audit decision\n"
            f"- 'critique': specific flaws or unsupported leaps\n"
            f"- 'suggested_fix': actionable advice for agent repair (if REJECT or REQUEST_REVISION)\n"
            f"- 'is_supported_by_evidence': bool\n"
            f"- 'is_evidence_relevant': bool\n"
            f"- 'has_contradictions': bool\n"
            f"- 'assumptions_presented_as_facts': bool\n"
            f"- 'logically_valid': bool"
        )

        try:
            eval_out = await self.client.generate_structured(
                prompt=prompt,
                response_schema=ClaimEvaluationSchema,
                system_instruction=system_instruction,
                model=self.model_name,
                temperature=0.1,  # Low temperature for strict auditing
            )
            return ReviewFeedback(
                claim_id=claim.claim_id,
                decision=eval_out.decision,
                reasoning=eval_out.reasoning,
                critique=eval_out.critique,
                suggested_fix=eval_out.suggested_fix,
                is_supported_by_evidence=eval_out.is_supported_by_evidence,
                is_evidence_relevant=eval_out.is_evidence_relevant,
                has_contradictions=eval_out.has_contradictions,
                assumptions_presented_as_facts=eval_out.assumptions_presented_as_facts,
                logically_valid=eval_out.logically_valid,
            )
        except Exception as e:
            logger.warning(f"Structured review failed ({e}), falling back to text parsing...")
            raw_text = await self.client.generate_text(
                prompt=prompt + "\nReturn ONLY valid JSON.",
                system_instruction=system_instruction,
                model=self.model_name,
                temperature=0.1,
            )
            parsed = parse_json_safely(raw_text)
            dec_str = parsed.get("decision", "REQUEST_REVISION").upper()
            decision = VerificationDecision(dec_str) if dec_str in VerificationDecision.__members__ else VerificationDecision.REQUEST_REVISION

            return ReviewFeedback(
                claim_id=claim.claim_id,
                decision=decision,
                reasoning=parsed.get("reasoning", "Audit completed via text fallback."),
                critique=parsed.get("critique", ""),
                suggested_fix=parsed.get("suggested_fix", None),
                is_supported_by_evidence=bool(parsed.get("is_supported_by_evidence", False)),
                is_evidence_relevant=bool(parsed.get("is_evidence_relevant", False)),
                has_contradictions=bool(parsed.get("has_contradictions", False)),
                assumptions_presented_as_facts=bool(parsed.get("assumptions_presented_as_facts", False)),
                logically_valid=bool(parsed.get("logically_valid", False)),
            )

    async def execute(self, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Generic execution implementation to satisfy BaseAgent contract."""
        self.update_status(AgentStatus.COMPLETED)
        return AgentResponse(
            agent_id=self.agent_id,
            role=self.role,
            task_id=self.assigned_task_id,
            subtask_id=self.assigned_subtask_id,
            content="Adversarial Reviewer initialized and ready for claim audit.",
            status=self.status,
        )

