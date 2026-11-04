import asyncio
from typing import Any, Dict, List, Optional

from agents.reviewer_agent import ReviewerAgent
from core.llm_client import GeminiClient
from models.schemas import ClaimStatus, VerificationDecision
from models.verification_models import Claim, ReviewFeedback
from verification.claim_validator import ClaimValidator
from verification.evidence_ledger import EvidenceLedger
from utils.logger import get_logger

logger = get_logger("adversarial_reviewer")


class AdversarialReviewer:
    """Orchestrates rigorous, independent adversarial auditing across all claims in the Evidence Ledger."""

    def __init__(
        self,
        reviewer_agent: Optional[ReviewerAgent] = None,
        llm_client: Optional[GeminiClient] = None,
    ):
        self.client = llm_client or GeminiClient.get_instance()
        self.reviewer_agent = reviewer_agent or ReviewerAgent(llm_client=self.client)
        self.validator = ClaimValidator()

    async def review_claims(
        self,
        ledger: EvidenceLedger,
        task_context: str = "",
        target_claims: Optional[List[Claim]] = None,
    ) -> List[ReviewFeedback]:
        """Perform adversarial auditing on pending claims in the ledger."""
        claims_to_review = target_claims or await ledger.get_all_claims(status=ClaimStatus.PENDING)

        logger.info(f"AdversarialReviewer auditing {len(claims_to_review)} claim(s)...")
        feedback_results: List[ReviewFeedback] = []

        for claim in claims_to_review:
            feedback = await self.review_single_claim(claim, ledger, task_context)
            feedback_results.append(feedback)

        accepted_count = sum(1 for f in feedback_results if f.decision == VerificationDecision.ACCEPT)
        rejected_count = sum(1 for f in feedback_results if f.decision == VerificationDecision.REJECT)
        revision_count = sum(1 for f in feedback_results if f.decision == VerificationDecision.REQUEST_REVISION)
        unverifiable_count = sum(1 for f in feedback_results if f.decision == VerificationDecision.UNVERIFIABLE)

        logger.info(
            f"Adversarial audit completed: "
            f"Accepted: {accepted_count}, Rejected: {rejected_count}, "
            f"Needs Revision: {revision_count}, Unverifiable: {unverifiable_count}"
        )

        return feedback_results

    async def review_single_claim(
        self,
        claim: Claim,
        ledger: EvidenceLedger,
        task_context: str = "",
    ) -> ReviewFeedback:
        """Audit an individual claim against its registered evidence."""
        # Step 1: Pre-check heuristic validation
        is_structurally_valid, issues = self.validator.validate_claim_structure(claim)
        evidence_items = await ledger.get_evidence_for_claim(claim.claim_id)

        # Step 2: Adversarial scrutiny via ReviewerAgent
        feedback = await self.reviewer_agent.evaluate_claim(
            claim=claim,
            evidence_items=evidence_items,
            task_context=task_context,
        )

        # If heuristic detected ungrounded superlatives, do not allow plain acceptance
        if not is_structurally_valid and feedback.decision == VerificationDecision.ACCEPT:
            feedback.decision = VerificationDecision.REQUEST_REVISION
            feedback.reasoning += f" [Heuristic Flag: {'; '.join(issues)}]"
            feedback.suggested_fix = (feedback.suggested_fix or "") + " Remove absolute unverified superlatives."

        # Step 3: Map decision to ClaimStatus and update ledger
        if feedback.decision == VerificationDecision.ACCEPT:
            status = ClaimStatus.SUPPORTED
        elif feedback.decision == VerificationDecision.REJECT:
            status = ClaimStatus.REJECTED
        elif feedback.decision == VerificationDecision.REQUEST_REVISION:
            status = ClaimStatus.NEEDS_REVISION
        else:
            status = ClaimStatus.UNVERIFIABLE

        await ledger.update_claim_status(
            claim_id=claim.claim_id,
            status=status,
            reviewer_feedback=f"[{feedback.decision}] {feedback.reasoning}",
        )

        return feedback

