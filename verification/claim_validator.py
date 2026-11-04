import re
from typing import List, Tuple
from models.verification_models import Claim
from utils.logger import get_logger

logger = get_logger("claim_validator")

# Red-flag superlative phrases that suggest unverified or exaggerated claims
UNSUBSTANTIATED_SUPERLATIVES = [
    r"100%\s+secure",
    r"zero\s+latency",
    r"completely\s+unhackable",
    r"flawless\s+scalability",
    r"infinite\s+throughput",
    r"guaranteed\s+100%",
    r"impossible\s+to\s+fail",
    r"no\s+single\s+point\s+of\s+failure\s+whatsoever",
]


class ClaimValidator:
    """Pre-audit heuristics and consistency checks on claims before adversarial LLM scrutiny."""

    @staticmethod
    def validate_claim_structure(claim: Claim) -> Tuple[bool, List[str]]:
        """Validate syntactic and structural validity of a claim."""
        issues: List[str] = []

        if not claim.claim_text or len(claim.claim_text.strip()) < 10:
            issues.append("Claim text is too short or empty to be verifiable.")

        # Check for ungrounded superlatives
        text_lower = claim.claim_text.lower()
        for pattern in UNSUBSTANTIATED_SUPERLATIVES:
            match = re.search(pattern, text_lower)
            if match:
                issues.append(f"Claim contains ungrounded absolute superlative: '{match.group(0)}'")

        # Check if the claim is just an opinion
        if text_lower.startswith(("i think", "i feel", "it might be nice", "perhaps we could")):
            issues.append("Claim expresses subjective preference rather than a verifiable assertion.")

        is_valid = len(issues) == 0
        return is_valid, issues

    @staticmethod
    def has_sufficient_evidence_grounding(claim: Claim, evidence_count: int) -> bool:
        """Heuristic check: does the claim cite any concrete evidence or verified reasoning?"""
        if evidence_count > 0:
            return True
        # If no explicit evidence, check if supporting reasoning is substantial
        return len(claim.supporting_reasoning.strip()) > 40
