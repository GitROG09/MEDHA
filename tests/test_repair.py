from unittest.mock import AsyncMock, MagicMock
import pytest

from core.agent_factory import AgentFactory
from core.llm_client import GeminiClient
from models.schemas import ClaimStatus, VerificationDecision
from models.verification_models import Claim
from verification.adversarial_reviewer import AdversarialReviewer
from verification.evidence_ledger import EvidenceLedger
from verification.repair_engine import RepairEngine


def _mock_llm_response(text: str):
    resp = MagicMock()
    resp.text = text
    resp.parsed = None
    usage = MagicMock()
    usage.prompt_token_count = 10
    usage.candidates_token_count = 20
    usage.total_token_count = 30
    usage.cached_content_token_count = 0
    resp.usage_metadata = usage
    return resp


@pytest.mark.asyncio
async def test_repair_engine_successful_revision():
    """Verify RepairEngine routes critique, gets revised claim, and triggers re-review."""
    ledger = EvidenceLedger(task_id="repair_test")
    c1 = Claim(
        claim_id="c1",
        claim_text="The system will handle unlimited traffic without failing.",
        originating_agent="agent_lead",
        originating_role="Lead Architect",
        status=ClaimStatus.REJECTED,
        reviewer_feedback="[REJECT] Claim of 'unlimited traffic' is physically impossible.",
    )
    await ledger.register_claim(c1)

    # Mock LLM for repair response
    repair_json = """{
        "revised_claim_text": "The system will handle up to 50,000 QPS with horizontal pod autoscaling.",
        "additional_reasoning": "Bounded capacity to 50,000 QPS based on network bandwidth and cluster limits.",
        "new_evidence_references": ["Kubernetes HPA load testing standard"],
        "converted_to_assumption": false
    }"""
    mock_repair_client = MagicMock()
    mock_repair_client.aio.models.generate_content = AsyncMock(return_value=_mock_llm_response(repair_json))
    gemini_repair_client = GeminiClient(api_key="mock_key", client=mock_repair_client)

    # Mock reviewer for re-review (accepting the repaired claim)
    review_accept_json = """{
        "claim_id": "c1",
        "decision": "ACCEPT",
        "reasoning": "Revised claim is realistically bounded.",
        "critique": "",
        "suggested_fix": null,
        "is_supported_by_evidence": true,
        "is_evidence_relevant": true,
        "has_contradictions": false,
        "assumptions_presented_as_facts": false,
        "logically_valid": true
    }"""
    mock_review_client = MagicMock()
    mock_review_client.aio.models.generate_content = AsyncMock(return_value=_mock_llm_response(review_accept_json))
    gemini_review_client = GeminiClient(api_key="mock_key", client=mock_review_client)

    reviewer = AdversarialReviewer(llm_client=gemini_review_client)
    factory = AgentFactory(llm_client=gemini_repair_client)
    repair_engine = RepairEngine(max_repair_iterations=2, llm_client=gemini_repair_client)

    attempts = await repair_engine.execute_repair_loop(
        ledger=ledger,
        reviewer=reviewer,
        agent_factory=factory,
        task_context="Architecture load testing",
    )

    assert len(attempts) >= 1
    assert "up to 50,000 QPS" in attempts[0].revised_claim_text
    repaired_claim = await ledger.get_claim("c1")
    assert repaired_claim.status == ClaimStatus.SUPPORTED


@pytest.mark.asyncio
async def test_repair_engine_preserves_uncertainty_on_exhaustion():
    """Verify unresolvable claims are marked UNVERIFIABLE after max iterations."""
    ledger = EvidenceLedger(task_id="repair_fail_test")
    c1 = Claim(
        claim_id="c1",
        claim_text="Aliens built the pyramids in 2500 BC.",
        originating_agent="agent_spec",
        originating_role="Speculator",
        status=ClaimStatus.NEEDS_REVISION,
        reviewer_feedback="[REQUEST_REVISION] No archaeological evidence supports extraterrestrial origin.",
    )
    await ledger.register_claim(c1)

    # Mock repair that still fails to provide evidence
    repair_json = """{
        "revised_claim_text": "Extraterrestrial artifacts remain undiscovered under the Sphinx.",
        "additional_reasoning": "Speculative hypothesis.",
        "new_evidence_references": [],
        "converted_to_assumption": false
    }"""
    mock_repair_client = MagicMock()
    mock_repair_client.aio.models.generate_content = AsyncMock(return_value=_mock_llm_response(repair_json))
    gemini_repair_client = GeminiClient(api_key="mock_key", client=mock_repair_client)

    # Mock reviewer continuously rejecting
    review_reject_json = """{
        "claim_id": "c1",
        "decision": "REQUEST_REVISION",
        "reasoning": "Still lacking empirical proof.",
        "critique": "Unproven speculation",
        "suggested_fix": null,
        "is_supported_by_evidence": false,
        "is_evidence_relevant": false,
        "has_contradictions": false,
        "assumptions_presented_as_facts": true,
        "logically_valid": false
    }"""
    mock_review_client = MagicMock()
    mock_review_client.aio.models.generate_content = AsyncMock(return_value=_mock_llm_response(review_reject_json))
    gemini_review_client = GeminiClient(api_key="mock_key", client=mock_review_client)

    reviewer = AdversarialReviewer(llm_client=gemini_review_client)
    factory = AgentFactory(llm_client=gemini_repair_client)
    repair_engine = RepairEngine(max_repair_iterations=1, llm_client=gemini_repair_client)

    await repair_engine.execute_repair_loop(
        ledger=ledger,
        reviewer=reviewer,
        agent_factory=factory,
    )

    repaired_claim = await ledger.get_claim("c1")
    # Must be marked UNVERIFIABLE to preserve uncertainty rather than falsely accepted
    assert repaired_claim.status == ClaimStatus.UNVERIFIABLE

