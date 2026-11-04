from unittest.mock import AsyncMock, MagicMock
import pytest

from core.llm_client import GeminiClient
from models.schemas import ClaimStatus, EvidenceSourceType, VerificationDecision
from models.task_models import TaskResult
from models.verification_models import Claim, Evidence, ReviewFeedback
from verification.adversarial_reviewer import AdversarialReviewer
from verification.claim_extractor import ClaimExtractor
from verification.claim_validator import ClaimValidator
from verification.evidence_ledger import EvidenceLedger


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
async def test_claim_extractor_from_result():
    """Verify ClaimExtractor converts pre-identified claims in TaskResult."""
    result = TaskResult(
        subtask_id="T1",
        agent_id="agent_1",
        role="Systems Architect",
        output="Complete architecture report.",
        claims=["Postgres handles 10,000 writes/sec with sharding", "Redis caching reduces DB read load by 80%"],
        assumptions=["Standard SSD NVMe drives used"],
    )
    extractor = ClaimExtractor()
    claims = await extractor.extract_claims_from_result(result, task_id="task_123")

    assert len(claims) == 2
    assert claims[0].originating_agent == "agent_1"
    assert claims[0].originating_role == "Systems Architect"
    assert "10,000 writes/sec" in claims[0].claim_text
    assert claims[0].status == ClaimStatus.PENDING


@pytest.mark.asyncio
async def test_evidence_ledger_lifecycle():
    """Verify registering, linking, and reporting on claims and evidence in EvidenceLedger."""
    ledger = EvidenceLedger(task_id="task_123")

    claim = Claim(
        claim_id="c1",
        claim_text="AES-256 is quantum-resistant against known Grover attacks within 128-bit security margin",
        originating_agent="sec_agent",
        originating_role="Security Specialist",
    )
    await ledger.register_claim(claim)

    evidence = Evidence(
        evidence_id="e1",
        source_type=EvidenceSourceType.DOMAIN_KNOWLEDGE,
        source_reference="NIST Post-Quantum Cryptography Guidelines",
        supporting_content="Grover's algorithm halves key length; AES-256 retains 128 bits of security.",
        originating_agent="sec_agent",
        associated_claim_ids=["c1"],
    )
    await ledger.register_evidence(evidence)

    # Check evidence association
    linked_ev = await ledger.get_evidence_for_claim("c1")
    assert len(linked_ev) == 1
    assert linked_ev[0].evidence_id == "e1"

    # Update claim status
    await ledger.update_claim_status("c1", ClaimStatus.SUPPORTED, reviewer_feedback="Verified via NIST standards")

    # Generate report
    report = await ledger.generate_report()
    assert report.total_claims == 1
    assert len(report.accepted_claims) == 1
    assert report.support_rate == 1.0


def test_claim_validator_flags_superlatives():
    """Verify ClaimValidator flags ungrounded absolutes."""
    c_bad = Claim(
        claim_id="c_bad",
        claim_text="Our system is 100% secure and will achieve zero latency under any load.",
        originating_agent="a1",
    )
    is_valid, issues = ClaimValidator.validate_claim_structure(c_bad)
    assert is_valid is False
    assert any("100% secure" in issue for issue in issues)
    assert any("zero latency" in issue for issue in issues)

    c_good = Claim(
        claim_id="c_good",
        claim_text="Database read replicas scale throughput to approximately 25,000 reads per second.",
        originating_agent="a1",
    )
    is_valid_good, issues_good = ClaimValidator.validate_claim_structure(c_good)
    assert is_valid_good is True
    assert len(issues_good) == 0


@pytest.mark.asyncio
async def test_adversarial_reviewer_audit():
    """Verify AdversarialReviewer rejects unsupported claims and accepts supported ones."""
    mock_client = MagicMock()
    review_accept_json = """{
        "claim_id": "c1",
        "decision": "ACCEPT",
        "reasoning": "Backed by empirical evidence and mathematical proof.",
        "critique": "",
        "suggested_fix": null,
        "is_supported_by_evidence": true,
        "is_evidence_relevant": true,
        "has_contradictions": false,
        "assumptions_presented_as_facts": false,
        "logically_valid": true
    }"""
    mock_client.aio.models.generate_content = AsyncMock(return_value=_mock_llm_response(review_accept_json))
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)

    ledger = EvidenceLedger(task_id="task_audit")
    c1 = Claim(
        claim_id="c1",
        claim_text="Cache-aside pattern prevents database overload on cache hits.",
        originating_agent="agent_arch",
    )
    await ledger.register_claim(c1)

    reviewer = AdversarialReviewer(llm_client=gemini_client)
    feedback_list = await reviewer.review_claims(ledger=ledger, task_context="Caching architecture")

    assert len(feedback_list) == 1
    assert feedback_list[0].decision == VerificationDecision.ACCEPT
    updated_c1 = await ledger.get_claim("c1")
    assert updated_c1.status == ClaimStatus.SUPPORTED

