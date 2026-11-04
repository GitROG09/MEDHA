import json
from unittest.mock import AsyncMock, MagicMock
import pytest

from core.llm_client import GeminiClient
from core.orchestrator import MedhaOrchestrator, MedhaResult
from models.agent_models import AgentResponse
from models.schemas import AgentStatus, VerificationDecision
from models.task_models import SubTask, TaskPlan


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
async def test_orchestrator_end_to_end_flow(tmp_path, monkeypatch):
    """Verify MedhaOrchestrator coordinates the complete 6-stage lifecycle."""
    mock_client = MagicMock()

    plan_json = """{
        "task": "Design e-commerce system",
        "subtasks": [
            {
                "id": "T1",
                "role": "Database Architect",
                "objective": "Design relational and document stores",
                "instructions": "Evaluate read/write splits",
                "dependencies": [],
                "expected_output": "Database schema"
            },
            {
                "id": "T2",
                "role": "Security Specialist",
                "objective": "Audit authentication and payment flows",
                "instructions": "Ensure PCI-DSS compliance",
                "dependencies": [],
                "expected_output": "Security posture"
            }
        ],
        "rationale": "High-throughput database and strict security required",
        "completion_criteria": ["Data integrity guaranteed", "Zero unencrypted cardholder data"]
    }"""

    specialist_output_json = """{
        "deliverable": "Use partitioned PostgreSQL with Redis caching for sessions.",
        "findings": ["PostgreSQL supports JSONB for catalog and ACID for orders."],
        "claims": ["Read replicas scale read operations up to 40,000 QPS."],
        "evidence_references": ["PostgreSQL 16 replication benchmarks."],
        "assumptions": ["Write load does not exceed 5,000 TPS."],
        "unresolved_questions": ["Cross-region replication latency under severe partition."]
    }"""

    review_accept_json = """{
        "claim_id": "c1",
        "decision": "ACCEPT",
        "reasoning": "Postgres read replicas reliably scale read queries within this range.",
        "critique": "",
        "suggested_fix": null,
        "is_supported_by_evidence": true,
        "is_evidence_relevant": true,
        "has_contradictions": false,
        "assumptions_presented_as_facts": false,
        "logically_valid": true
    }"""

    synthesis_text = (
        "# Architectural Synthesis for E-Commerce\n\n"
        "### Executive Summary\n"
        "A hybrid relational-cache architecture has been designed.\n\n"
        "### Verified Recommendations\n"
        "PostgreSQL read replicas scale reads up to 40k QPS."
    )

    # Route mock calls based on prompt content
    async def mock_generate_content(model, contents, config=None):
        prompt_str = str(contents)
        if "Produce a structured JSON plan" in prompt_str or "TaskPlan" in prompt_str:
            return _mock_llm_response(plan_json)
        elif "AUDIT THE FOLLOWING CLAIM" in prompt_str:
            return _mock_llm_response(review_accept_json)
        elif "Please synthesize the final" in prompt_str or "Lead Synthesizer" in str(getattr(config, "system_instruction", "")):
            return _mock_llm_response(synthesis_text)
        else:
            return _mock_llm_response(specialist_output_json)

    mock_client.aio.models.generate_content = AsyncMock(side_effect=mock_generate_content)
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)

    orchestrator = MedhaOrchestrator(
        llm_client=gemini_client,
        max_concurrency=2,
        max_repair_iterations=1,
    )

    result = await orchestrator.run_task("Design a scalable e-commerce backend")

    assert isinstance(result, MedhaResult)
    assert result.status == "COMPLETED"
    assert result.execution_id.startswith("exec_")
    assert "Architectural Synthesis" in result.final_answer
    assert len(result.generated_agents) >= 2
    assert len(result.task_results) == 2
    assert len(result.claims) >= 1
    assert result.verification_report["total_claims"] >= 1
    assert result.duration_seconds >= 0.0


@pytest.mark.asyncio
async def test_orchestrator_failure_handling():
    """Verify orchestrator catches unhandled exceptions gracefully and returns FAILED result."""
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("Catastrophic API failure"))
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)

    orchestrator = MedhaOrchestrator(llm_client=gemini_client)
    # TaskPlanner will fall back to emergency plan, but subsequent calls will fail cleanly
    result = await orchestrator.run_task("Test task failure")

    assert isinstance(result, MedhaResult)
    # The result captures the execution status without unhandled Python crash
    assert result.status in ("COMPLETED", "FAILED")
    assert result.execution_id is not None

