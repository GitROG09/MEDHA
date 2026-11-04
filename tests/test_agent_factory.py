from unittest.mock import AsyncMock, MagicMock
import pytest

from agents.analyst_agent import AnalystAgent
from agents.research_agent import ResearchAgent
from agents.specialist_agent import SpecialistAgent
from agents.technical_agent import TechnicalAgent
from core.agent_factory import AgentFactory
from core.llm_client import GeminiClient
from core.planner import TaskPlanner
from models.agent_models import AgentConfig
from models.schemas import AgentStatus
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
async def test_planner_creates_plan():
    """Verify TaskPlanner decomposes a task into subtasks dynamically."""
    mock_client = MagicMock()
    plan_json = """{
        "task": "Design scalable quantum algorithm",
        "subtasks": [
            {
                "id": "T1",
                "role": "Quantum Physicist",
                "objective": "Formulate Hamiltonian",
                "instructions": "Focus on qubit coherence",
                "dependencies": [],
                "expected_output": "Hamiltonian formulation"
            },
            {
                "id": "T2",
                "role": "Quantum Error Correction Specialist",
                "objective": "Design surface code",
                "instructions": "Estimate threshold",
                "dependencies": ["T1"],
                "expected_output": "Error budget"
            }
        ],
        "rationale": "Specialized quantum physics and error correction expertise required",
        "completion_criteria": ["Hamiltonian verified", "Error threshold met"]
    }"""
    mock_client.aio.models.generate_content = AsyncMock(return_value=_mock_llm_response(plan_json))
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)
    planner = TaskPlanner(llm_client=gemini_client)

    plan = await planner.create_plan("Design a quantum algorithm")

    assert isinstance(plan, TaskPlan)
    assert len(plan.subtasks) == 2
    assert plan.subtasks[0].role == "Quantum Physicist"
    assert plan.subtasks[1].role == "Quantum Error Correction Specialist"
    assert plan.subtasks[1].dependencies == ["T1"]


@pytest.mark.asyncio
async def test_planner_fallback_on_invalid_json():
    """Verify TaskPlanner falls back to emergency plan if LLM output fails."""
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=_mock_llm_response("NOT VALID JSON"))
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)
    planner = TaskPlanner(llm_client=gemini_client)

    plan = await planner.create_plan("Build an emergency shelter")

    assert isinstance(plan, TaskPlan)
    assert len(plan.subtasks) >= 2
    assert "Primary Domain Specialist" in plan.subtasks[0].role


def test_agent_factory_creates_diverse_roles():
    """Verify AgentFactory routes roles to appropriate classes and caches duplicates."""
    mock_client = MagicMock()
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)
    factory = AgentFactory(llm_client=gemini_client)

    s1 = SubTask(id="T1", role="Deep Learning Research Specialist", objective="Survey transformers")
    s2 = SubTask(id="T2", role="Systems Performance Analyst", objective="Analyze latency")
    s3 = SubTask(id="T3", role="Database Architect", objective="Design schema")
    s4 = SubTask(id="T4", role="Epidemiologist", objective="Model transmission rate")

    a1 = factory.create_from_subtask(s1, task_id="task_1")
    a2 = factory.create_from_subtask(s2, task_id="task_1")
    a3 = factory.create_from_subtask(s3, task_id="task_1")
    a4 = factory.create_from_subtask(s4, task_id="task_1")

    assert isinstance(a1, ResearchAgent)
    assert isinstance(a2, AnalystAgent)
    assert isinstance(a3, TechnicalAgent)
    assert isinstance(a4, SpecialistAgent)
    assert a4.role == "Epidemiologist"

    # Verify duplicate prevention
    a1_dup = factory.create_from_subtask(s1, task_id="task_1")
    assert a1_dup.agent_id == a1.agent_id

    # Verify Reviewer and Synthesizer creation
    rev = factory.create_reviewer()
    synth = factory.create_synthesizer()
    assert rev is not None
    assert synth is not None
    assert len(factory.list_agents()) == 6


@pytest.mark.asyncio
async def test_specialist_agent_execution():
    """Verify SpecialistAgent executes and records findings, claims, and assumptions."""
    mock_client = MagicMock()
    agent_output_json = """{
        "deliverable": "The distributed ledger should use Raft consensus.",
        "findings": ["Raft provides strict leader election"],
        "claims": ["Raft achieves sub-10ms election in local networks", "Throughput reaches 5000 ops/sec"],
        "evidence_references": ["Ongaro & Ousterhout 2014 paper"],
        "assumptions": ["Network partitions are transient"],
        "unresolved_questions": ["Behavior under asymmetric Byzantine faults"]
    }"""
    mock_client.aio.models.generate_content = AsyncMock(return_value=_mock_llm_response(agent_output_json))
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)

    config = AgentConfig(
        agent_id="test_agent",
        role="Distributed Systems Architect",
        objective="Select consensus mechanism",
        instructions="Analyze Raft vs Paxos",
    )
    agent = SpecialistAgent(config=config, llm_client=gemini_client)

    response = await agent.execute(context={"task_context": "High-throughput cluster"})

    assert response.status == AgentStatus.COMPLETED
    assert len(response.findings) == 1
    assert len(response.claims) == 2
    assert "Raft achieves sub-10ms election in local networks" in response.claims
    assert len(response.assumptions) == 1
    assert len(response.unresolved_questions) == 1
    assert agent.status == AgentStatus.COMPLETED

