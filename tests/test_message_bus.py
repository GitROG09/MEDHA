from unittest.mock import AsyncMock, MagicMock
import pytest

from communication.message import Message
from communication.message_bus import MessageBus
from communication.shared_context import SharedContext
from core.agent_factory import AgentFactory
from core.execution_engine import ExecutionEngine
from core.llm_client import GeminiClient
from core.task_manager import TaskManager
from models.agent_models import AgentConfig, AgentResponse
from models.schemas import AgentStatus, MessageType, TaskStatus
from models.task_models import SubTask, TaskPlan, TaskResult


@pytest.mark.asyncio
async def test_message_bus_routing_and_isolation():
    """Verify direct and broadcast messaging across multiple agent inboxes."""
    bus = MessageBus()
    bus.register_agent("agent_1")
    bus.register_agent("agent_2")
    bus.register_agent("agent_3")

    # Direct message from agent_1 to agent_2
    msg_direct = Message(
        sender_id="agent_1",
        recipient_id="agent_2",
        message_type=MessageType.QUESTION,
        content="What is the expected QPS?",
    )
    await bus.publish(msg_direct)

    # Broadcast message from agent_3
    msg_broadcast = Message(
        sender_id="agent_3",
        recipient_id="ALL",
        message_type=MessageType.FINDING,
        content="Found cache invalidation bottleneck",
    )
    await bus.publish(msg_broadcast)

    # Check agent_2 inbox (should receive direct message and broadcast message)
    a2_msgs = await bus.drain_inbox("agent_2")
    assert len(a2_msgs) == 2
    assert a2_msgs[0].content == "What is the expected QPS?"
    assert a2_msgs[1].content == "Found cache invalidation bottleneck"

    # Check agent_1 inbox (should only receive broadcast, not direct message meant for agent_2)
    a1_msgs = await bus.drain_inbox("agent_1")
    assert len(a1_msgs) == 1
    assert a1_msgs[0].content == "Found cache invalidation bottleneck"

    # Check agent_3 inbox (should not receive its own broadcast)
    a3_msgs = await bus.drain_inbox("agent_3")
    assert len(a3_msgs) == 0


@pytest.mark.asyncio
async def test_shared_context_thread_safety():
    """Verify concurrent updates to SharedContext."""
    ctx = SharedContext(task_id="t1", task_description="Concurrent test")

    async def add_data(idx):
        await ctx.add_finding(f"agent_{idx}", f"Role_{idx}", f"Finding {idx}")
        await ctx.set_note(f"key_{idx}", idx)

    import asyncio
    await asyncio.gather(*[add_data(i) for i in range(10)])

    findings = await ctx.get_findings()
    notes = await ctx.get_all_notes()
    assert len(findings) == 10
    assert len(notes) == 10
    summary = await ctx.format_summary()
    assert "Finding 0" in summary


def test_task_manager_dependency_resolution():
    """Verify TaskManager identifies ready subtasks and blocks on failed dependencies."""
    plan = TaskPlan(
        task="Test Dependencies",
        subtasks=[
            SubTask(id="T1", role="Role 1", objective="Obj 1", dependencies=[]),
            SubTask(id="T2", role="Role 2", objective="Obj 2", dependencies=[]),
            SubTask(id="T3", role="Role 3", objective="Obj 3", dependencies=["T1", "T2"]),
            SubTask(id="T4", role="Role 4", objective="Obj 4", dependencies=["T3"]),
        ],
    )
    tm = TaskManager(plan=plan, task_id="task_test")

    # Initial ready tasks should be T1 and T2
    ready = tm.get_ready_subtasks()
    ready_ids = {s.id for s in ready}
    assert ready_ids == {"T1", "T2"}

    # Complete T1
    tm.record_result(
        TaskResult(subtask_id="T1", agent_id="a1", role="Role 1", status=TaskStatus.COMPLETED)
    )

    # T3 is not yet ready because T2 is still pending
    ready = tm.get_ready_subtasks()
    assert {s.id for s in ready} == {"T2"}

    # Complete T2
    tm.record_result(
        TaskResult(subtask_id="T2", agent_id="a2", role="Role 2", status=TaskStatus.COMPLETED)
    )

    # Now T3 is ready
    ready = tm.get_ready_subtasks()
    assert {s.id for s in ready} == {"T3"}

    # Fail T3
    tm.record_result(
        TaskResult(subtask_id="T3", agent_id="a3", role="Role 3", status=TaskStatus.FAILED, error="OOM")
    )

    # T4 depends on failed T3, so it should be marked BLOCKED and not be ready
    ready = tm.get_ready_subtasks()
    assert len(ready) == 0
    assert tm.subtasks["T4"].status == TaskStatus.BLOCKED
    assert tm.is_all_finished() is True


@pytest.mark.asyncio
async def test_execution_engine_parallel_flow():
    """Verify ExecutionEngine executes subtasks and respects dependencies."""
    mock_client = MagicMock()
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)
    factory = AgentFactory(llm_client=gemini_client)
    bus = MessageBus()
    shared_ctx = SharedContext(task_id="exec_1", task_description="Parallel test")

    plan = TaskPlan(
        task="Test Parallel Plan",
        subtasks=[
            SubTask(id="T1", role="Systems Architect", objective="Design core", dependencies=[]),
            SubTask(id="T2", role="Security Auditor", objective="Audit security", dependencies=[]),
            SubTask(id="T3", role="Integration Specialist", objective="Integrate", dependencies=["T1", "T2"]),
        ],
    )

    # Mock agent execution
    async def mock_execute(context=None):
        return AgentResponse(
            agent_id="test",
            role="Specialist",
            task_id="exec_1",
            subtask_id="sub",
            content="Deliverable content",
            findings=["Key finding"],
            claims=["Concrete claim"],
            status=AgentStatus.COMPLETED,
        )

    # Patch factory to return mock-executing agents
    orig_create = factory.create_from_subtask

    def patched_create(subtask, task_id=""):
        agent = orig_create(subtask, task_id)
        agent.execute = AsyncMock(side_effect=mock_execute)
        return agent

    factory.create_from_subtask = patched_create

    engine = ExecutionEngine(max_concurrency=2, timeout_seconds=30.0)
    results = await engine.execute_plan(
        plan=plan,
        agent_factory=factory,
        message_bus=bus,
        shared_context=shared_ctx,
        task_id="exec_1",
    )

    assert len(results) == 3
    assert all(r.status == TaskStatus.COMPLETED for r in results)
    history = bus.get_history()
    # Ensure findings were published to message bus
    assert len(history) >= 3

