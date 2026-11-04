import asyncio
import time
from typing import Any, Dict, List, Optional

from communication.message import Message
from communication.message_bus import MessageBus
from communication.shared_context import SharedContext
from config.settings import EXECUTION_TIMEOUT_SECONDS, MAX_CONCURRENT_AGENTS
from core.agent_factory import AgentFactory
from core.task_manager import TaskManager
from models.schemas import MessageType, TaskStatus
from models.task_models import SubTask, TaskPlan, TaskResult
from utils.logger import get_logger

logger = get_logger("execution_engine")


class ExecutionEngine:
    """Dependency-aware parallel execution engine with concurrency limits and fault tolerance."""

    def __init__(
        self,
        max_concurrency: int = MAX_CONCURRENT_AGENTS,
        timeout_seconds: float = EXECUTION_TIMEOUT_SECONDS,
    ):
        self.max_concurrency = max_concurrency
        self.timeout_seconds = timeout_seconds
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def execute_plan(
        self,
        plan: TaskPlan,
        agent_factory: AgentFactory,
        message_bus: MessageBus,
        shared_context: SharedContext,
        task_id: str = "",
    ) -> List[TaskResult]:
        """Execute all planned subtasks respecting dependencies, running parallel stages concurrently."""
        task_manager = TaskManager(plan=plan, task_id=task_id)
        start_time = time.perf_counter()

        logger.info(
            f"ExecutionEngine starting plan with {len(plan.subtasks)} subtasks. "
            f"Max concurrency: {self.max_concurrency}, Timeout: {self.timeout_seconds}s"
        )

        while not task_manager.is_all_finished():
            # Check for global timeout
            elapsed = time.perf_counter() - start_time
            if elapsed > self.timeout_seconds:
                logger.error(f"ExecutionEngine exceeded global timeout of {self.timeout_seconds}s. Halting.")
                break

            ready_subtasks = task_manager.get_ready_subtasks()
            if not ready_subtasks:
                if not task_manager.is_all_finished():
                    logger.warning("No subtasks currently ready but unfinished subtasks remain. Breaking to prevent deadlock.")
                break

            logger.info(
                f"Executing parallel batch of {len(ready_subtasks)} ready subtask(s): "
                f"{[s.id for s in ready_subtasks]}"
            )

            # Mark all ready tasks as in progress
            for subtask in ready_subtasks:
                task_manager.mark_in_progress(subtask.id)

            # Execute batch concurrently under semaphore control
            coros = [
                self._execute_single_subtask(
                    subtask=subtask,
                    task_manager=task_manager,
                    agent_factory=agent_factory,
                    message_bus=message_bus,
                    shared_context=shared_context,
                    task_id=task_id,
                )
                for subtask in ready_subtasks
            ]

            results = await asyncio.gather(*coros, return_exceptions=True)

            # Record results into TaskManager
            for subtask, outcome in zip(ready_subtasks, results):
                if isinstance(outcome, Exception):
                    logger.error(f"Worker for subtask [{subtask.id}] threw an unhandled exception: {outcome}")
                    failed_res = TaskResult(
                        subtask_id=subtask.id,
                        agent_id="unknown",
                        role=subtask.role,
                        status=TaskStatus.FAILED,
                        output="",
                        error=str(outcome),
                    )
                    task_manager.record_result(failed_res)
                elif isinstance(outcome, TaskResult):
                    task_manager.record_result(outcome)

        total_duration = round(time.perf_counter() - start_time, 3)
        summary = task_manager.get_summary()
        logger.info(
            f"ExecutionEngine finished plan in {total_duration}s. "
            f"Completed: {summary['completed']}, Failed: {summary['failed']}, Blocked: {summary['blocked']}"
        )

        return list(task_manager.results.values())

    async def _execute_single_subtask(
        self,
        subtask: SubTask,
        task_manager: TaskManager,
        agent_factory: AgentFactory,
        message_bus: MessageBus,
        shared_context: SharedContext,
        task_id: str,
    ) -> TaskResult:
        """Run a single specialist agent within concurrency limits and error isolation."""
        async with self.semaphore:
            start_time = time.perf_counter()
            agent = agent_factory.create_from_subtask(subtask, task_id=task_id)
            message_bus.register_agent(agent.agent_id)

            try:
                # Gather upstream findings from dependency subtasks
                upstream_results = task_manager.get_upstream_results(subtask.id)

                # Fetch any messages specifically addressed to this agent
                pending_messages = await message_bus.drain_inbox(agent.agent_id)
                incoming_notes = "\n".join(f"[{m.sender_id}]: {m.content}" for m in pending_messages)

                context: Dict[str, Any] = {
                    "upstream_results": upstream_results,
                    "shared_notes": incoming_notes,
                    "task_description": shared_context.task_description,
                }

                # Execute specialist agent
                response = await agent.execute(context=context)

                # Broadcast findings to collaborators via message bus
                for finding in response.findings:
                    await message_bus.publish(
                        Message(
                            sender_id=agent.agent_id,
                            recipient_id="ALL",
                            task_id=task_id,
                            subtask_id=subtask.id,
                            message_type=MessageType.FINDING,
                            content=finding,
                        )
                    )
                    await shared_context.add_finding(agent.agent_id, agent.role, finding)

                # Share evidence references if any
                if response.evidence_references:
                    await message_bus.share_evidence(
                        sender_id=agent.agent_id,
                        content=f"Evidence provided by {agent.role} for subtask {subtask.id}",
                        evidence_references=response.evidence_references,
                        task_id=task_id,
                    )

                duration = round(time.perf_counter() - start_time, 3)

                return TaskResult(
                    subtask_id=subtask.id,
                    agent_id=agent.agent_id,
                    role=agent.role,
                    status=TaskStatus.COMPLETED,
                    output=response.content,
                    findings=response.findings,
                    claims=response.claims,
                    assumptions=response.assumptions,
                    unresolved_questions=response.unresolved_questions,
                    duration_seconds=duration,
                )

            except Exception as e:
                duration = round(time.perf_counter() - start_time, 3)
                logger.error(f"Subtask [{subtask.id}] agent [{agent.role}] failed: {e}")
                return TaskResult(
                    subtask_id=subtask.id,
                    agent_id=agent.agent_id,
                    role=agent.role,
                    status=TaskStatus.FAILED,
                    output="",
                    error=str(e),
                    duration_seconds=duration,
                )

