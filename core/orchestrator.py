import asyncio
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from communication.message_bus import MessageBus
from communication.shared_context import SharedContext
from config.settings import (
    EXECUTION_TIMEOUT_SECONDS,
    MAX_CONCURRENT_AGENTS,
    MAX_REPAIR_ITERATIONS,
    validate_settings,
)
from core.agent_factory import AgentFactory
from core.execution_engine import ExecutionEngine
from core.llm_client import GeminiClient
from core.planner import TaskPlanner
from core.workflow import WorkflowState, WorkflowTracker
from models.schemas import BaseSchema
from models.task_models import TaskPlan, TaskResult
from utils.helpers import generate_id, utcnow_iso
from utils.logger import ExecutionRecord, get_logger
from utils.token_tracker import TokenTracker
from verification.adversarial_reviewer import AdversarialReviewer
from verification.claim_extractor import ClaimExtractor
from verification.evidence_ledger import EvidenceLedger
from verification.repair_engine import RepairEngine

logger = get_logger("orchestrator")


class MedhaResult(BaseSchema):
    """Complete, transparent outcome returned by the MEDHA multi-agent system."""

    execution_id: str
    task: str
    final_answer: str
    status: str = "COMPLETED"
    duration_seconds: float = 0.0
    token_usage: Dict[str, Any] = Field(default_factory=dict)
    generated_agents: List[Dict[str, Any]] = Field(default_factory=list)
    task_results: List[Dict[str, Any]] = Field(default_factory=list)
    claims: List[Dict[str, Any]] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    review_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    repair_history: List[Dict[str, Any]] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)
    verification_report: Dict[str, Any] = Field(default_factory=dict)
    workflow_stages: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None


class MedhaOrchestrator:
    """Master controller managing the dynamic lifecycle of MEDHA."""

    def __init__(
        self,
        llm_client: Optional[GeminiClient] = None,
        max_concurrency: int = MAX_CONCURRENT_AGENTS,
        max_repair_iterations: int = MAX_REPAIR_ITERATIONS,
        timeout_seconds: float = EXECUTION_TIMEOUT_SECONDS,
    ):
        self.tracker = TokenTracker(name="medha_orchestrator")
        self.client = llm_client or GeminiClient(token_tracker=self.tracker)
        self.max_concurrency = max_concurrency
        self.max_repair_iterations = max_repair_iterations
        self.timeout_seconds = timeout_seconds

    async def run_task(self, user_input: str) -> MedhaResult:
        """Execute the end-to-end evidence-driven multi-agent lifecycle for a given user task."""
        execution_id = generate_id("exec")
        exec_record = ExecutionRecord(execution_id=execution_id, task_description=user_input)
        workflow = WorkflowTracker(execution_id=execution_id)
        start_perf = time.perf_counter()

        logger.info(f"=== Starting MEDHA Execution [{execution_id}] ===")
        logger.info(f"Task: '{user_input[:80]}...'")

        try:
            # -------------------------------------------------------------
            # Stage 1: DYNAMIC TASK PLANNING & SPECIALIST IDENTIFICATION
            # -------------------------------------------------------------
            workflow.transition_to(WorkflowState.PLANNING)
            planner = TaskPlanner(llm_client=self.client)
            plan: TaskPlan = await planner.create_plan(
                task_description=user_input,
                max_subtasks=self.max_concurrency,
            )

            # -------------------------------------------------------------
            # Stage 2: AGENT FACTORY & DEPENDENCY-AWARE EXECUTION
            # -------------------------------------------------------------
            workflow.transition_to(
                WorkflowState.EXECUTING_SUBTASKS,
                details={"planned_subtasks": [s.role for s in plan.subtasks]},
            )
            agent_factory = AgentFactory(llm_client=self.client)
            message_bus = MessageBus()
            shared_context = SharedContext(task_id=execution_id, task_description=user_input)

            execution_engine = ExecutionEngine(
                max_concurrency=self.max_concurrency,
                timeout_seconds=self.timeout_seconds,
            )
            task_results: List[TaskResult] = await execution_engine.execute_plan(
                plan=plan,
                agent_factory=agent_factory,
                message_bus=message_bus,
                shared_context=shared_context,
                task_id=execution_id,
            )

            # Log agent activity to execution record
            for r in task_results:
                exec_record.log_agent_activity(
                    agent_id=r.agent_id,
                    role=r.role,
                    task_id=r.subtask_id,
                    status=r.status,
                    duration_seconds=r.duration_seconds,
                    error=r.error,
                )

            # -------------------------------------------------------------
            # Stage 3: CLAIM EXTRACTION & EVIDENCE LEDGER INITIALIZATION
            # -------------------------------------------------------------
            workflow.transition_to(WorkflowState.EXTRACTING_CLAIMS)
            ledger = EvidenceLedger(task_id=execution_id)
            claim_extractor = ClaimExtractor(llm_client=self.client)

            for result in task_results:
                if result.status == "COMPLETED":
                    extracted_claims = await claim_extractor.extract_claims_from_result(
                        result=result,
                        task_id=execution_id,
                    )
                    await ledger.register_claims_batch(extracted_claims)

            # -------------------------------------------------------------
            # Stage 4: ADVERSARIAL AUDIT
            # -------------------------------------------------------------
            workflow.transition_to(WorkflowState.AUDITING_CLAIMS)
            reviewer_agent = agent_factory.create_reviewer()
            reviewer = AdversarialReviewer(reviewer_agent=reviewer_agent, llm_client=self.client)
            review_feedbacks = await reviewer.review_claims(
                ledger=ledger,
                task_context=user_input,
            )

            # -------------------------------------------------------------
            # Stage 5: TARGETED REPAIR & RE-EVALUATION
            # -------------------------------------------------------------
            workflow.transition_to(WorkflowState.REPAIRING_CLAIMS)
            repair_engine = RepairEngine(
                max_repair_iterations=self.max_repair_iterations,
                llm_client=self.client,
            )
            repair_attempts = await repair_engine.execute_repair_loop(
                ledger=ledger,
                reviewer=reviewer,
                agent_factory=agent_factory,
                task_context=user_input,
            )

            # -------------------------------------------------------------
            # Stage 6: FINAL SYNTHESIS & REPORT GENERATION
            # -------------------------------------------------------------
            workflow.transition_to(WorkflowState.SYNTHESIZING)
            verification_report = await ledger.generate_report(
                review_decisions=review_feedbacks,
                repair_history=repair_attempts,
            )

            # Collect unresolved questions from results and agents
            all_questions = []
            for r in task_results:
                all_questions.extend(r.unresolved_questions)
            for a in agent_factory.list_agents():
                all_questions.extend(a.unresolved_questions)
            # Deduplicate
            all_questions = list(dict.fromkeys(all_questions))

            # Synthesize final answer
            synthesizer = agent_factory.create_synthesizer()
            final_answer = await synthesizer.synthesize(
                task_description=user_input,
                specialist_results=[r.model_dump() for r in task_results],
                verification_report=verification_report,
                unresolved_questions=all_questions,
            )

            workflow.transition_to(WorkflowState.COMPLETED)
            total_duration = round(time.perf_counter() - start_perf, 3)

            # Retrieve token usage
            token_summary = await self.tracker.get_summary()

            # Finalize execution record and save structured log
            exec_record.finish(status="COMPLETED", token_usage=token_summary)
            exec_record.save()

            logger.info(
                f"=== MEDHA Execution [{execution_id}] COMPLETED in {total_duration}s ==="
            )

            all_claims_list = [c.model_dump() for c in await ledger.get_all_claims()]
            all_evidence_list = [e.model_dump() for e in await ledger.get_all_evidence()]

            return MedhaResult(
                execution_id=execution_id,
                task=user_input,
                final_answer=final_answer,
                status="COMPLETED",
                duration_seconds=total_duration,
                token_usage=token_summary,
                generated_agents=[a.get_summary() for a in agent_factory.list_agents()],
                task_results=[r.model_dump() for r in task_results],
                claims=all_claims_list,
                evidence=all_evidence_list,
                review_decisions=[rf.model_dump() for rf in review_feedbacks],
                repair_history=[ra.model_dump() for ra in repair_attempts],
                unresolved_questions=all_questions,
                verification_report=verification_report.model_dump(),
                workflow_stages=workflow.get_summary(),
            )

        except Exception as e:
            total_duration = round(time.perf_counter() - start_perf, 3)
            logger.error(f"MEDHA Execution [{execution_id}] encountered fatal error: {e}", exc_info=True)
            workflow.fail(str(e))
            exec_record.record_error(component="orchestrator", message=str(e))
            exec_record.finish(status="FAILED")
            exec_record.save()

            token_summary = await self.tracker.get_summary()

            return MedhaResult(
                execution_id=execution_id,
                task=user_input,
                final_answer=f"# Execution Error\n\nMEDHA encountered an error during task processing: {e}",
                status="FAILED",
                duration_seconds=total_duration,
                token_usage=token_summary,
                workflow_stages=workflow.get_summary(),
                error=str(e),
            )

