from typing import Any, Dict, List, Optional, Set

from models.schemas import TaskStatus
from models.task_models import SubTask, TaskPlan, TaskResult
from utils.logger import get_logger

logger = get_logger("task_manager")


class TaskManager:
    """Manages subtask lifecycle, dependency resolution, execution status, and collected results."""

    def __init__(self, plan: TaskPlan, task_id: str = ""):
        self.task_id = task_id
        self.plan = plan
        self.subtasks: Dict[str, SubTask] = {s.id: s.model_copy() for s in plan.subtasks}
        self.results: Dict[str, TaskResult] = {}
        self.completed_subtask_ids: Set[str] = set()
        self.failed_subtask_ids: Set[str] = set()

    def get_ready_subtasks(self) -> List[SubTask]:
        """Return all subtasks whose dependencies have been successfully completed and are pending execution."""
        ready = []
        for s_id, subtask in self.subtasks.items():
            if subtask.status == TaskStatus.PENDING:
                # Check if all dependencies are in completed_subtask_ids
                deps = set(subtask.dependencies)
                if deps.issubset(self.completed_subtask_ids):
                    ready.append(subtask)
                elif any(dep in self.failed_subtask_ids for dep in deps):
                    # One of the upstream dependencies failed
                    logger.warning(
                        f"Subtask [{s_id}] has a failed dependency ({deps.intersection(self.failed_subtask_ids)}). "
                        f"Marking as BLOCKED."
                    )
                    subtask.status = TaskStatus.BLOCKED
        return ready

    def mark_in_progress(self, subtask_id: str) -> None:
        """Mark a subtask as currently executing."""
        if subtask_id in self.subtasks:
            self.subtasks[subtask_id].status = TaskStatus.IN_PROGRESS

    def record_result(self, result: TaskResult) -> None:
        """Store the execution result of a subtask and update lifecycle state."""
        subtask_id = result.subtask_id
        self.results[subtask_id] = result

        if subtask_id in self.subtasks:
            self.subtasks[subtask_id].status = result.status

        if result.status == TaskStatus.COMPLETED:
            self.completed_subtask_ids.add(subtask_id)
            logger.info(f"Subtask [{subtask_id}] marked COMPLETED.")
        else:
            self.failed_subtask_ids.add(subtask_id)
            logger.error(f"Subtask [{subtask_id}] marked FAILED ({result.error}).")

    def get_upstream_results(self, subtask_id: str) -> List[Dict[str, Any]]:
        """Get the outputs of all dependencies that completed before this subtask."""
        if subtask_id not in self.subtasks:
            return []
        subtask = self.subtasks[subtask_id]
        upstream = []
        for dep_id in subtask.dependencies:
            if dep_id in self.results:
                r = self.results[dep_id]
                upstream.append({
                    "subtask_id": dep_id,
                    "role": r.role,
                    "output": r.output,
                    "findings": r.findings,
                    "claims": r.claims,
                })
        return upstream

    def is_all_finished(self) -> bool:
        """Check if all subtasks have reached a terminal state (COMPLETED, FAILED, or BLOCKED)."""
        terminal_states = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.BLOCKED}
        return all(s.status in terminal_states for s in self.subtasks.values())

    def get_summary(self) -> Dict[str, Any]:
        """Return execution summary statistics."""
        return {
            "total_subtasks": len(self.subtasks),
            "completed": len(self.completed_subtask_ids),
            "failed": len(self.failed_subtask_ids),
            "blocked": sum(1 for s in self.subtasks.values() if s.status == TaskStatus.BLOCKED),
            "all_finished": self.is_all_finished(),
        }

