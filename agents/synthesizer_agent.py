import json
import time
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from models.agent_models import AgentConfig, AgentResponse
from models.schemas import AgentStatus
from models.verification_models import Claim, RepairAttempt, VerificationReport
from utils.logger import get_logger

logger = get_logger("synthesizer_agent")


class SynthesizerAgent(BaseAgent):
    """Synthesizer agent responsible for combining accepted specialist findings into a transparent final answer."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm_client: Optional[Any] = None,
    ):
        cfg = config or AgentConfig(
            role="Final Synthesizer",
            objective="Synthesize verified specialist outputs into a rigorous, transparent solution.",
            instructions=(
                "Synthesize accepted claims, explicitly exclude rejected claims, "
                "disclose assumptions, and preserve unresolved questions and uncertainties."
            ),
        )
        super().__init__(cfg, llm_client=llm_client)

    async def synthesize(
        self,
        task_description: str,
        specialist_results: List[Dict[str, Any]],
        verification_report: VerificationReport,
        unresolved_questions: List[str],
    ) -> str:
        """Produce the final transparent synthesized answer."""
        self.update_status(AgentStatus.EXECUTING)
        start_time = time.perf_counter()

        logger.info(f"Synthesizer synthesizing {len(specialist_results)} specialist results...")

        # Format accepted claims
        accepted_text = "\n".join(
            f"- [Claim {c.claim_id}] ({c.originating_role}): {c.claim_text}"
            for c in verification_report.accepted_claims
        ) or "None explicitly accepted."

        # Format rejected / excluded claims
        rejected_text = "\n".join(
            f"- [REJECTED Claim {c.claim_id}] ({c.originating_role}): {c.claim_text} | Reason: {c.reviewer_feedback or 'Unsupported'}"
            for c in verification_report.rejected_claims
        ) or "None rejected."

        # Format repairs
        repairs_text = "\n".join(
            f"- [Repaired Claim {r.claim_id}]: Original '{r.original_claim_text}' -> Revised: '{r.revised_claim_text}'"
            for r in verification_report.repair_history if r.revised_claim_text
        ) or "No repairs performed."

        # Format specialist deliverables
        specialist_text = ""
        for s in specialist_results:
            specialist_text += (
                f"\n\n### Specialist: {s.get('role')} (Subtask: {s.get('subtask_id')})\n"
                f"Findings: {json.dumps(s.get('findings', []))}\n"
                f"Assumptions: {json.dumps(s.get('assumptions', []))}\n"
                f"Deliverable Details:\n{s.get('output', '')}\n"
            )

        questions_text = "\n".join(f"- {q}" for q in unresolved_questions) or "None recorded."

        system_instruction = (
            "You are the Lead Synthesizer for MEDHA (Multi-Agent Evidence-Driven Architecture).\n"
            "Your objective is to craft the definitive, evidence-grounded final answer to the user's task.\n\n"
            "SYNTHESIS GUIDELINES:\n"
            "1. Coherent Integration: Harmonize all validated contributions into an actionable, comprehensive solution.\n"
            "2. Evidence Traceability: Base factual conclusions strictly on ACCEPTED claims. Reference claim IDs where helpful.\n"
            "3. Explicit Exclusions: Do NOT include rejected claims as facts. Disclose what was rejected or unverified and why.\n"
            "4. Transparent Uncertainty: Do not paper over disagreements or unknowns. Emphasize unresolved questions.\n"
            "5. Structure: Use clean Markdown headers (Executive Summary, Architecture/Solution Design, "
            "Evidence & Verification Audit, Operating Assumptions, Unresolved Edge Cases)."
        )

        prompt = (
            f"ORIGINAL USER TASK:\n{task_description}\n\n"
            f"VERIFICATION STATUS:\n"
            f"- Accepted Claims ({len(verification_report.accepted_claims)}):\n{accepted_text}\n\n"
            f"- Rejected Claims ({len(verification_report.rejected_claims)}):\n{rejected_text}\n\n"
            f"- Repaired Claims ({len(verification_report.repair_history)}):\n{repairs_text}\n\n"
            f"SPECIALIST CONTRIBUTIONS:\n{specialist_text}\n\n"
            f"UNRESOLVED QUESTIONS & RISKS:\n{questions_text}\n\n"
            f"Please synthesize the final, comprehensive, and evidence-traceable response for the user."
        )

        try:
            final_answer = await self.client.generate_text(
                prompt=prompt,
                system_instruction=system_instruction,
                model=self.model_name,
                temperature=self.temperature,
            )
        except Exception as e:
            logger.error(f"Synthesizer failed to generate final text: {e}")
            final_answer = (
                f"# MEDHA Synthesis (Fallback)\n\n"
                f"An error occurred during final text generation: {e}.\n\n"
                f"### Verified Findings Summary:\n{accepted_text}\n\n"
                f"### Excluded/Rejected Claims:\n{rejected_text}"
            )

        duration = round(time.perf_counter() - start_time, 3)
        self.last_execution_duration = duration
        self.update_status(AgentStatus.COMPLETED)

        logger.info(f"Synthesizer completed final answer in {duration}s ({len(final_answer)} chars).")
        return final_answer

    async def execute(self, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Generic execution implementation to satisfy BaseAgent contract."""
        self.update_status(AgentStatus.COMPLETED)
        return AgentResponse(
            agent_id=self.agent_id,
            role=self.role,
            task_id=self.assigned_task_id,
            subtask_id=self.assigned_subtask_id,
            content="Synthesizer initialized.",
            status=self.status,
        )

