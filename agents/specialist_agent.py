import json
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from models.agent_models import AgentConfig, AgentResponse
from models.schemas import AgentStatus
from utils.helpers import parse_json_safely
from utils.logger import get_logger

logger = get_logger("specialist_agent")


class SpecialistOutputSchema(BaseModel):
    """Pydantic schema requested from LLM for specialist agent execution."""

    deliverable: str = Field(description="Comprehensive analysis and technical deliverable")
    findings: List[str] = Field(default_factory=list, description="Core proposals, decisions, or results")
    claims: List[str] = Field(default_factory=list, description="Factual assertions or quantitative statements requiring verification")
    evidence_references: List[str] = Field(default_factory=list, description="Sources, benchmarks, or logical derivations supporting claims")
    assumptions: List[str] = Field(default_factory=list, description="Underlying assumptions made")
    unresolved_questions: List[str] = Field(default_factory=list, description="Open questions, risks, or caveats")


class SpecialistAgent(BaseAgent):
    """Dynamic specialist agent created on the fly based on planner requirements."""

    def __init__(
        self,
        config: AgentConfig,
        llm_client: Optional[Any] = None,
    ):
        super().__init__(config, llm_client=llm_client)

    def _build_system_instruction(self) -> str:
        return (
            f"You are a dedicated, top-tier domain specialist operating as: {self.role}.\n"
            f"Your specific objective is: {self.objective}\n"
            f"Guidelines and Instructions: {self.instructions or 'Provide rigorous, evidence-grounded analysis.'}\n\n"
            f"CRITICAL VERIFICATION RULES:\n"
            f"1. Clearly separate established facts from working assumptions and inferences.\n"
            f"2. Every concrete assertion (performance metrics, architectural limits, trade-offs, security postures) "
            f"must be presented as a verifiable claim.\n"
            f"3. Do not invent fake research papers, URLs, or benchmarks. If a number is an estimate, state it as an assumption.\n"
            f"4. Highlight any edge cases, risks, or unresolved questions that require attention."
        )

    def _build_execution_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        context_str = ""
        if context:
            # If there are upstream dependencies or shared context
            upstream_results = context.get("upstream_results", [])
            if upstream_results:
                context_str += "\n\n### Upstream Findings from Collaborating Specialists:\n"
                for res in upstream_results:
                    context_str += (
                        f"- Specialist: {res.get('role')} (Subtask: {res.get('subtask_id')}):\n"
                        f"  Summary: {res.get('output', '')[:400]}\n"
                        f"  Key Findings: {json.dumps(res.get('findings', []))}\n"
                    )

            shared_notes = context.get("shared_notes", "")
            if shared_notes:
                context_str += f"\n\n### Shared Task Context:\n{shared_notes}\n"

        prompt = (
            f"Execute your assigned task:\n"
            f"Role: {self.role}\n"
            f"Objective: {self.objective}\n"
            f"{context_str}\n\n"
            f"Provide your response formatted as a structured JSON object with fields:\n"
            f"- 'deliverable': comprehensive technical explanation and recommendations\n"
            f"- 'findings': list of key analytical conclusions\n"
            f"- 'claims': list of specific factual, architectural, or performance claims requiring verification\n"
            f"- 'evidence_references': list of supporting reasons, benchmarks, or domain rationales for the claims\n"
            f"- 'assumptions': list of explicitly acknowledged assumptions\n"
            f"- 'unresolved_questions': list of open questions or unresolved risks"
        )
        return prompt

    async def execute(self, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Run the specialist agent against Gemini LLM and parse structured output."""
        self.update_status(AgentStatus.EXECUTING)
        start_time = time.perf_counter()

        system_instruction = self._build_system_instruction()
        prompt = self._build_execution_prompt(context)

        logger.info(f"Agent [{self.role} ({self.agent_id})] executing objective: {self.objective[:70]}...")

        try:
            # Generate structured response adhering to SpecialistOutputSchema
            structured_out = await self.client.generate_structured(
                prompt=prompt,
                response_schema=SpecialistOutputSchema,
                system_instruction=system_instruction,
                model=self.model_name,
                temperature=self.temperature,
            )

            deliverable = structured_out.deliverable
            findings = structured_out.findings or []
            claims = structured_out.claims or []
            ev_refs = structured_out.evidence_references or []
            assumptions = structured_out.assumptions or []
            questions = structured_out.unresolved_questions or []

        except Exception as e:
            logger.warning(
                f"Agent [{self.role}] structured generation failed ({e}); falling back to text parsing..."
            )
            # Fallback to plain text generation and attempt JSON extraction
            try:
                raw_text = await self.client.generate_text(
                    prompt=prompt + "\nReturn ONLY valid JSON.",
                    system_instruction=system_instruction,
                    model=self.model_name,
                    temperature=self.temperature,
                )
                parsed = parse_json_safely(raw_text)
                deliverable = parsed.get("deliverable", raw_text)
                findings = parsed.get("findings", [])
                claims = parsed.get("claims", [])
                ev_refs = parsed.get("evidence_references", [])
                assumptions = parsed.get("assumptions", [])
                questions = parsed.get("unresolved_questions", [])
            except Exception as fallback_err:
                logger.error(f"Agent [{self.role}] execution failed completely: {fallback_err}")
                self.update_status(AgentStatus.FAILED)
                duration = round(time.perf_counter() - start_time, 3)
                self.last_execution_duration = duration
                raise fallback_err

        # Update internal state
        for f in findings:
            self.record_finding(f)
        for c in claims:
            self.record_claim(c)
        for a in assumptions:
            self.record_assumption(a)
        for r in ev_refs:
            self.record_evidence_ref(r)
        for q in questions:
            self.record_unresolved_question(q)

        duration = round(time.perf_counter() - start_time, 3)
        self.last_execution_duration = duration
        self.update_status(AgentStatus.COMPLETED)

        logger.info(
            f"Agent [{self.role}] completed in {duration}s. "
            f"Claims: {len(self.claims)}, Findings: {len(self.findings)}"
        )

        return AgentResponse(
            agent_id=self.agent_id,
            role=self.role,
            task_id=self.assigned_task_id,
            subtask_id=self.assigned_subtask_id,
            content=deliverable,
            findings=self.findings,
            claims=self.claims,
            evidence_references=self.evidence_references,
            assumptions=self.assumptions,
            unresolved_questions=self.unresolved_questions,
            status=self.status,
            duration_seconds=duration,
        )

