from typing import Any, Optional
from agents.specialist_agent import SpecialistAgent
from models.agent_models import AgentConfig


class ResearchAgent(SpecialistAgent):
    """Specialist agent focused on factual literature, evidence gathering, and state-of-the-art surveying."""

    def __init__(self, config: AgentConfig, llm_client: Optional[Any] = None):
        if not config.role or config.role == "Specialist":
            config.role = "Research Specialist"
        super().__init__(config, llm_client=llm_client)

    def _build_system_instruction(self) -> str:
        base = super()._build_system_instruction()
        return (
            f"{base}\n\n"
            f"RESEARCH DIRECTIVE:\n"
            f"Prioritize historical precedents, established architectural patterns, known constraints, "
            f"and empirical tradeoffs. Avoid speculative extrapolations."
        )

