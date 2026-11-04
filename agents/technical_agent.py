from typing import Any, Optional
from agents.specialist_agent import SpecialistAgent
from models.agent_models import AgentConfig


class TechnicalAgent(SpecialistAgent):
    """Specialist agent focused on low-level implementation, protocols, algorithms, and engineering patterns."""

    def __init__(self, config: AgentConfig, llm_client: Optional[Any] = None):
        if not config.role or config.role == "Specialist":
            config.role = "Technical Specialist"
        super().__init__(config, llm_client=llm_client)

    def _build_system_instruction(self) -> str:
        base = super()._build_system_instruction()
        return (
            f"{base}\n\n"
            f"TECHNICAL DIRECTIVE:\n"
            f"Focus on concrete specifications, protocols, data schemas, API contracts, failure modes, "
            f"and resilience mechanics (circuit breakers, idempotency, caching tiers)."
        )

