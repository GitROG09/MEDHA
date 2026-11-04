from typing import Any, Optional
from agents.specialist_agent import SpecialistAgent
from models.agent_models import AgentConfig


class CriticAgent(SpecialistAgent):
    """Specialist agent focused on identifying single points of failure, edge-case risks, and unexamined assumptions."""

    def __init__(self, config: AgentConfig, llm_client: Optional[Any] = None):
        if not config.role or config.role == "Specialist":
            config.role = "Adversarial Critic"
        super().__init__(config, llm_client=llm_client)

    def _build_system_instruction(self) -> str:
        base = super()._build_system_instruction()
        return (
            f"{base}\n\n"
            f"CRITIC DIRECTIVE:\n"
            f"Proactively challenge optimism. Look for hidden bottlenecks, cascade failures, security loopholes, "
            f"and unrealistic operational assumptions."
        )

