from typing import Any, Optional
from agents.specialist_agent import SpecialistAgent
from models.agent_models import AgentConfig


class AnalystAgent(SpecialistAgent):
    """Specialist agent focused on quantitative reasoning, bottleneck analysis, and tradeoff evaluation."""

    def __init__(self, config: AgentConfig, llm_client: Optional[Any] = None):
        if not config.role or config.role == "Specialist":
            config.role = "Systems Analyst"
        super().__init__(config, llm_client=llm_client)

    def _build_system_instruction(self) -> str:
        base = super()._build_system_instruction()
        return (
            f"{base}\n\n"
            f"ANALYST DIRECTIVE:\n"
            f"Decompose the problem quantitatively where feasible. Quantify throughput, latency, costs, "
            f"scaling ceilings, and risk factors. Scrutinize hidden interdependencies."
        )

