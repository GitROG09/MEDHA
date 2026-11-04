from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from core.llm_client import GeminiClient
from models.schemas import ClaimStatus
from models.task_models import TaskResult
from models.verification_models import Claim
from utils.helpers import generate_id, parse_json_safely
from utils.logger import get_logger

logger = get_logger("claim_extractor")


class ExtractedClaimsSchema(BaseModel):
    """Pydantic schema for structured claim extraction."""

    claims: List[str] = Field(description="Discrete, atomic, verifiable factual or technical assertions")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions underlying the assertions")


class ClaimExtractor:
    """Extracts atomic, testable claims from agent deliverables and task results."""

    def __init__(self, llm_client: Optional[GeminiClient] = None):
        self.client = llm_client or GeminiClient.get_instance()

    async def extract_claims_from_result(
        self,
        result: TaskResult,
        task_id: str = "",
    ) -> List[Claim]:
        """Extract claims from a completed TaskResult, using already identified claims or LLM extraction."""
        claims: List[Claim] = []

        # If the agent already identified discrete claims, wrap them directly
        if result.claims:
            for claim_text in result.claims:
                text_clean = claim_text.strip()
                if text_clean:
                    claims.append(
                        Claim(
                            claim_id=generate_id("claim"),
                            claim_text=text_clean,
                            originating_agent=result.agent_id,
                            originating_role=result.role,
                            task_id=task_id or result.task_id,
                            subtask_id=result.subtask_id,
                            supporting_reasoning=result.output[:300],
                            assumptions=result.assumptions,
                            status=ClaimStatus.PENDING,
                        )
                    )
            logger.info(f"Adopted {len(claims)} pre-identified claims from [{result.role}]")
            return claims

        # Otherwise, extract atomic claims from the deliverable content using LLM
        if not result.output:
            return []

        prompt = (
            f"Extract discrete, verifiable factual, performance, or architectural claims from this text:\n\n"
            f"TEXT:\n{result.output}\n\n"
            f"Extract 2 to 6 atomic, testable assertions that could be empirically true or false. "
            f"Separate established statements from acknowledged assumptions."
        )

        try:
            extracted = await self.client.generate_structured(
                prompt=prompt,
                response_schema=ExtractedClaimsSchema,
                temperature=0.1,
            )
            for c_text in extracted.claims:
                claims.append(
                    Claim(
                        claim_id=generate_id("claim"),
                        claim_text=c_text,
                        originating_agent=result.agent_id,
                        originating_role=result.role,
                        task_id=task_id or result.task_id,
                        subtask_id=result.subtask_id,
                        supporting_reasoning=result.output[:300],
                        assumptions=extracted.assumptions or result.assumptions,
                        status=ClaimStatus.PENDING,
                    )
                )
        except Exception as e:
            logger.warning(f"Structured claim extraction failed ({e}), using line splitting fallback...")
            # Fallback heuristic: extract non-empty lines that state assertions
            for line in result.output.split("\n"):
                line = line.strip().lstrip("-*123456789. ")
                if len(line) > 25 and not line.endswith("?"):
                    claims.append(
                        Claim(
                            claim_id=generate_id("claim"),
                            claim_text=line,
                            originating_agent=result.agent_id,
                            originating_role=result.role,
                            task_id=task_id or result.task_id,
                            subtask_id=result.subtask_id,
                            supporting_reasoning=result.output[:300],
                            assumptions=result.assumptions,
                            status=ClaimStatus.PENDING,
                        )
                    )
                    if len(claims) >= 4:
                        break

        logger.info(f"Extracted {len(claims)} claims from [{result.role}]")
        return claims

