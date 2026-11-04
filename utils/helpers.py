import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict


def generate_id(prefix: str = "id") -> str:
    """Generate a clean, readable unique identifier with a given prefix."""
    short_uuid = uuid.uuid4().hex[:10]
    return f"{prefix}_{short_uuid}"


def utcnow_iso() -> str:
    """Return the current UTC timestamp formatted as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def sanitize_json_text(raw_text: str) -> str:
    """Strip markdown code fence blocks from text to extract raw JSON."""
    text = raw_text.strip()
    if text.startswith("```"):
        # Match ```json or ``` at start and ``` at end
        pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
        match = re.match(pattern, text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    return text


def parse_json_safely(raw_text: str) -> Dict[str, Any]:
    """Parse JSON text safely, handling optional markdown code fences and whitespace."""
    clean_text = sanitize_json_text(raw_text)
    return json.loads(clean_text)

