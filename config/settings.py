import os
from pathlib import Path

from dotenv import load_dotenv

# Locate the MEDHA project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from the root .env file
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# Retry & Rate Limit Configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
INITIAL_RETRY_DELAY = float(os.getenv("INITIAL_RETRY_DELAY", "1.0"))
MAX_RETRY_DELAY = float(os.getenv("MAX_RETRY_DELAY", "10.0"))

# Multi-Agent Execution Configuration
MAX_CONCURRENT_AGENTS = int(os.getenv("MAX_CONCURRENT_AGENTS", "4"))
MAX_REPAIR_ITERATIONS = int(os.getenv("MAX_REPAIR_ITERATIONS", "2"))
EXECUTION_TIMEOUT_SECONDS = float(os.getenv("EXECUTION_TIMEOUT_SECONDS", "180.0"))

# Application & Logging Configuration
APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOGS_DIR = BASE_DIR / "logs"
BENCHMARK_DIR = BASE_DIR / "benchmark"

# Ensure directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def validate_settings() -> bool:
    """Validate essential MEDHA configuration."""
    if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing from .env")

    if not GEMINI_MODEL:
        raise ValueError("GEMINI_MODEL is not configured.")

    if MAX_CONCURRENT_AGENTS < 1:
        raise ValueError("MAX_CONCURRENT_AGENTS must be at least 1.")

    if MAX_REPAIR_ITERATIONS < 0:
        raise ValueError("MAX_REPAIR_ITERATIONS cannot be negative.")

    return True
