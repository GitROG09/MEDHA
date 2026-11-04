import pytest

import config.settings as settings


def test_settings_load_defaults():
    """Verify default settings values are correctly loaded."""
    assert settings.LLM_PROVIDER == "gemini"
    assert settings.GEMINI_MODEL is not None
    assert settings.MAX_CONCURRENT_AGENTS >= 1
    assert settings.MAX_REPAIR_ITERATIONS >= 0
    assert settings.LOGS_DIR.exists()


def test_validate_settings_success(monkeypatch):
    """Verify settings validation succeeds when API key and model are present."""
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "dummy_test_key_123")
    monkeypatch.setattr(settings, "GEMINI_MODEL", "gemini-3.8-flash")
    assert settings.validate_settings() is True


def test_validate_settings_missing_api_key(monkeypatch):
    """Verify validation fails when GEMINI_API_KEY is missing."""
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)
    with pytest.raises(ValueError, match="GEMINI_API_KEY is missing"):
        settings.validate_settings()


def test_validate_settings_missing_model(monkeypatch):
    """Verify validation fails when GEMINI_MODEL is missing."""
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "dummy_test_key_123")
    monkeypatch.setattr(settings, "GEMINI_MODEL", "")
    with pytest.raises(ValueError, match="GEMINI_MODEL is not configured"):
        settings.validate_settings()


def test_validate_settings_invalid_concurrency(monkeypatch):
    """Verify validation fails when MAX_CONCURRENT_AGENTS < 1."""
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "dummy_test_key_123")
    monkeypatch.setattr(settings, "MAX_CONCURRENT_AGENTS", 0)
    with pytest.raises(ValueError, match="MAX_CONCURRENT_AGENTS must be at least 1"):
        settings.validate_settings()


def test_validate_settings_invalid_repair_iterations(monkeypatch):
    """Verify validation fails when MAX_REPAIR_ITERATIONS < 0."""
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "dummy_test_key_123")
    monkeypatch.setattr(settings, "MAX_REPAIR_ITERATIONS", -1)
    with pytest.raises(ValueError, match="MAX_REPAIR_ITERATIONS cannot be negative"):
        settings.validate_settings()

