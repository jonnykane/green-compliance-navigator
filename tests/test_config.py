import os
import importlib
from pathlib import Path
from unittest.mock import patch


def test_defaults_without_env(monkeypatch):
    """Config falls back to hard-coded defaults when no env vars and no .env file."""
    all_keys = [
        "CORPUS_REAL_DIR", "CORPUS_MOCK_DIR", "CORPUS_METADATA_DIR",
        "CHROMA_PERSIST_DIR", "CHROMA_COLLECTION_NAME",
        "VOYAGE_API_KEY", "VOYAGE_MODEL", "VOYAGE_BATCH_SIZE", "VOYAGE_REQUEST_INTERVAL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
        "CHUNK_MAX_TOKENS", "CHUNK_OVERLAP_TOKENS",
    ]
    for key in all_keys:
        monkeypatch.delenv(key, raising=False)

    # Suppress .env file loading so project-local overrides don't interfere.
    # Must patch the source (dotenv.load_dotenv) because reload re-imports the name.
    with patch("dotenv.load_dotenv"):
        import src.config.settings as settings
        importlib.reload(settings)

    assert settings.CORPUS_REAL_DIR.name == "real"
    assert settings.CHROMA_COLLECTION_NAME == "green_compliance"
    assert settings.VOYAGE_MODEL == "voyage-3"
    assert settings.VOYAGE_BATCH_SIZE == 128
    assert settings.VOYAGE_REQUEST_INTERVAL == 0.0
    assert settings.CHUNK_MAX_TOKENS == 512
    assert settings.CHUNK_OVERLAP_TOKENS == 64


def test_env_overrides(monkeypatch):
    """Env vars are respected when set."""
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "test_collection")
    monkeypatch.setenv("VOYAGE_MODEL", "voyage-2")
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "256")

    import src.config.settings as settings
    importlib.reload(settings)

    assert settings.CHROMA_COLLECTION_NAME == "test_collection"
    assert settings.VOYAGE_MODEL == "voyage-2"
    assert settings.CHUNK_MAX_TOKENS == 256


def test_paths_are_path_objects(monkeypatch):
    import src.config.settings as settings
    importlib.reload(settings)

    assert isinstance(settings.CORPUS_REAL_DIR, Path)
    assert isinstance(settings.CORPUS_MOCK_DIR, Path)
    assert isinstance(settings.CHROMA_PERSIST_DIR, Path)
