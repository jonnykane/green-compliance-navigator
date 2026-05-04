import json
import os
import pytest
from datetime import datetime, timezone
from src.agent.registry import FrameworkSource, FrameworkRegistry, RegistryLoadError


SAMPLE_SOURCES = [
    {
        "framework_id": "SECR",
        "name": "Streamlined Energy and Carbon Reporting",
        "urls": [
            "https://www.gov.uk/guidance/streamlined-energy-and-carbon-reporting"
        ],
    },
    {
        "framework_id": "ESOS",
        "name": "Energy Savings Opportunity Scheme",
        "urls": [
            "https://www.gov.uk/guidance/energy-savings-opportunity-scheme-esos"
        ],
    },
]


class TestFrameworkSource:
    def test_minimal_construction(self):
        source = FrameworkSource(
            framework_id="SECR",
            name="SECR",
            urls=["https://example.com/secr"],
        )
        assert source.framework_id == "SECR"
        assert source.last_checked_at is None
        assert source.last_content_hash is None

    def test_requires_at_least_one_url(self):
        with pytest.raises(ValueError):
            FrameworkSource(framework_id="SECR", name="SECR", urls=[])

    def test_framework_id_uppercased_on_construction(self):
        source = FrameworkSource(
            framework_id="secr", name="SECR", urls=["https://example.com"]
        )
        assert source.framework_id == "SECR"


class TestFrameworkRegistry:
    def _registry(self):
        sources = [FrameworkSource(**s) for s in SAMPLE_SOURCES]
        return FrameworkRegistry(sources)

    def test_all_sources_returns_all(self):
        registry = self._registry()
        assert len(registry.all_sources()) == 2

    def test_get_source_by_id(self):
        registry = self._registry()
        source = registry.get_source("SECR")
        assert source.name == "Streamlined Energy and Carbon Reporting"

    def test_get_source_unknown_id_raises(self):
        registry = self._registry()
        with pytest.raises(KeyError):
            registry.get_source("UNKNOWN")

    def test_update_check_result_persists(self):
        registry = self._registry()
        checked_at = datetime(2026, 5, 4, 12, 0, 0, tzinfo=timezone.utc)
        registry.update_check_result("SECR", hash_value="abc123", checked_at=checked_at)
        source = registry.get_source("SECR")
        assert source.last_content_hash == "abc123"
        assert source.last_checked_at == checked_at

    def test_update_check_result_unknown_id_raises(self):
        registry = self._registry()
        with pytest.raises(KeyError):
            registry.update_check_result("UNKNOWN", hash_value="abc", checked_at=datetime.now())

    def test_duplicate_framework_ids_raise_on_construction(self):
        sources = [FrameworkSource(**SAMPLE_SOURCES[0])] * 2
        with pytest.raises(ValueError, match="Duplicate"):
            FrameworkRegistry(sources)


class TestRegistryFromEnv:
    def test_loads_from_env_json(self, monkeypatch):
        monkeypatch.setenv("FRAMEWORK_REGISTRY_JSON", json.dumps(SAMPLE_SOURCES))
        registry = FrameworkRegistry.from_env()
        assert len(registry.all_sources()) == 2

    def test_raises_when_env_var_missing(self, monkeypatch):
        monkeypatch.delenv("FRAMEWORK_REGISTRY_JSON", raising=False)
        with pytest.raises(RegistryLoadError, match="FRAMEWORK_REGISTRY_JSON"):
            FrameworkRegistry.from_env()

    def test_raises_on_malformed_json(self, monkeypatch):
        monkeypatch.setenv("FRAMEWORK_REGISTRY_JSON", "not-valid-json{")
        with pytest.raises(RegistryLoadError, match="invalid JSON"):
            FrameworkRegistry.from_env()

    def test_raises_on_invalid_source_shape(self, monkeypatch):
        bad = [{"framework_id": "SECR"}]  # missing name and urls
        monkeypatch.setenv("FRAMEWORK_REGISTRY_JSON", json.dumps(bad))
        with pytest.raises(RegistryLoadError, match="validation"):
            FrameworkRegistry.from_env()
