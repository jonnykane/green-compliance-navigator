import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


class RegistryLoadError(Exception):
    pass


@dataclass
class FrameworkSource:
    framework_id: str
    name: str
    urls: list[str]
    last_checked_at: Optional[datetime] = field(default=None)
    last_content_hash: Optional[str] = field(default=None)

    def __post_init__(self):
        self.framework_id = self.framework_id.upper()
        if not self.urls:
            raise ValueError("urls must contain at least one entry")


class FrameworkRegistry:
    def __init__(self, sources: list[FrameworkSource]):
        self._sources: dict[str, FrameworkSource] = {}
        for source in sources:
            if source.framework_id in self._sources:
                raise ValueError(f"Duplicate framework_id: {source.framework_id}")
            self._sources[source.framework_id] = source

    def all_sources(self) -> list[FrameworkSource]:
        return list(self._sources.values())

    def get_source(self, framework_id: str) -> FrameworkSource:
        if framework_id not in self._sources:
            raise KeyError(framework_id)
        return self._sources[framework_id]

    def update_check_result(self, framework_id: str, *, hash_value: str, checked_at: datetime) -> None:
        source = self.get_source(framework_id)
        source.last_content_hash = hash_value
        source.last_checked_at = checked_at

    @classmethod
    def from_env(cls) -> "FrameworkRegistry":
        raw = os.environ.get("FRAMEWORK_REGISTRY_JSON")
        if raw is None:
            raise RegistryLoadError("FRAMEWORK_REGISTRY_JSON environment variable is not set")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RegistryLoadError(f"invalid JSON in FRAMEWORK_REGISTRY_JSON: {exc}") from exc

        sources = []
        for entry in data:
            try:
                sources.append(FrameworkSource(**entry))
            except (TypeError, ValueError) as exc:
                raise RegistryLoadError(f"validation error in registry entry: {exc}") from exc

        return cls(sources)
