from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, field_validator, model_validator


class RegistryLoadError(Exception):
    pass


class FrameworkSource(BaseModel):
    framework_id: str
    name: str
    urls: list[str]
    last_checked_at: Optional[datetime] = None
    last_content_hash: Optional[str] = None

    @field_validator("framework_id", mode="before")
    @classmethod
    def uppercase_id(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def urls_not_empty(self) -> "FrameworkSource":
        if not self.urls:
            raise ValueError("urls must contain at least one entry")
        return self


class FrameworkRegistry:
    def __init__(self, sources: list[FrameworkSource]) -> None:
        seen: set[str] = set()
        for source in sources:
            if source.framework_id in seen:
                raise ValueError(
                    f"Duplicate framework_id: {source.framework_id}"
                )
            seen.add(source.framework_id)
        self._sources: dict[str, FrameworkSource] = {
            s.framework_id: s for s in sources
        }

    def all_sources(self) -> list[FrameworkSource]:
        return list(self._sources.values())

    def get_source(self, framework_id: str) -> FrameworkSource:
        try:
            return self._sources[framework_id.upper()]
        except KeyError:
            raise KeyError(f"Unknown framework_id: {framework_id!r}")

    def update_check_result(
        self,
        framework_id: str,
        *,
        hash_value: str,
        checked_at: datetime,
    ) -> None:
        source = self.get_source(framework_id)  # raises KeyError if unknown
        updated = source.model_copy(
            update={
                "last_content_hash": hash_value,
                "last_checked_at": checked_at,
            }
        )
        self._sources[framework_id.upper()] = updated

    @classmethod
    def from_env(cls) -> "FrameworkRegistry":
        raw = os.environ.get("FRAMEWORK_REGISTRY_JSON")
        if raw is None:
            raise RegistryLoadError(
                "FRAMEWORK_REGISTRY_JSON environment variable is not set"
            )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RegistryLoadError(f"FRAMEWORK_REGISTRY_JSON contains invalid JSON: {exc}") from exc

        sources: list[FrameworkSource] = []
        for i, entry in enumerate(data):
            try:
                sources.append(FrameworkSource(**entry))
            except Exception as exc:
                raise RegistryLoadError(
                    f"Registry entry {i} failed validation: {exc}"
                ) from exc

        return cls(sources)
