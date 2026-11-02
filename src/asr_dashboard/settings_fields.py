"""Declarative setting descriptions. The UI builds forms from these (see ui/widgets/schema_form.py)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

FieldType = Literal["int", "float", "bool", "enum", "str", "text", "path", "secret", "json"]


@dataclass
class SettingField:
    key: str  # dotted key relative to the owning settings dict, e.g. "aligner.enabled"
    label: str
    type: FieldType
    default: Any = None
    description: str = ""
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    # enum choices: list of (value, label)
    choices: list[tuple[Any, str]] = field(default_factory=list)
    # condition helpers: {"key": value} or {"key": [v1, v2]}; all must match
    visible_if: dict[str, Any] | None = None
    enabled_if: dict[str, Any] | None = None
    disabled_reason: str = ""  # tooltip shown when disabled by availability
    special_values: dict[Any, str] = field(default_factory=dict)  # e.g. {-1: "不限"}
    group: str = ""
    advanced: bool = False

    def validate(self, value: Any) -> Any:
        """Coerce and validate a value; raises ValueError on invalid input."""
        if self.type == "int":
            v = int(value)
            if v in self.special_values:
                return v
            self._check_range(v)
            return v
        if self.type == "float":
            v = float(value)
            self._check_range(v)
            return v
        if self.type == "bool":
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if self.type == "enum":
            allowed = [c[0] for c in self.choices]
            if allowed and value not in allowed:
                raise ValueError(f"{self.label}: {value!r} 不在可选范围 {allowed}")
            return value
        if self.type == "json":
            import json

            if isinstance(value, str):
                return json.loads(value) if value.strip() else {}
            return value
        return "" if value is None else str(value)

    def _check_range(self, v: float) -> None:
        if self.minimum is not None and v < self.minimum:
            raise ValueError(f"{self.label}: {v} 小于最小值 {self.minimum}")
        if self.maximum is not None and v > self.maximum:
            raise ValueError(f"{self.label}: {v} 大于最大值 {self.maximum}")


def get_dotted(d: dict, key: str, default: Any = None) -> Any:
    cur: Any = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_dotted(d: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def condition_met(cond: dict[str, Any] | None, values: dict) -> bool:
    if not cond:
        return True
    for k, expected in cond.items():
        actual = get_dotted(values, k)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True
