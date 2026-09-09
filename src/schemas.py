"""Models for the input and output files."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# JSON Schema type names as they appear in
# functions_definition.json. Each maps to the
# corresponding Python type:
#
#   string  -> str
#   number  -> float
#   integer -> int
#   boolean -> bool
JsonType = Literal[
    "string",
    "number",
    "integer",
    "boolean",
]


class ParameterSpec(BaseModel):
    """The declared JSON type of a parameter or return value."""

    type: JsonType


class FunctionDefinition(BaseModel):
    """One entry of ``functions_definition.json``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ParameterSpec

    def ordered_parameters(self) -> list[tuple[str, JsonType]]:
        """Return parameters as ``(name, type)`` pairs.

        The declaration order is preserved because Python dictionaries
        maintain insertion order. The decoder emits arguments using this
        same order.
        """
        return [
            (name, spec.type)
            for name, spec in self.parameters.items()
        ]


class FunctionCall(BaseModel):
    """One entry of the output file."""

    prompt: str
    name: str
    parameters: dict[str, Any]
