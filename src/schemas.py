"""Models for the input and output files."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# string -> str, number -> float, integer -> int, boolean -> bool
JsonType = Literal["string", "number", "integer", "boolean"]


class ParameterSpec(BaseModel):
    type: JsonType


class FunctionDefinition(BaseModel):
    """One entry of functions_definition.json."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ParameterSpec

    def ordered_parameters(self) -> list[tuple[str, JsonType]]:
        # keep the order from the file; the output is written in that order
        return [(name, spec.type) for name, spec in self.parameters.items()]


class FunctionCall(BaseModel):
    """One entry of the output file: prompt, name, parameters."""

    prompt: str
    name: str
    parameters: dict[str, Any]
