# -*- coding: utf-8 -*-
from typing import Any

from pydantic import BaseModel
from sinapsis_core.utils.env_var_keys import EnvVarEntry, doc_str, return_docs_for_vars


class _DiarizationEnvVars(BaseModel):
    """Env vars for HuggingFace."""

    HF_TOKEN: EnvVarEntry = EnvVarEntry(
        var_name="HF_TOKEN",
        default_value=" ",
        allowed_values=None,
        description="set api key for HuggingFace API",
    )


DiarizationEnvVars = _DiarizationEnvVars()

doc_str = return_docs_for_vars(DiarizationEnvVars, docs=doc_str, string_for_doc="""Env vars available: \n""")
__doc__ = doc_str


def __getattr__(name: str) -> Any:
    """Allows accessing environment variable default values directly as module attributes.

    Args:
        name (str): The name of the environment variable.

    Raises:
        AttributeError: If the requested attribute `name` is not a defined
                        environment variable.

    Returns:
        Any: The default value of the requested environment variable.
    """
    if name in DiarizationEnvVars.model_fields:
        return DiarizationEnvVars.model_fields[name].default.value

    raise AttributeError(f"Agent does not have `{name}` env var")


_all__ = (*list(DiarizationEnvVars.model_fields.keys()), "DiarizationEnvVars")
