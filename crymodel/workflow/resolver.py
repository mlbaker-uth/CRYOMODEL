"""Template resolver for UI-style workflow job cards."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re
import shlex


@dataclass
class ResolveResult:
    ok: bool
    command: Optional[str] = None
    resolved_inputs: Dict[str, Any] = field(default_factory=dict)
    resolved_params: Dict[str, Any] = field(default_factory=dict)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)


def _error(code: str, field: str, message: str) -> Dict[str, str]:
    return {"code": code, "field": field, "message": message}


def _quote_if_path(value: Any) -> str:
    if isinstance(value, str) and ("/" in value or "." in value):
        return shlex.quote(value)
    return str(value)


def _cast_and_validate_param(param_spec: Dict[str, Any], raw: Any) -> tuple[Any, Optional[Dict[str, str]]]:
    ptype = param_spec.get("type", "string")
    pid = param_spec["id"]
    try:
        if ptype == "int":
            val = int(raw)
        elif ptype == "float":
            val = float(raw)
        elif ptype == "bool":
            if isinstance(raw, bool):
                val = raw
            elif str(raw).lower() in {"true", "1", "yes"}:
                val = True
            elif str(raw).lower() in {"false", "0", "no"}:
                val = False
            else:
                raise ValueError("invalid boolean")
        else:
            val = str(raw)
    except Exception:
        return None, _error("INVALID_PARAM_TYPE", pid, f"Invalid value for parameter '{pid}'.")

    pmin = param_spec.get("min")
    pmax = param_spec.get("max")
    if isinstance(val, (int, float)):
        if pmin is not None and val < pmin:
            return None, _error("PARAM_OUT_OF_RANGE", pid, f"Parameter '{pid}' is below minimum {pmin}.")
        if pmax is not None and val > pmax:
            return None, _error("PARAM_OUT_OF_RANGE", pid, f"Parameter '{pid}' is above maximum {pmax}.")

    return val, None


def _resolve_input(
    card: Dict[str, Any],
    input_spec: Dict[str, Any],
    workspace: Dict[str, Any],
) -> tuple[Any, Optional[Dict[str, str]]]:
    iid = input_spec["id"]
    binding = (card.get("inputs") or {}).get(iid, {})
    mode = binding.get("mode", "manual")

    if mode == "manual":
        val = binding.get("value")
        if input_spec.get("required", False) and (val is None or str(val).strip() == ""):
            return None, _error("MISSING_REQUIRED_INPUT", iid, f"Required input '{iid}' is missing.")
        return val, None

    if mode == "inherited":
        src = binding.get("source") or {}
        src_card_id = src.get("card_id")
        src_output_id = src.get("output_id")
        if not src_card_id or not src_output_id:
            return None, _error("INHERIT_SOURCE_NOT_FOUND", iid, f"Inherited input '{iid}' has no source.")

        cards = workspace.get("cards") or []
        src_card = next((c for c in cards if c.get("card_id") == src_card_id), None)
        if src_card is None:
            return None, _error("INHERIT_SOURCE_NOT_FOUND", iid, f"Source card for '{iid}' not found.")
        if src_card.get("run_state") != "success":
            return None, _error("INHERIT_SOURCE_NOT_READY", iid, f"Source card for '{iid}' is not ready.")

        src_output = ((src_card.get("outputs") or {}).get("resolved") or {}).get(src_output_id)
        if not src_output:
            return None, _error("INHERIT_SOURCE_NOT_FOUND", iid, f"Source output '{src_output_id}' is missing.")

        in_type = input_spec.get("artifact_type")
        out_type = ((src_card.get("outputs") or {}).get("types") or {}).get(src_output_id)
        if in_type and out_type and in_type != out_type:
            return None, _error("INHERIT_TYPE_MISMATCH", iid, f"Type mismatch for inherited input '{iid}'.")
        return src_output, None

    return None, _error("MISSING_REQUIRED_INPUT", iid, f"Unsupported binding mode for '{iid}'.")


def resolve_command(card: Dict[str, Any], spec: Dict[str, Any], workspace: Dict[str, Any]) -> ResolveResult:
    ctx: Dict[str, Any] = {}
    resolved_inputs: Dict[str, Any] = {}
    resolved_params: Dict[str, Any] = {}
    errors: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []

    # Inputs
    for input_spec in spec.get("inputs", []):
        val, err = _resolve_input(card, input_spec, workspace)
        if err:
            errors.append(err)
            continue
        if val is not None:
            resolved_inputs[input_spec["id"]] = val
            ctx[input_spec["id"]] = val

    # Params
    card_params = card.get("params") or {}
    for param_spec in spec.get("params", []):
        pid = param_spec["id"]
        raw = card_params.get(pid, param_spec.get("default"))
        if param_spec.get("required", False) and raw is None:
            errors.append(_error("MISSING_REQUIRED_PARAM", pid, f"Required parameter '{pid}' is missing."))
            continue
        if raw is None:
            continue
        val, err = _cast_and_validate_param(param_spec, raw)
        if err:
            errors.append(err)
            continue
        resolved_params[pid] = val
        ctx[pid] = val

    # Runtime defaults from outputs (out paths etc.)
    for out in spec.get("outputs", []):
        oid = out.get("id")
        default = out.get("default")
        if oid and default and oid not in ctx:
            ctx[oid] = default

    # Optional arg builders
    for name, builder in (spec.get("arg_builders") or {}).items():
        when_input = builder.get("when_input_present")
        template = builder.get("value_template", "")
        if when_input and ctx.get(when_input):
            rendered = template
            for k, v in ctx.items():
                rendered = rendered.replace("{" + k + "}", _quote_if_path(v))
            ctx[name] = rendered
        else:
            ctx[name] = ""

    if errors:
        return ResolveResult(
            ok=False,
            resolved_inputs=resolved_inputs,
            resolved_params=resolved_params,
            warnings=warnings,
            errors=errors,
        )

    template = (spec.get("command") or {}).get("template", "")
    cmd = template
    for k, v in ctx.items():
        cmd = cmd.replace("{" + k + "}", _quote_if_path(v))

    # unresolved tokens
    unresolved = re.findall(r"\{[^{}]+\}", cmd)
    if unresolved:
        for tok in unresolved:
            errors.append(_error("TEMPLATE_TOKEN_UNRESOLVED", tok, f"Unresolved token {tok}"))
        return ResolveResult(
            ok=False,
            resolved_inputs=resolved_inputs,
            resolved_params=resolved_params,
            warnings=warnings,
            errors=errors,
        )

    cmd = " ".join(cmd.split())
    return ResolveResult(
        ok=True,
        command=cmd,
        resolved_inputs=resolved_inputs,
        resolved_params=resolved_params,
        warnings=warnings,
        errors=[],
    )

