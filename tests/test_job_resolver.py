import pytest

from cryomodel.workflow.resolver import resolve_command


@pytest.fixture
def specs():
    return {
        "dnaaxis_extract": {
            "command": {
                "template": (
                    "cryomodel dnaaxis extract --map {map} --threshold {threshold} "
                    "--out-pdb {out_pdb} --out-mrc {out_mrc} {guides_arg}"
                )
            },
            "inputs": [
                {"id": "map", "required": True, "artifact_type": "map.mrc"},
                {"id": "guides_pdb", "required": False, "artifact_type": "model.structure"},
            ],
            "params": [
                {"id": "threshold", "type": "float", "required": True, "default": 0.25, "min": 0.0, "max": 10.0},
            ],
            "outputs": [
                {"id": "out_pdb", "default": "outputs/dnaaxis/dna_axis.pdb"},
                {"id": "out_mrc", "default": "outputs/dnaaxis/dna_axis.mrc"},
            ],
            "arg_builders": {
                "guides_arg": {
                    "when_input_present": "guides_pdb",
                    "value_template": "--guides-pdb {guides_pdb}",
                }
            },
        },
        "dnabuild_build": {
            "command": {
                "template": (
                    "cryomodel dnabuild build-2bp --centerline-pdb {centerline_pdb} "
                    "--template-2bp-pdb {template_2bp_pdb} --map {map} "
                    "--out-pdb {out_pdb} --target-spacing {target_spacing} {threshold_arg}"
                )
            },
            "inputs": [
                {"id": "centerline_pdb", "required": True, "artifact_type": "model.structure"},
                {"id": "map", "required": True, "artifact_type": "map.mrc"},
            ],
            "params": [
                {
                    "id": "template_2bp_pdb",
                    "type": "string",
                    "required": True,
                    "default": "data/DNA-TEMPLATES/2AT-template.pdb",
                },
                {"id": "target_spacing", "type": "float", "required": True, "default": 3.4, "min": 2.0, "max": 5.0},
                {"id": "threshold", "type": "float", "required": False, "default": None, "min": 0.0, "max": 10.0},
            ],
            "outputs": [{"id": "out_pdb", "default": "outputs/dnabuild/dna_initial.pdb"}],
            "arg_builders": {
                "threshold_arg": {
                    "when_input_present": "threshold",
                    "value_template": "--threshold {threshold}",
                }
            },
        },
        "basehunter_run": {
            "command": {
                "template": (
                    "cryomodel basehunter --map {map} --model {model} "
                    "--out-dir {out_dir} --resolution {resolution} {chain_arg}"
                )
            },
            "inputs": [
                {"id": "map", "required": True, "artifact_type": "map.mrc"},
                {"id": "model", "required": True, "artifact_type": "model.structure"},
            ],
            "params": [
                {"id": "resolution", "type": "float", "required": True, "default": 3.0, "min": 0.5, "max": 20.0},
                {"id": "chain", "type": "string", "required": False, "default": None},
            ],
            "outputs": [{"id": "out_dir", "default": "outputs/basehunter"}],
            "arg_builders": {
                "chain_arg": {
                    "when_input_present": "chain",
                    "value_template": "--chain {chain}",
                }
            },
        },
    }


def _workspace_with_cards():
    return {"cards": []}


def _has_error(result, code, field):
    return any(e.get("code") == code and e.get("field") == field for e in result.errors)


def test_01_dnaaxis_happy_path(specs):
    card = {
        "card_id": "card_a",
        "run_state": "draft",
        "inputs": {"map": {"mode": "manual", "value": "/data/map.mrc"}},
        "params": {"threshold": 0.25},
        "outputs": {"resolved": {}, "types": {"out_pdb": "model.structure", "out_mrc": "map.mrc"}},
    }
    result = resolve_command(card, specs["dnaaxis_extract"], _workspace_with_cards())
    assert result.ok
    assert "cryomodel dnaaxis extract" in result.command
    assert "--map /data/map.mrc" in result.command


def test_02_missing_required_input(specs):
    card = {"card_id": "card_a", "run_state": "draft", "inputs": {}, "params": {"threshold": 0.25}}
    result = resolve_command(card, specs["dnaaxis_extract"], _workspace_with_cards())
    assert not result.ok
    assert _has_error(result, "MISSING_REQUIRED_INPUT", "map")


def test_03_param_type_error(specs):
    card = {
        "card_id": "card_a",
        "run_state": "draft",
        "inputs": {"map": {"mode": "manual", "value": "/data/map.mrc"}},
        "params": {"threshold": "abc"},
    }
    result = resolve_command(card, specs["dnaaxis_extract"], _workspace_with_cards())
    assert not result.ok
    assert _has_error(result, "INVALID_PARAM_TYPE", "threshold")


def test_04_param_range_error(specs):
    card = {
        "card_id": "card_a",
        "run_state": "draft",
        "inputs": {"map": {"mode": "manual", "value": "/data/map.mrc"}},
        "params": {"threshold": -1.0},
    }
    result = resolve_command(card, specs["dnaaxis_extract"], _workspace_with_cards())
    assert not result.ok
    assert _has_error(result, "PARAM_OUT_OF_RANGE", "threshold")


def test_05_optional_arg_emission(specs):
    card = {
        "card_id": "card_a",
        "run_state": "draft",
        "inputs": {
            "map": {"mode": "manual", "value": "/data/map.mrc"},
            "guides_pdb": {"mode": "manual", "value": "/data/guides.pdb"},
        },
        "params": {"threshold": 0.25},
    }
    result = resolve_command(card, specs["dnaaxis_extract"], _workspace_with_cards())
    assert result.ok
    assert "--guides-pdb /data/guides.pdb" in result.command


def test_06_optional_arg_suppressed(specs):
    card = {
        "card_id": "card_a",
        "run_state": "draft",
        "inputs": {"map": {"mode": "manual", "value": "/data/map.mrc"}},
        "params": {"threshold": 0.25},
    }
    result = resolve_command(card, specs["dnaaxis_extract"], _workspace_with_cards())
    assert result.ok
    assert "--guides-pdb" not in result.command
    assert "  " not in result.command


def test_07_inherit_success(specs):
    src_card = {
        "card_id": "card_a",
        "run_state": "success",
        "outputs": {
            "resolved": {"out_pdb": "outputs/dnaaxis/dna_axis.pdb"},
            "types": {"out_pdb": "model.structure"},
        },
    }
    dst_card = {
        "card_id": "card_b",
        "run_state": "draft",
        "inputs": {
            "centerline_pdb": {
                "mode": "inherited",
                "source": {"card_id": "card_a", "output_id": "out_pdb"},
            },
            "map": {"mode": "manual", "value": "/data/map.mrc"},
        },
        "params": {"target_spacing": 3.4},
    }
    ws = {"cards": [src_card, dst_card]}
    result = resolve_command(dst_card, specs["dnabuild_build"], ws)
    assert result.ok
    assert "dnabuild build-2bp" in result.command
    assert "--centerline-pdb outputs/dnaaxis/dna_axis.pdb" in result.command


def test_08_inherit_source_not_ready(specs):
    src_card = {
        "card_id": "card_a",
        "run_state": "running",
        "outputs": {
            "resolved": {"out_pdb": "outputs/dnaaxis/dna_axis.pdb"},
            "types": {"out_pdb": "model.structure"},
        },
    }
    dst_card = {
        "card_id": "card_b",
        "run_state": "draft",
        "inputs": {
            "centerline_pdb": {
                "mode": "inherited",
                "source": {"card_id": "card_a", "output_id": "out_pdb"},
            },
            "map": {"mode": "manual", "value": "/data/map.mrc"},
        },
        "params": {"target_spacing": 3.4},
    }
    ws = {"cards": [src_card, dst_card]}
    result = resolve_command(dst_card, specs["dnabuild_build"], ws)
    assert not result.ok
    assert _has_error(result, "INHERIT_SOURCE_NOT_READY", "centerline_pdb")


def test_09_inherit_type_mismatch(specs):
    src_card = {
        "card_id": "card_a",
        "run_state": "success",
        "outputs": {
            "resolved": {"out_pdb": "outputs/dnaaxis/dna_axis.pdb"},
            "types": {"out_pdb": "map.mrc"},
        },
    }
    dst_card = {
        "card_id": "card_b",
        "run_state": "draft",
        "inputs": {
            "centerline_pdb": {
                "mode": "inherited",
                "source": {"card_id": "card_a", "output_id": "out_pdb"},
            },
            "map": {"mode": "manual", "value": "/data/map.mrc"},
        },
        "params": {"target_spacing": 3.4},
    }
    ws = {"cards": [src_card, dst_card]}
    result = resolve_command(dst_card, specs["dnabuild_build"], ws)
    assert not result.ok
    assert _has_error(result, "INHERIT_TYPE_MISMATCH", "centerline_pdb")


def test_10_end_to_end_chain(specs):
    card_a = {
        "card_id": "card_a",
        "run_state": "success",
        "outputs": {
            "resolved": {"out_pdb": "outputs/dnaaxis/dna_axis.pdb"},
            "types": {"out_pdb": "model.structure"},
        },
    }
    card_b = {
        "card_id": "card_b",
        "run_state": "success",
        "outputs": {
            "resolved": {"out_pdb": "outputs/dnabuild/dna_initial.pdb"},
            "types": {"out_pdb": "model.structure"},
        },
    }
    card_c = {
        "card_id": "card_c",
        "run_state": "draft",
        "inputs": {
            "map": {"mode": "manual", "value": "/data/map.mrc"},
            "model": {"mode": "inherited", "source": {"card_id": "card_b", "output_id": "out_pdb"}},
        },
        "params": {"resolution": 3.0},
    }
    ws = {"cards": [card_a, card_b, card_c]}
    result = resolve_command(card_c, specs["basehunter_run"], ws)
    assert result.ok
    assert "--model outputs/dnabuild/dna_initial.pdb" in result.command

