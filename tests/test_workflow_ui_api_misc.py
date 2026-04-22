"""Small tests for workflow UI API helpers (no httpx TestClient)."""

from cryomodel.workflow import ui_api


def test_ui_home_dir_returns_path() -> None:
    out = ui_api.ui_home_dir()
    assert "home_dir" in out
    assert isinstance(out["home_dir"], str)
    assert len(out["home_dir"]) > 0
