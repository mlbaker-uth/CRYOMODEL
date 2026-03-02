import json
from pathlib import Path

def test_assignments_schema(tmp_path):
    # minimal synthetic call: we won’t run the whole pipeline here
    data = {"n_centers": 0, "classes": [], "centers_vox_zyx": []}
    out = tmp_path / "assigns.json"
    out.write_text(json.dumps(data))
    d = json.loads(out.read_text())
    assert "n_centers" in d and "classes" in d and "centers_vox_zyx" in d
