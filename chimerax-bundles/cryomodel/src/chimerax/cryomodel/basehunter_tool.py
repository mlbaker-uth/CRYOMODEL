"""ChimeraX scaffold tool for BaseHunter Interactive (Phase 2 mock workflow)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple

from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from Qt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cryomodel.nucleotide.interactive_pairs import Marker3D, PairResult, PairState
from cryomodel.nucleotide.template_registry import validate_template_pack
from cryomodel.resources import basehunter_template_pack_dir


def _disk_path_for_model(m) -> Optional[str]:
    """Best-effort file path extraction for an open ChimeraX model/volume."""
    def _check_path(v) -> Optional[str]:
        if not v:
            return None
        p = Path(str(v))
        return str(p.resolve()) if p.is_file() else None

    for owner in (getattr(m, "opened_data", None), getattr(m, "data", None), m):
        if owner is None:
            continue
        for attr in ("path", "filename", "file_name"):
            got = _check_path(getattr(owner, attr, None))
            if got:
                return got
    return None


def _model_kind(m) -> str:
    cls = getattr(m, "__class__", type(m)).__name__
    if "Volume" in cls:
        return "map"
    if "Structure" in cls:
        return "structure"
    return "other"


class BaseHunterInteractiveTool(ToolInstance):
    """Phase 2 UI scaffold with mocked compute/build state machine."""

    SESSION_ENDURING = True

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        self.tool_window = MainToolWindow(self)
        self.tool_window.fill_context_menu = True
        self._map_models: List[object] = []
        self._structure_models: List[object] = []
        self._state = PairState()
        self._build_ui()
        self._refresh_models()
        self._sync_pair_table()
        self.tool_window.manage(None)

    def _build_ui(self) -> None:
        layout = QVBoxLayout()

        # Data
        data_group = QGroupBox("Data")
        data_layout = QVBoxLayout()
        map_row = QHBoxLayout()
        map_row.addWidget(QLabel("Map"))
        self.map_combo = QComboBox()
        self.refresh_btn = QPushButton("Refresh models")
        self.refresh_btn.clicked.connect(self._refresh_models)
        map_row.addWidget(self.map_combo)
        map_row.addWidget(self.refresh_btn)
        data_layout.addLayout(map_row)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Working model"))
        self.model_combo = QComboBox()
        model_row.addWidget(self.model_combo)
        data_layout.addLayout(model_row)
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # Threshold + template pack
        tmpl_group = QGroupBox("Threshold + templates")
        tmpl_layout = QVBoxLayout()
        tr = QHBoxLayout()
        self.inherit_thresh = QCheckBox("Use Volume Viewer threshold")
        self.inherit_thresh.setChecked(True)
        self.threshold_edit = QLineEdit("0.30")
        self.threshold_edit.setEnabled(False)
        self.inherit_thresh.toggled.connect(self._toggle_threshold_mode)
        tr.addWidget(self.inherit_thresh)
        tr.addWidget(QLabel("Value"))
        tr.addWidget(self.threshold_edit)
        sync_btn = QPushButton("Sync from viewer")
        sync_btn.clicked.connect(self._sync_threshold_from_viewer)
        tr.addWidget(sync_btn)
        tmpl_layout.addLayout(tr)

        path_row = QHBoxLayout()
        self.template_dir = QLineEdit()
        self.template_dir.setText(str(basehunter_template_pack_dir()))
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_template_dir)
        path_row.addWidget(self.template_dir)
        path_row.addWidget(browse)
        tmpl_layout.addLayout(path_row)

        btn_row = QHBoxLayout()
        validate_btn = QPushButton("Validate templates")
        validate_btn.clicked.connect(self._validate_templates)
        btn_row.addWidget(validate_btn)
        tmpl_layout.addLayout(btn_row)
        tmpl_group.setLayout(tmpl_layout)
        layout.addWidget(tmpl_group)

        # Pair markers
        markers_group = QGroupBox("Pair markers")
        markers_layout = QVBoxLayout()
        mode_row = QHBoxLayout()
        self.mode_buttons: List[Tuple[str, QPushButton]] = []
        for key, label in (
            ("place", "Place marker"),
            ("move", "Move marker"),
            ("select", "Select marker"),
            ("delete", "Delete marker"),
        ):
            b = QPushButton(label)
            b.setCheckable(True)
            b.clicked.connect(lambda checked, k=key: self._set_marker_mode(k))
            mode_row.addWidget(b)
            self.mode_buttons.append((key, b))
        markers_layout.addLayout(mode_row)

        pair_btn_row = QHBoxLayout()
        add_pair_btn = QPushButton("New pair")
        add_pair_btn.clicked.connect(self._new_pair)
        pair_btn_row.addWidget(add_pair_btn)
        add_sel_btn = QPushButton("Add marker from selection")
        add_sel_btn.clicked.connect(self._add_marker_from_selection)
        pair_btn_row.addWidget(add_sel_btn)
        clear_sel_btn = QPushButton("Clear selected pair")
        clear_sel_btn.clicked.connect(self._clear_selected_pair)
        pair_btn_row.addWidget(clear_sel_btn)
        clear_all_btn = QPushButton("Clear all")
        clear_all_btn.clicked.connect(self._clear_all_pairs)
        pair_btn_row.addWidget(clear_all_btn)
        markers_layout.addLayout(pair_btn_row)

        auto_row = QHBoxLayout()
        self.auto_pair = QCheckBox("Auto-pair consecutive clicks")
        self.auto_pair.setChecked(True)
        self.auto_pair.toggled.connect(self._set_auto_pair)
        auto_row.addWidget(self.auto_pair)
        markers_layout.addLayout(auto_row)

        self.pair_table = QTableWidget(0, 5)
        self.pair_table.setHorizontalHeaderLabels(["Pair ID", "A marker", "B marker", "Status", "Label"])
        self.pair_table.itemSelectionChanged.connect(self._pair_table_selected)
        markers_layout.addWidget(self.pair_table)
        markers_group.setLayout(markers_layout)
        layout.addWidget(markers_group)

        # Compute
        compute_group = QGroupBox("Compute")
        compute_layout = QVBoxLayout()
        c_row = QHBoxLayout()
        run_sel = QPushButton("Compute selected pair")
        run_sel.clicked.connect(self._compute_selected)
        c_row.addWidget(run_sel)
        run_all = QPushButton("Compute all ready pairs")
        run_all.clicked.connect(self._compute_all)
        c_row.addWidget(run_all)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._cancel_compute)
        c_row.addWidget(cancel_btn)
        compute_layout.addLayout(c_row)
        q_row = QHBoxLayout()
        q_row.addWidget(QLabel("Quality"))
        self.quality = QComboBox()
        self.quality.addItems(["Fast", "Balanced", "Thorough"])
        self.quality.setCurrentText("Balanced")
        q_row.addWidget(self.quality)
        compute_layout.addLayout(q_row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        compute_layout.addWidget(self.progress)
        self.compute_msg = QLabel("Idle.")
        compute_layout.addWidget(self.compute_msg)
        compute_group.setLayout(compute_layout)
        layout.addWidget(compute_group)

        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        self.results_label = QLabel("No computed result selected.")
        results_layout.addWidget(self.results_label)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Build
        build_group = QGroupBox("Build")
        build_layout = QVBoxLayout()
        b_row = QHBoxLayout()
        build_sel = QPushButton("Build selected pair")
        build_sel.clicked.connect(self._build_selected)
        b_row.addWidget(build_sel)
        build_all = QPushButton("Build all computed pairs")
        build_all.clicked.connect(self._build_all)
        b_row.addWidget(build_all)
        build_layout.addLayout(b_row)

        self.identity_best = QRadioButton("Use best-scoring class")
        self.identity_best.setChecked(True)
        self.identity_polyat = QRadioButton("Default poly-AT")
        self.identity_manual = QRadioButton("Manual override")
        build_layout.addWidget(self.identity_best)
        build_layout.addWidget(self.identity_polyat)
        build_layout.addWidget(self.identity_manual)
        ov_row = QHBoxLayout()
        ov_row.addWidget(QLabel("Side A"))
        self.side_a_combo = QComboBox()
        self.side_a_combo.addItems(["A", "G"])
        ov_row.addWidget(self.side_a_combo)
        ov_row.addWidget(QLabel("Side B"))
        self.side_b_combo = QComboBox()
        self.side_b_combo.addItems(["C", "T"])
        ov_row.addWidget(self.side_b_combo)
        build_layout.addLayout(ov_row)
        self.wc_checkbox = QCheckBox("Enforce WC")
        self.wc_checkbox.setChecked(True)
        build_layout.addWidget(self.wc_checkbox)
        build_group.setLayout(build_layout)
        layout.addWidget(build_group)

        # Session/export
        session_group = QGroupBox("Session / export")
        session_layout = QHBoxLayout()
        save_btn = QPushButton("Save session")
        save_btn.clicked.connect(self._save_session_json)
        session_layout.addWidget(save_btn)
        load_btn = QPushButton("Load session")
        load_btn.clicked.connect(self._load_session_json)
        session_layout.addWidget(load_btn)
        session_group.setLayout(session_layout)
        layout.addWidget(session_group)

        self.status_label = QLabel("Ready.")
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        self.tool_window.ui_area.setLayout(layout)
        self._set_marker_mode("place")

    def _toggle_threshold_mode(self, checked: bool) -> None:
        self.threshold_edit.setEnabled(not checked)

    def _sync_threshold_from_viewer(self) -> None:
        # Placeholder for real Volume Viewer linkage; safe default for scaffold.
        self.threshold_edit.setText("0.30")
        self.status_label.setText("Threshold synced (placeholder).")

    def _refresh_models(self) -> None:
        self.map_combo.clear()
        self.model_combo.clear()
        self._map_models = []
        self._structure_models = []
        try:
            models = list(self.session.models.list())
        except Exception:
            models = []
        for m in models:
            path = _disk_path_for_model(m)
            cls = getattr(m, "__class__", type(m)).__name__
            label = f"#{getattr(m, 'id_string', '?')} {getattr(m, 'name', cls)}"
            if path:
                label += f" [{Path(path).name}]"
            kind = _model_kind(m)
            if kind == "map":
                self._map_models.append(m)
                self.map_combo.addItem(label)
            elif kind == "structure":
                self._structure_models.append(m)
                self.model_combo.addItem(label)
        self.status_label.setText(
            f"Detected {len(self._map_models)} map(s), {len(self._structure_models)} structure model(s)."
        )

    def _browse_template_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(None, "Select template directory")
        if d:
            self.template_dir.setText(d)

    def _validate_templates(self) -> None:
        root = Path(self.template_dir.text().strip()).expanduser()
        if not root.is_dir():
            self.status_label.setText(f"Invalid template directory: {root}")
            return
        res = validate_template_pack(root)
        msg = (
            f"entries={len(res.entries)}, missing={len(res.missing_files)}, "
            f"thresholds={len(res.suggested_thresholds())}"
        )
        if res.missing_files:
            preview = ", ".join(res.missing_files[:3])
            if len(res.missing_files) > 3:
                preview += ", ..."
            self.status_label.setText(f"{msg} | missing: {preview}")
            self.session.logger.warning(f"[BaseHunter] Template validation: {msg}")
        else:
            self.status_label.setText(f"{msg} | validation passed.")
            self.session.logger.info(f"[BaseHunter] Template validation: {msg}")

    def _set_marker_mode(self, mode: str) -> None:
        self._state.marker_mode = mode
        for key, btn in self.mode_buttons:
            btn.setChecked(key == mode)
        self.status_label.setText(f"Marker mode: {mode}")

    def _set_auto_pair(self, checked: bool) -> None:
        self._state.auto_pair = bool(checked)

    def _new_pair(self) -> None:
        p = self._state.add_pair()
        self._sync_pair_table()
        self._select_pair_in_table(p.pair_id)
        self.status_label.setText(f"Created {p.pair_id}")

    def _clear_selected_pair(self) -> None:
        if self._state.clear_selected():
            self._sync_pair_table()
            self._update_results_label()
            self.status_label.setText("Selected pair cleared.")

    def _clear_all_pairs(self) -> None:
        self._state.clear_all()
        self._sync_pair_table()
        self._update_results_label()
        self.status_label.setText("Cleared all pairs.")

    def _selected_position(self) -> Optional[Marker3D]:
        # Prefer selected atoms if available.
        try:
            from chimerax.atomic import selected_atoms

            atoms = selected_atoms(self.session)
            if atoms is not None and len(atoms) > 0:
                xs = ys = zs = 0.0
                n = 0
                for a in atoms:
                    pos = getattr(a, "scene_coord", None) or getattr(a, "coord", None)
                    if pos is None:
                        continue
                    xs += float(pos[0])
                    ys += float(pos[1])
                    zs += float(pos[2])
                    n += 1
                if n > 0:
                    return Marker3D(xs / n, ys / n, zs / n)
        except Exception:
            pass
        return None

    def _add_marker_from_selection(self) -> None:
        marker = self._selected_position()
        if marker is None:
            self.status_label.setText("No selected atoms found; select atoms and retry.")
            return
        p = self._state.add_marker(marker)
        if p is None:
            self.status_label.setText("Could not add marker.")
            return
        self._sync_pair_table()
        self._select_pair_in_table(p.pair_id)
        side = "A" if p.marker_b is None else "B"
        self.status_label.setText(f"Added marker {side} to {p.pair_id}.")

    def _pair_table_selected(self) -> None:
        row = self.pair_table.currentRow()
        if row < 0 or row >= len(self._state.pairs):
            return
        pair_id = self.pair_table.item(row, 0).text()
        self._state.set_selected(pair_id)
        self._update_results_label()

    def _sync_pair_table(self) -> None:
        self.pair_table.setRowCount(len(self._state.pairs))
        for i, p in enumerate(self._state.pairs):
            self.pair_table.setItem(i, 0, QTableWidgetItem(p.pair_id))
            self.pair_table.setItem(i, 1, QTableWidgetItem("set" if p.marker_a else "missing"))
            self.pair_table.setItem(i, 2, QTableWidgetItem("set" if p.marker_b else "missing"))
            self.pair_table.setItem(i, 3, QTableWidgetItem(p.status))
            self.pair_table.setItem(i, 4, QTableWidgetItem(p.label or ""))
        self._update_results_label()

    def _select_pair_in_table(self, pair_id: str) -> None:
        for row in range(self.pair_table.rowCount()):
            item = self.pair_table.item(row, 0)
            if item and item.text() == pair_id:
                self.pair_table.selectRow(row)
                break

    @staticmethod
    def _mock_result(pair_id: str) -> PairResult:
        digest = hashlib.sha1(pair_id.encode("utf-8")).digest()
        vals = [x / 255.0 for x in digest[:8]]
        pA_pur = 0.55 + 0.4 * vals[0]
        pB_pyr = 0.55 + 0.4 * vals[1]
        pA_pyr = 1.0 - pA_pur
        pB_pur = 1.0 - pB_pyr
        joint = min(0.99, 0.5 * (pA_pur + pB_pyr))
        plan = 0.1 + 0.4 * vals[2]
        clash = 0.05 + 0.4 * vals[3]
        conf = min(0.99, 0.5 * (abs(pA_pur - pA_pyr) + abs(pB_pyr - pB_pur)))
        call = "A-T" if joint >= 0.75 else "G-C"
        return PairResult(
            pA_purine=float(pA_pur),
            pA_pyrimidine=float(pA_pyr),
            pB_purine=float(pB_pur),
            pB_pyrimidine=float(pB_pyr),
            joint_wc=float(joint),
            planarity_rms=float(plan),
            clash_metric=float(clash),
            confidence=float(conf),
            call=call,
        )

    def _compute_selected(self) -> None:
        p = self._state.get_selected()
        if p is None:
            self.compute_msg.setText("No pair selected.")
            return
        if p.marker_a is None or p.marker_b is None:
            self.compute_msg.setText(f"{p.pair_id} is not ready (needs two markers).")
            return
        self.progress.setValue(20)
        self.compute_msg.setText(f"Computing {p.pair_id} ...")
        res = self._mock_result(p.pair_id)
        self._state.mark_computed(p.pair_id, res)
        self.progress.setValue(100)
        self.compute_msg.setText(f"Computed {p.pair_id}.")
        self._sync_pair_table()
        self._update_results_label()

    def _compute_all(self) -> None:
        ready = [p for p in self._state.pairs if p.marker_a and p.marker_b]
        if not ready:
            self.compute_msg.setText("No ready pairs to compute.")
            return
        total = len(ready)
        for i, p in enumerate(ready, start=1):
            self.compute_msg.setText(f"Computing {p.pair_id} ({i}/{total}) ...")
            self.progress.setValue(int((i - 1) * 100 / max(total, 1)))
            self._state.mark_computed(p.pair_id, self._mock_result(p.pair_id))
        self.progress.setValue(100)
        self.compute_msg.setText(f"Computed {total} pair(s).")
        self._sync_pair_table()

    def _cancel_compute(self) -> None:
        self.compute_msg.setText("No background worker in phase-2 mock; cancel noop.")

    def _update_results_label(self) -> None:
        p = self._state.get_selected()
        if p is None or p.result is None:
            self.results_label.setText("No computed result selected.")
            return
        r = p.result
        self.results_label.setText(
            (
                f"{p.pair_id}: A P(pur)={r.pA_purine:.2f}, P(pyr)={r.pA_pyrimidine:.2f} | "
                f"B P(pur)={r.pB_purine:.2f}, P(pyr)={r.pB_pyrimidine:.2f}\n"
                f"Joint WC={r.joint_wc:.2f}, Planarity={r.planarity_rms:.2f} A, "
                f"Clash={r.clash_metric:.2f}, Call={r.call}, Confidence={r.confidence:.2f}"
            )
        )

    def _build_assignment_for_pair(self, p) -> str:
        if self.identity_polyat.isChecked():
            return "A-T"
        if self.identity_manual.isChecked():
            a = self.side_a_combo.currentText().strip()
            b = self.side_b_combo.currentText().strip()
            if self.wc_checkbox.isChecked():
                ok = (a, b) in {("A", "T"), ("G", "C")}
                if not ok:
                    self.status_label.setText("Build blocked: manual override violates WC mode.")
                    return ""
            return f"{a}-{b}"
        if p.result is not None:
            return p.result.call
        return "A-T"

    def _build_selected(self) -> None:
        p = self._state.get_selected()
        if p is None:
            self.status_label.setText("No pair selected for build.")
            return
        assign = self._build_assignment_for_pair(p)
        if not assign:
            return
        self.session.logger.info(f"[BaseHunter] Build (mock) {p.pair_id}: {assign}")
        self.status_label.setText(f"Build (mock) complete for {p.pair_id}: {assign}")

    def _build_all(self) -> None:
        computed = [p for p in self._state.pairs if p.result is not None]
        if not computed:
            self.status_label.setText("No computed pairs available for build.")
            return
        n_ok = 0
        for p in computed:
            assign = self._build_assignment_for_pair(p)
            if not assign:
                continue
            n_ok += 1
            self.session.logger.info(f"[BaseHunter] Build (mock) {p.pair_id}: {assign}")
        self.status_label.setText(f"Build (mock) complete for {n_ok} pair(s).")

    def _save_session_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self.tool_window.ui_area,
            "Save BaseHunter session",
            str(Path.home() / "basehunter_session.json"),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        payload = self._state.to_dict()
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.status_label.setText(f"Session saved: {path}")

    def _load_session_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.tool_window.ui_area,
            "Load BaseHunter session",
            str(Path.home()),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self._state = PairState.from_dict(payload)
        self.auto_pair.setChecked(self._state.auto_pair)
        self._set_marker_mode(self._state.marker_mode)
        self._sync_pair_table()
        if self._state.selected_pair_id:
            self._select_pair_in_table(self._state.selected_pair_id)
        self.status_label.setText(f"Session loaded: {path}")

