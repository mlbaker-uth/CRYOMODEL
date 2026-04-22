"""ChimeraX BaseHunter Interactive tool (Phase 3a: wired heuristic backend)."""
from __future__ import annotations

import copy
import json
import math
import re
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from Qt.QtCore import QTimer
from Qt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
)


@dataclass
class Pair:
    pair_id: str
    marker_a: Optional[List[float]] = None
    marker_b: Optional[List[float]] = None
    status: str = "new"
    result: Optional[Dict[str, Any]] = None
    marker_atom_a: Optional[object] = None
    marker_atom_b: Optional[object] = None
    # 180° rotation about the A–B marker axis through the midpoint (swap backbone groove side).
    build_flip: bool = False


@dataclass(frozen=True)
class TemplateAtom:
    name: str
    element: str
    coord: np.ndarray


# When False, the build stops after stage 3 (per-chain ``fitmap`` on one dimer model). Set True to restore joint / extra polish.
_BASEHUNTER_RUN_STAGES_AFTER_3 = False
# When True (default), stage 3 runs per-chain ``fitmap`` on the dimer model (map-driven refinement). Set False to keep stage-2 placement only.
_BASEHUNTER_RUN_STAGE3_FITMAP = True


def _disk_path_for_model(m) -> Optional[str]:
    def _check(v) -> Optional[str]:
        if not v:
            return None
        p = Path(str(v))
        return str(p.resolve()) if p.is_file() else None

    for owner in (getattr(m, "opened_data", None), getattr(m, "data", None), m):
        if owner is None:
            continue
        for attr in ("path", "filename", "file_name"):
            got = _check(getattr(owner, attr, None))
            if got:
                return got
    return None


def _kind_for_model(m) -> str:
    cls = getattr(m, "__class__", type(m)).__name__
    if "Volume" in cls:
        return "map"
    if "Structure" in cls:
        return "structure"
    return "other"


def _parse_templates_txt(path: Path) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    line_re = re.compile(r"^\s*([A-Za-z0-9._-]+\.(?:mrc|map|ccp4|pdb|cif))\s*:\s*(.+?)\s*$")
    thr_re = re.compile(r"(?:~|approx(?:\.|imately)?\s*)([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        m = line_re.match(s)
        if not m:
            continue
        desc = m.group(2).strip()
        thr = None
        tm = thr_re.search(desc)
        if tm:
            try:
                thr = float(tm.group(1))
            except ValueError:
                thr = None
        out.append({"filename": m.group(1).strip(), "threshold": thr, "description": desc})
    return out


class BaseHunterInteractiveTool(ToolInstance):
    """Interactive BaseHunter panel with a first wired compute/build backend."""

    SESSION_ENDURING = True

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        self.tool_window = MainToolWindow(self)
        self.tool_window.fill_context_menu = True
        self._pairs: List[Pair] = []
        self._counter = 0
        self._selected_pair_id: Optional[str] = None
        self._map_models: List[object] = []
        self._structure_models: List[object] = []
        self._marker_model = None
        self._template_cache: Dict[str, List[np.ndarray]] = {}
        self._base_template_cache: Dict[str, List[TemplateAtom]] = {}
        # Nucleotide build cache: key (letter, resolved template path) so C vs T never share coordinates.
        self._nucleotide_build_cache: Dict[Tuple[str, str], List[TemplateAtom]] = {}
        self._template_bp_dimer_cache: Optional[Tuple[List[TemplateAtom], List[TemplateAtom]]] = None
        self._template_bp_dimer_cache_path: Optional[Path] = None
        self._scene_coord_note_logged = False
        self._in_build = False
        self._build_ui()
        self._refresh_models()
        self._model_refresh_timer = QTimer(self.tool_window.ui_area)
        self._model_refresh_timer.setInterval(1500)
        self._model_refresh_timer.timeout.connect(self._refresh_models)
        self._model_refresh_timer.start()
        self.tool_window.manage(None)

    def _build_ui(self):
        layout = QVBoxLayout()
        tabs = QTabWidget()
        basic_page = QGroupBox()
        basic_layout = QVBoxLayout()
        advanced_page = QGroupBox()
        advanced_layout = QVBoxLayout()

        data_group = QGroupBox("Data")
        data_layout = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel("Map"))
        self.map_combo = QComboBox()
        row.addWidget(self.map_combo)
        data_layout.addLayout(row)
        thr_row = QHBoxLayout()
        self.inherit_threshold = QCheckBox("Use Volume Viewer threshold")
        self.inherit_threshold.setToolTip(
            "When on, compute/build use the isosurface level shown for the selected map in Volume Viewer "
            "(not the number in Value). Sync copies that level into Value for editing when this is off."
        )
        self.inherit_threshold.setChecked(True)
        self.threshold_edit = QLineEdit("0.30")
        self.threshold_edit.setEnabled(False)
        self.inherit_threshold.toggled.connect(self._on_inherit_threshold_toggled)
        thr_row.addWidget(self.inherit_threshold)
        thr_row.addWidget(QLabel("Value"))
        thr_row.addWidget(self.threshold_edit)
        sync_thr = QPushButton("Sync")
        sync_thr.setToolTip(
            "Copy the current isosurface / image threshold from the selected map into Value (manual mode)."
        )
        sync_thr.clicked.connect(self._sync_threshold_from_volume)
        thr_row.addWidget(sync_thr)
        data_layout.addLayout(thr_row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Working model"))
        self.model_combo = QComboBox()
        row2.addWidget(self.model_combo)
        data_layout.addLayout(row2)
        data_group.setLayout(data_layout)

        tmpl_group = QGroupBox("Templates")
        tmpl_layout = QVBoxLayout()

        path_row = QHBoxLayout()
        self.template_dir = QLineEdit("/Users/mbaker-local/Downloads/CRYOMODEL_LOCAL/NEW-DNA-TEMPLATES")
        path_row.addWidget(self.template_dir)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_templates)
        path_row.addWidget(browse)
        tmpl_layout.addLayout(path_row)
        vbtn = QPushButton("Validate templates")
        vbtn.clicked.connect(self._validate_templates)
        tmpl_layout.addWidget(vbtn)
        tmpl_group.setLayout(tmpl_layout)
        advanced_layout.addWidget(tmpl_group)

        marker_group = QGroupBox("Pair markers")
        marker_layout = QVBoxLayout()
        b_row = QHBoxLayout()
        new_pair = QPushButton("New pair")
        new_pair.clicked.connect(self._new_pair)
        b_row.addWidget(new_pair)
        add_marker = QPushButton("Add marker from selection")
        add_marker.clicked.connect(self._add_marker_from_selection)
        b_row.addWidget(add_marker)
        clear_sel = QPushButton("Clear selected")
        clear_sel.clicked.connect(self._clear_selected)
        b_row.addWidget(clear_sel)
        swap_chain = QPushButton("Swap chains")
        swap_chain.setToolTip(
            "Exchange Marker A ↔ Marker B for the selected pair only (coordinates, marker atoms, and phase-1 scores) while keeping the "
            "same WC call (A–T vs G–C). Use when quad strand labels disagree with how you want chain A/B colored."
        )
        swap_chain.clicked.connect(self._swap_selected_chains)
        b_row.addWidget(swap_chain)
        swap_assign = QPushButton("Swap assignment")
        swap_assign.setToolTip(
            "Swap R↔Y assignment for the selected pair only (does not move markers or change chain A/B). "
            "Use this when the selected BP has the wrong purine/pyrimidine assignment."
        )
        swap_assign.clicked.connect(self._swap_selected_assignment)
        b_row.addWidget(swap_assign)
        marker_layout.addLayout(b_row)
        self.pair_table = QTableWidget(0, 6)
        self.pair_table.setHorizontalHeaderLabels(
            ["Pair ID", "Marker A", "Marker B", "Assign", "Status", "Refine"]
        )
        self.pair_table.itemSelectionChanged.connect(self._table_selected)
        marker_layout.addWidget(self.pair_table)
        marker_group.setLayout(marker_layout)
        basic_layout.addWidget(data_group)
        basic_layout.addWidget(marker_group)

        compute_group = QGroupBox("Compute / build / refine")
        compute_layout = QVBoxLayout()
        c_row = QHBoxLayout()
        run_sel = QPushButton("Compute selected")
        run_sel.clicked.connect(self._compute_selected)
        c_row.addWidget(run_sel)
        run_all = QPushButton("Compute all")
        run_all.clicked.connect(self._compute_all)
        c_row.addWidget(run_all)
        compute_layout.addLayout(c_row)
        build_row = QHBoxLayout()
        b1 = QPushButton("Build selected")
        b1.clicked.connect(self._build_selected)
        build_row.addWidget(b1)
        b2 = QPushButton("Build all")
        b2.clicked.connect(self._build_all)
        build_row.addWidget(b2)
        compute_layout.addLayout(build_row)
        r_row = QHBoxLayout()
        ref_sel = QPushButton("Refine assignments (selected)")
        ref_sel.setToolTip(
            "Second-round NSP-only assignment refinement on the selected pair. "
            "Geometry/chain IDs stay fixed; only purine/pyrimidine identity is re-evaluated."
        )
        ref_sel.clicked.connect(self._refine_assignments_selected)
        r_row.addWidget(ref_sel)
        ref_all = QPushButton("Refine assignments (all)")
        ref_all.setToolTip(
            "Second-round NSP-only assignment refinement for all computed pairs. "
            "Flags accepted/suspect/ambiguous from hypothesis score deltas."
        )
        ref_all.clicked.connect(self._refine_assignments_all)
        r_row.addWidget(ref_all)
        compute_layout.addLayout(r_row)
        q_row = QHBoxLayout()
        q_row.addWidget(QLabel("Quality"))
        self.quality = QComboBox()
        self.quality.addItems(["Fast"])
        self.quality.setToolTip(
            "Fast: phase-1 marker template scores only. "
            "Balanced / Thorough quality presets are hidden for now; post-build refine remains available via the Refine buttons."
        )
        q_row.addWidget(self.quality)
        compute_layout.addLayout(q_row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        compute_layout.addWidget(self.progress)
        self.compute_msg = QLabel("Idle.")
        compute_layout.addWidget(self.compute_msg)
        compute_group.setLayout(compute_layout)
        basic_layout.addWidget(compute_group)

        results_group = QGroupBox("Results (all pairs)")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Compute pairs to see per-marker scores and WC call here.")
        self.results_text.setMinimumHeight(140)
        self.results_text.setMaximumHeight(260)
        results_layout.addWidget(self.results_text)
        export_row = QHBoxLayout()
        export_calc = QPushButton("Export calculations to JSON…")
        export_calc.setToolTip(
            "Write threshold, map/model picks, template directory, and per-pair scores (phase-1, refine, tiers) "
            "to a JSON file for debugging or records."
        )
        export_calc.clicked.connect(self._export_calculations_json)
        export_row.addWidget(export_calc)
        export_row.addStretch(1)
        results_layout.addLayout(export_row)
        results_group.setLayout(results_layout)
        basic_layout.addWidget(results_group)

        ses_group = QGroupBox("Session / export")
        ses_row = QHBoxLayout()
        save = QPushButton("Save session")
        save.clicked.connect(self._save_session)
        ses_row.addWidget(save)
        load = QPushButton("Load session")
        load.clicked.connect(self._load_session)
        ses_row.addWidget(load)
        ses_group.setLayout(ses_row)
        advanced_layout.addWidget(ses_group)

        self.status = QLabel("Ready.")
        layout.addWidget(self.status)
        basic_page.setLayout(basic_layout)
        advanced_page.setLayout(advanced_layout)
        tabs.addTab(basic_page, "Main")
        tabs.addTab(advanced_page, "Advanced")
        layout.addWidget(tabs)
        layout.addStretch(1)
        self.tool_window.ui_area.setLayout(layout)

    def _refresh_models(self, prefer_structure=None, force: bool = False):
        if self._in_build and not force:
            return
        prev_map_i = self.map_combo.currentIndex()
        prev_model_i = self.model_combo.currentIndex()
        prev_map = self._map_models[prev_map_i] if 0 <= prev_map_i < len(self._map_models) else None
        prev_model = self._structure_models[prev_model_i] if 0 <= prev_model_i < len(self._structure_models) else None
        self.map_combo.clear()
        self.model_combo.clear()
        self._map_models = []
        self._structure_models = []
        try:
            models = list(self.session.models.list())
        except Exception:
            models = []
        for m in models:
            cls = getattr(m, "__class__", type(m)).__name__
            label = f"#{getattr(m, 'id_string', '?')} {getattr(m, 'name', cls)}"
            p = _disk_path_for_model(m)
            if p:
                label += f" [{Path(p).name}]"
            k = _kind_for_model(m)
            if k == "map":
                self._map_models.append(m)
                self.map_combo.addItem(label)
            elif k == "structure":
                self._structure_models.append(m)
                self.model_combo.addItem(label)
        if self._map_models:
            tgt_map = prev_map if prev_map in self._map_models else self._map_models[0]
            self.map_combo.setCurrentIndex(self._map_models.index(tgt_map))
        if self._structure_models:
            if prefer_structure in self._structure_models:
                tgt_model = prefer_structure
            elif prev_model in self._structure_models:
                tgt_model = prev_model
            else:
                tgt_model = self._structure_models[-1]
            self.model_combo.setCurrentIndex(self._structure_models.index(tgt_model))
        self.status.setText(f"Detected {len(self._map_models)} map(s), {len(self._structure_models)} model(s).")

    def _browse_templates(self):
        d = QFileDialog.getExistingDirectory(None, "Select template directory")
        if d:
            self.template_dir.setText(d)
            self._template_cache = {}
            self._class_atom_templates_lists_cache = None

    def _validate_templates(self):
        root = Path(self.template_dir.text().strip()).expanduser()
        txt = root / "templates.txt"
        if not root.is_dir():
            self.status.setText(f"Invalid template directory: {root}")
            return
        if not txt.is_file():
            self.status.setText("templates.txt not found; validation skipped.")
            return
        entries = _parse_templates_txt(txt)
        missing: List[str] = []
        n_thr = 0
        for e in entries:
            fp = root / str(e["filename"])
            if not fp.exists():
                missing.append(str(e["filename"]))
            if e.get("threshold") is not None:
                n_thr += 1
        msg = f"entries={len(entries)}, missing={len(missing)}, thresholds={n_thr}"
        if missing:
            preview = ", ".join(missing[:3]) + (", ..." if len(missing) > 3 else "")
            self.status.setText(f"{msg} | missing: {preview}")
            self.session.logger.warning(f"[BaseHunter] {msg}")
        else:
            self.status.setText(f"{msg} | validation passed.")
            self.session.logger.info(f"[BaseHunter] {msg}")

    def _new_pair(self):
        self._counter += 1
        pair = Pair(pair_id=f"BP_{self._counter:03d}")
        self._pairs.append(pair)
        self._selected_pair_id = pair.pair_id
        self._sync_table()
        self.status.setText(f"Created {pair.pair_id}")

    def _selected_atom_centroid(self) -> Optional[List[float]]:
        try:
            from chimerax.atomic import selected_atoms

            atoms = selected_atoms(self.session)
            if atoms is None or len(atoms) == 0:
                return None
            sx = sy = sz = 0.0
            n = 0
            for a in atoms:
                # Prefer scene coordinates so markers match the active volume in the same frame as fitmap/superposition.
                pos = getattr(a, "scene_coord", None) or getattr(a, "coord", None)
                if pos is None:
                    continue
                sx += float(pos[0])
                sy += float(pos[1])
                sz += float(pos[2])
                n += 1
            if n == 0:
                return None
            return [sx / n, sy / n, sz / n]
        except Exception:
            return None

    def _find_pair(self, pair_id: Optional[str]) -> Optional[Pair]:
        if not pair_id:
            return None
        for p in self._pairs:
            if p.pair_id == pair_id:
                return p
        return None

    def _add_marker_from_selection(self):
        xyz = self._selected_atom_centroid()
        if xyz is None:
            self.status.setText("No selected atoms found; select atoms and retry.")
            return
        p = self._find_pair(self._selected_pair_id)
        if p is None:
            self._new_pair()
            p = self._find_pair(self._selected_pair_id)
        if p is None:
            self.status.setText("Could not create/select pair.")
            return
        if p.marker_a is None:
            p.marker_a = xyz
            p.marker_atom_a = self._upsert_marker_atom(p, side="A", xyz=xyz)
            side = "A"
        elif p.marker_b is None:
            p.marker_b = xyz
            p.marker_atom_b = self._upsert_marker_atom(p, side="B", xyz=xyz)
            side = "B"
        else:
            self._new_pair()
            p = self._find_pair(self._selected_pair_id)
            if p is None:
                self.status.setText("Could not allocate new pair.")
                return
            p.marker_a = xyz
            p.marker_atom_a = self._upsert_marker_atom(p, side="A", xyz=xyz)
            side = "A"
        p.status = "ready" if p.marker_a is not None and p.marker_b is not None else "new"
        p.result = None
        self._sync_table()
        self.status.setText(f"Added marker {side} to {p.pair_id}.")

    def _ensure_marker_model(self):
        if self._marker_model is not None:
            return self._marker_model
        from chimerax.atomic import AtomicStructure
        from chimerax.core.commands import run

        self._marker_model = AtomicStructure(self.session, name="BaseHunter Markers")
        self.session.models.add([self._marker_model])
        try:
            run(self.session, f"style {self._marker_model.id_string} sphere")
            run(self.session, f"size {self._marker_model.id_string} atomRadius 1.1")
        except Exception:
            pass
        return self._marker_model

    def _upsert_marker_atom(self, pair: Pair, side: str, xyz: List[float]):
        from chimerax.atomic import Element

        model = self._ensure_marker_model()
        atom_attr = "marker_atom_a" if side == "A" else "marker_atom_b"
        atom = getattr(pair, atom_attr)
        if atom is None:
            chain_id = "A" if side == "A" else "B"
            try:
                seq = int(pair.pair_id.split("_")[-1])
            except Exception:
                seq = len(self._pairs) + 1
            residue = model.new_residue("MRK", chain_id, seq)
            atom = model.new_atom(f"M{side}", Element.get_element("C"))
            residue.add_atom(atom)
            atom.radius = 1.1
            atom.color = self._color_from_class_conf("unknown", 0.0)
            setattr(pair, atom_attr, atom)
        atom.coord = xyz
        return atom

    @staticmethod
    def _marker_atom_label(atom) -> str:
        """Compact ChimeraX-style id for table cells and logs (e.g. ``#7/A:3@MA``)."""
        if atom is None:
            return "—"
        try:
            m = atom.structure
            mid = getattr(m, "id_string", "?")
            r = atom.residue
            cid = str(getattr(r, "chain_id", "?")).strip()
            num = getattr(r, "number", "?")
            nm = str(getattr(atom, "name", "M")).strip()
            return f"#{mid}/{cid}:{num}@{nm}"
        except Exception:
            return str(getattr(atom, "name", "?"))

    @staticmethod
    def _swap_marker_axis_scores_in_result(r: Dict[str, float], *, freeze_call: Optional[str] = None) -> None:
        """Exchange ``pA_*`` / ``pB_*`` entries so scores track marker coordinates after A↔B swap."""
        pairs = (("pA_purine", "pB_purine"), ("pA_pyrimidine", "pB_pyrimidine"))
        for ka, kb in pairs:
            if ka in r and kb in r:
                va, vb = float(r[ka]), float(r[kb])
                r[ka], r[kb] = vb, va
        for ka, kb in (("pA_purine_phase1", "pB_purine_phase1"), ("pA_pyrimidine_phase1", "pB_pyrimidine_phase1")):
            if ka in r and kb in r:
                va, vb = float(r[ka]), float(r[kb])
                r[ka], r[kb] = vb, va
        pA = float(r.get("pA_purine", 0.5))
        pA_pyr = float(r.get("pA_pyrimidine", 0.5))
        pB_pur = float(r.get("pB_purine", 0.5))
        pB_pyr = float(r.get("pB_pyrimidine", 0.5))
        clash = float(r.get("clash_metric", 0.0))
        r["joint_wc"] = float(max(0.0, min(0.99, 0.5 * (pA + pB_pyr) - 0.20 * clash)))
        r["confidence"] = float(
            max(0.0, min(0.99, 0.5 * (abs(pA - pA_pyr) + abs(pB_pur - pB_pyr))))
        )
        if freeze_call is not None:
            r["call"] = str(freeze_call)
        else:
            r["call"] = "A-T" if (pA + pB_pyr) >= (pA_pyr + pB_pur) else "G-C"

    @staticmethod
    def _invert_marker_class_scores_in_result(r: Dict[str, float]) -> None:
        """Invert per-marker purine/pyrimidine identity (R↔Y) while keeping marker A/B fixed."""
        for ka, kb in (("pA_purine", "pA_pyrimidine"), ("pB_purine", "pB_pyrimidine")):
            if ka in r and kb in r:
                va, vb = float(r[ka]), float(r[kb])
                r[ka], r[kb] = vb, va
        for ka, kb in (("pA_purine_phase1", "pA_pyrimidine_phase1"), ("pB_purine_phase1", "pB_pyrimidine_phase1")):
            if ka in r and kb in r:
                va, vb = float(r[ka]), float(r[kb])
                r[ka], r[kb] = vb, va
        pA = float(r.get("pA_purine", 0.5))
        pA_pyr = float(r.get("pA_pyrimidine", 0.5))
        pB_pur = float(r.get("pB_purine", 0.5))
        pB_pyr = float(r.get("pB_pyrimidine", 0.5))
        clash = float(r.get("clash_metric", 0.0))
        r["joint_wc"] = float(max(0.0, min(0.99, 0.5 * (pA + pB_pyr) - 0.20 * clash)))
        r["confidence"] = float(
            max(0.0, min(0.99, 0.5 * (abs(pA - pA_pyr) + abs(pB_pur - pB_pyr))))
        )
        r["call"] = "A-T" if (pA + pB_pyr) >= (pA_pyr + pB_pur) else "G-C"

    @staticmethod
    def _color_from_class_conf(cls_name: str, conf: float):
        """Return RGBA marker color for class with confidence-modulated intensity."""
        conf = max(0.0, min(1.0, float(conf)))
        # Base colors requested by user:
        # purine -> violet, pyrimidine -> teal.
        if cls_name == "purine":
            base = (148, 0, 211)
        elif cls_name == "pyrimidine":
            base = (0, 128, 128)
        else:
            base = (140, 140, 140)
        # Keep hue even at low confidence: interpolate from a dim version of class color.
        min_intensity = 0.35
        mix = min_intensity + (1.0 - min_intensity) * conf
        r = int(mix * base[0])
        g = int(mix * base[1])
        b = int(mix * base[2])
        return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)), 255)

    @staticmethod
    def _complement_class(cls_name: str) -> str:
        return "pyrimidine" if cls_name == "purine" else "purine"

    def _display_decision(self, p: Pair, result_override: Optional[Dict[str, float]] = None) -> Optional[Dict[str, object]]:
        """Resolve per-side class/probability for display and coloring with WC-aware tie handling."""
        r = result_override if result_override is not None else (p.result if p is not None else None)
        if r is None:
            return None
        pA_pur = float(r.get("pA_purine", 0.5))
        pA_pyr = float(r.get("pA_pyrimidine", 0.5))
        pB_pur = float(r.get("pB_purine", 0.5))
        pB_pyr = float(r.get("pB_pyrimidine", 0.5))

        cls_a = "purine" if pA_pur >= pA_pyr else "pyrimidine"
        cls_b = "purine" if pB_pur >= pB_pyr else "pyrimidine"
        conf_a = abs(pA_pur - pA_pyr)
        conf_b = abs(pB_pur - pB_pyr)

        # Strong evidence gate (~50% larger than alternative) for "dominant" calls.
        ratio_a = max(pA_pur, pA_pyr) / max(1e-6, min(pA_pur, pA_pyr))
        ratio_b = max(pB_pur, pB_pyr) / max(1e-6, min(pB_pur, pB_pyr))
        strong_a = ratio_a >= 1.5
        strong_b = ratio_b >= 1.5

        # If one side is strongly classified and the other is ambiguous, optionally enforce WC-consistent
        # display on the ambiguous side — only when that side is still ~50/50 (ratio < ~1.12). Otherwise
        # the weak side keeps its own marginal (avoids forcing pyr on B when B's softmax already leans pur
        # due to geometry, a case that produced wrong builds when combined with biased phase-1 scores).
        margin_ambig = 1.12
        if strong_a and not strong_b and ratio_b < margin_ambig:
            cls_b = self._complement_class(cls_a)
        elif strong_b and not strong_a and ratio_a < margin_ambig:
            cls_a = self._complement_class(cls_b)
        elif strong_a and strong_b and cls_a == cls_b:
            # Conflict: keep stronger side, flip weaker side for WC display consistency.
            if conf_a >= conf_b:
                cls_b = self._complement_class(cls_a)
            else:
                cls_a = self._complement_class(cls_b)
        elif cls_a == cls_b:
            # e.g. both marginals same class and WC complement was skipped (weak side not flat enough).
            if conf_a >= conf_b:
                cls_b = self._complement_class(cls_a)
            else:
                cls_a = self._complement_class(cls_b)

        score_a = pA_pur if cls_a == "purine" else pA_pyr
        score_b = pB_pur if cls_b == "purine" else pB_pyr
        return {
            "class_a": cls_a,
            "class_b": cls_b,
            "score_a": float(score_a),
            "score_b": float(score_b),
            "conf_a": float(conf_a),
            "conf_b": float(conf_b),
            "pA_pur": pA_pur,
            "pA_pyr": pA_pyr,
            "pB_pur": pB_pur,
            "pB_pyr": pB_pyr,
        }

    def _raw_density_classes(self, p: Pair) -> Optional[Tuple[str, str]]:
        """Per-marker purine vs pyrimidine from softmax scores only (no WC display smoothing)."""
        if p.result is None:
            return None
        r = p.result
        pA_pur = float(r.get("pA_purine", 0.5))
        pB_pur = float(r.get("pB_purine", 0.5))
        ca = "purine" if pA_pur >= 0.5 else "pyrimidine"
        cb = "purine" if pB_pur >= 0.5 else "pyrimidine"
        return ca, cb

    def _apply_marker_colors(self, p: Pair):
        decision = self._display_decision(p)
        if decision is None:
            return
        if p.marker_atom_a is not None:
            p.marker_atom_a.color = self._color_from_class_conf(
                str(decision["class_a"]), float(decision["conf_a"])
            )
        if p.marker_atom_b is not None:
            p.marker_atom_b.color = self._color_from_class_conf(
                str(decision["class_b"]), float(decision["conf_b"])
            )

    def _clear_selected(self):
        p = self._find_pair(self._selected_pair_id)
        if p is None:
            return
        try:
            if p.marker_atom_a is not None:
                p.marker_atom_a.delete()
            if p.marker_atom_b is not None:
                p.marker_atom_b.delete()
        except Exception:
            pass
        self._pairs = [x for x in self._pairs if x.pair_id != p.pair_id]
        self._selected_pair_id = self._pairs[0].pair_id if self._pairs else None
        self._sync_table()
        self.status.setText("Selected pair cleared.")

    def _table_selected(self):
        row = self.pair_table.currentRow()
        if row < 0 or row >= len(self._pairs):
            return
        pid = self.pair_table.item(row, 0).text()
        self._selected_pair_id = pid
        self._update_results()

    def _sync_table(self):
        """Refresh the pair table and restore the row for ``_selected_pair_id`` (Qt clears selection on repopulate)."""
        sel = self._selected_pair_id
        self.pair_table.blockSignals(True)
        if not self._pairs:
            self.pair_table.setRowCount(0)
            self.pair_table.blockSignals(False)
            self._selected_pair_id = None
            self._update_results()
            return
        self.pair_table.setRowCount(len(self._pairs))
        for i, p in enumerate(self._pairs):
            self.pair_table.setItem(i, 0, QTableWidgetItem(p.pair_id))
            self.pair_table.setItem(
                i,
                1,
                QTableWidgetItem(self._marker_atom_label(p.marker_atom_a) if p.marker_a else "missing"),
            )
            self.pair_table.setItem(
                i,
                2,
                QTableWidgetItem(self._marker_atom_label(p.marker_atom_b) if p.marker_b else "missing"),
            )
            assign = "?"
            d = self._display_decision(p)
            if d is not None:
                a_cls = str(d.get("class_a", "")).lower()
                b_cls = str(d.get("class_b", "")).lower()
                a_tag = "R" if a_cls == "purine" else ("Y" if a_cls == "pyrimidine" else "?")
                b_tag = "R" if b_cls == "purine" else ("Y" if b_cls == "pyrimidine" else "?")
                assign = f"A:{a_tag} B:{b_tag}"
            self.pair_table.setItem(i, 3, QTableWidgetItem(assign))
            self.pair_table.setItem(i, 4, QTableWidgetItem(p.status))
            ref_cell = ""
            if p.result and bool(p.result.get("refine_enabled")):
                tier = str(p.result.get("refine_tier", "")).strip() or "—"
                cp2 = float(p.result.get("refine_confidence_phase2", 0.0))
                ref_cell = f"{tier}·{cp2:.2f}"
            self.pair_table.setItem(i, 5, QTableWidgetItem(ref_cell))
        row = 0
        if sel:
            for i, p in enumerate(self._pairs):
                if p.pair_id == sel:
                    row = i
                    break
            else:
                sel = self._pairs[0].pair_id
                row = 0
        else:
            sel = self._pairs[0].pair_id
            row = 0
        self._selected_pair_id = sel
        self.pair_table.selectRow(row)
        self.pair_table.blockSignals(False)
        self._update_results()

    def _current_map_model(self):
        idx = self.map_combo.currentIndex()
        if idx < 0 or idx >= len(self._map_models):
            return None
        return self._map_models[idx]

    def _current_structure_model(self):
        idx = self.model_combo.currentIndex()
        if idx < 0 or idx >= len(self._structure_models):
            return None
        return self._structure_models[idx]

    def _on_inherit_threshold_toggled(self, _state: int) -> None:
        """Manual value when unchecked; live Volume level when checked (see :meth:`_threshold_value`)."""
        self.threshold_edit.setEnabled(not self.inherit_threshold.isChecked())

    def _volume_display_threshold(self, map_model) -> Optional[float]:
        """Isosurface level from the map's Volume Viewer state (ChimeraX :class:`~chimerax.map.Volume`)."""
        if map_model is None:
            return None
        ro = getattr(map_model, "rendering_options", None)
        if ro is not None:
            for attr in ("surface_levels", "contour_levels"):
                sl = getattr(ro, attr, None)
                if sl is not None and len(sl) > 0:
                    try:
                        return float(sl[0])
                    except (TypeError, ValueError, IndexError):
                        pass
            il = getattr(ro, "image_levels", None)
            if il is not None and len(il) > 0:
                try:
                    first = il[0]
                    if isinstance(first, (list, tuple)) and len(first) > 0:
                        return float(first[0])
                except (TypeError, ValueError, IndexError):
                    pass
        for attr in ("surface_levels", "image_levels"):
            sl = getattr(map_model, attr, None)
            if not sl:
                continue
            try:
                if attr == "surface_levels":
                    return float(sl[0])
                first = sl[0]
                if isinstance(first, (list, tuple)) and len(first) > 0:
                    return float(first[0])
            except (TypeError, ValueError, IndexError):
                pass
        surfs = getattr(map_model, "surfaces", None)
        if surfs:
            for s in surfs:
                for attr in ("level", "levels"):
                    lv = getattr(s, attr, None)
                    if lv is None:
                        continue
                    try:
                        if isinstance(lv, (list, tuple)) and len(lv) > 0:
                            return float(lv[0])
                        return float(lv)
                    except (TypeError, ValueError):
                        pass
        return None

    def _sync_threshold_from_volume(self) -> None:
        """Fill the Value field from the selected map's displayed threshold (for manual override / inspection)."""
        vm = self._current_map_model()
        v = self._volume_display_threshold(vm)
        if v is not None:
            self.threshold_edit.setText(f"{v:.6g}".rstrip("0").rstrip("."))
            self.status.setText(f"Threshold value set from map: {v:g}")
        else:
            self.status.setText(
                "Could not read threshold from the selected map (use a Volume model; set level in Volume Viewer)."
            )

    def _threshold_value(self) -> float:
        v, _src = self._threshold_value_with_source()
        return v

    def _threshold_value_with_source(self) -> Tuple[float, str]:
        if self.inherit_threshold.isChecked():
            vm = self._current_map_model()
            vv = self._volume_display_threshold(vm)
            if vv is not None:
                return float(vv), "Volume Viewer"
        try:
            return float(self.threshold_edit.text().strip()), "manual Value"
        except Exception:
            return 0.30, "fallback default"

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    @staticmethod
    def _dist(a: List[float], b: List[float]) -> float:
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        dz = float(a[2]) - float(b[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def _rotation_from_to(src_vec: np.ndarray, dst_vec: np.ndarray) -> np.ndarray:
        """Rodrigues rotation matrix from src_vec to dst_vec."""
        s = BaseHunterInteractiveTool._normalize(src_vec)
        d = BaseHunterInteractiveTool._normalize(dst_vec)
        v = np.cross(s, d)
        c = float(np.dot(s, d))
        if np.linalg.norm(v) < 1e-12:
            if c > 0:
                return np.eye(3, dtype=np.float64)
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(s, axis))) > 0.9:
                axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            return BaseHunterInteractiveTool._rot_matrix(np.cross(s, axis), 180.0)
        vx = np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ],
            dtype=np.float64,
        )
        return np.eye(3, dtype=np.float64) + vx + (vx @ vx) * (1.0 / max(1e-12, 1.0 + c))

    @staticmethod
    def _template_plane_normal(coords: np.ndarray) -> np.ndarray:
        if coords.shape[0] < 3:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        _, _, vh = np.linalg.svd(coords - coords.mean(axis=0, keepdims=True), full_matrices=False)
        return BaseHunterInteractiveTool._normalize(vh[-1])

    def _sample_density(self, map_model, xyz: List[float]) -> float:
        """Sample map density near xyz with multiple API fallbacks.

        ChimeraX volumes expect ``xyz`` in **scene coordinates**, the same space as
        ``atom.scene_coord`` from selections (see ``_selected_atom_centroid``).
        """
        # Preferred: Volume interpolation API.
        try:
            vals = map_model.interpolated_values([xyz])
            if vals is not None and len(vals) > 0:
                return float(vals[0])
        except Exception:
            pass
        # Fallback: GridData.value_at_point (correct trilinear / skewed cell handling).
        try:
            data = getattr(map_model, "data", None)
            if data is not None:
                vfn = getattr(data, "value_at_point", None)
                if callable(vfn):
                    v = vfn(xyz)
                    if v is not None:
                        return float(v)
        except Exception:
            pass
        # Last resort: nearest voxel (array axis order varies; prefer value_at_point above).
        try:
            data = getattr(map_model, "data", None)
            if data is None:
                return 0.0
            full = data.full_matrix()
            ijk = data.xyz_to_ijk(xyz)
            i = int(round(float(ijk[0])))
            j = int(round(float(ijk[1])))
            k = int(round(float(ijk[2])))
            size = getattr(data, "size", None)
            if size is not None and len(size) >= 3:
                nx, ny, nz = int(size[0]), int(size[1]), int(size[2])
                if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                    if full.ndim == 3 and full.shape == (nx, ny, nz):
                        return float(full[i, j, k])
                    if full.ndim == 3 and full.shape == (nz, ny, nx):
                        return float(full[k, j, i])
            if full.ndim == 3 and 0 <= k < full.shape[0] and 0 <= j < full.shape[1] and 0 <= i < full.shape[2]:
                return float(full[k, j, i])
        except Exception:
            pass
        return 0.0

    def _sample_local_density(self, map_model, xyz: List[float], threshold: Optional[float] = None) -> float:
        """Mean of center + 6 axis offsets for a stable local score.

        When ``threshold`` is provided, only above-threshold mass contributes (``max(v-threshold, 0)``).
        """
        offsets = (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        )
        vals: List[float] = []
        for dx, dy, dz in offsets:
            vals.append(self._sample_density(map_model, [xyz[0] + dx, xyz[1] + dy, xyz[2] + dz]))
        if not vals:
            return 0.0
        if threshold is not None:
            vals = [max(float(v) - float(threshold), 0.0) for v in vals]
        return float(sum(vals) / len(vals))

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return (v / n).astype(np.float64)

    @staticmethod
    def _rot_matrix(axis: np.ndarray, degrees: float) -> np.ndarray:
        axis = BaseHunterInteractiveTool._normalize(axis)
        a = math.radians(float(degrees))
        c = math.cos(a)
        s = math.sin(a)
        x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
        return np.array(
            [
                [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _is_sugar_phosphate_atom_name(atom_name: str) -> bool:
        """True for DNA backbone / sugar atoms (primed); base ring atoms like C5 are not sugar."""
        s = atom_name.strip().upper().replace("*", "'")
        if s in {"P", "OP1", "OP2", "OP3", "O1P", "O2P", "PA", "PB", "PG"}:
            return True
        if s.startswith("P") and len(s) <= 2:
            return True
        return s in {
            "O5'",
            "C5'",
            "C4'",
            "O4'",
            "C3'",
            "O3'",
            "C2'",
            "O2'",
            "C1'",
        }

    @staticmethod
    def _dna_three_letter(one: str) -> str:
        return {"A": "DA", "G": "DG", "C": "DC", "T": "DT"}.get(one.upper(), "DA")

    @staticmethod
    def _is_purine_letter(ch: str) -> bool:
        return ch.upper() in {"A", "G"}

    @staticmethod
    def _wc_ordered_pur_pyr(call: str) -> Tuple[str, str]:
        """Return (purine_letter, pyrimidine_letter) for a WC pair string (e.g. A-T, G-C)."""
        parts = [x.strip().upper() for x in call.split("-")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return "A", "T"
        x, y = parts[0][0], parts[1][0]
        if BaseHunterInteractiveTool._is_purine_letter(x):
            return x, y
        if BaseHunterInteractiveTool._is_purine_letter(y):
            return y, x
        return x, y

    def _letters_at_markers(self, p: Pair, call: str) -> Tuple[str, str]:
        """(letter_at_marker_a, letter_at_marker_b). Manual: UI bases. Auto: WC letters from ``call``, mapped using
        **raw** per-marker purine scores when they are WC-consistent; otherwise WC-smoothed display classes."""
        parts = [x.strip().upper() for x in call.split("-")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return "A", "T"
        pur, pyr = self._wc_ordered_pur_pyr(call)
        raw = self._raw_density_classes(p)
        if raw is not None:
            ra, rb = raw
            if ra != rb:
                return (pur, pyr) if ra == "purine" else (pyr, pur)
        d = self._display_decision(p)
        if d is None:
            return pur, pyr
        return (pur, pyr) if str(d["class_a"]) == "purine" else (pyr, pur)

    def _rot_place_by_chemistry(
        self,
        letter: str,
        full: List[TemplateAtom],
        e1: np.ndarray,
        e2w: np.ndarray,
        e3w: np.ndarray,
        is_purine_residue: bool,
    ) -> np.ndarray:
        """Purine vs pyrimidine template gets the matching +e3 / −e3 placement convention."""
        n_t, b_t = self._ring_normal_and_sugar_inplane(letter, full)
        if is_purine_residue:
            return self._rotation_pair_place(n_t, b_t, e1, e2w, e3w, 1.0, 1.0)
        return self._rotation_pair_place(n_t, b_t, e1, e2w, e3w, -1.0, -1.0)

    @staticmethod
    def _norm_atom_key(name: str) -> str:
        s = name.strip().upper().replace("*", "'")
        return {"O1P": "OP1", "O2P": "OP2"}.get(s, s)

    @staticmethod
    def _is_purine_resname(resname: str) -> bool:
        r = resname.strip().upper()
        return r in {"DA", "DG", "RA", "RG"}

    def _template_bp_path(self, root: Path) -> Optional[Path]:
        for fn in ("templateBP.pdb", "templatebp.pdb"):
            p = root / fn
            if p.is_file():
                return p
        return None

    @staticmethod
    def _count_base_heavy_atoms(atoms: List[TemplateAtom]) -> int:
        n = 0
        for a in atoms:
            if (a.element or "C").upper() == "H":
                continue
            if BaseHunterInteractiveTool._is_sugar_phosphate_atom_name(a.name):
                continue
            n += 1
        return n

    def _infer_pyr_pur_sites_from_geometry(
        self, items: List[Tuple[Tuple[str, int], str, List[TemplateAtom]]]
    ) -> Optional[Tuple[List[TemplateAtom], List[TemplateAtom]]]:
        """Pick a WC-like cross-chain pair when resnames do not separate pur vs pyr; pur = more base heavy atoms."""
        if len(items) < 2:
            return None
        target_c1 = 10.55
        target_com = 6.05
        scored: List[Tuple[float, float, List[TemplateAtom], List[TemplateAtom]]] = []
        for i, (_ki, _rni, at_i) in enumerate(items):
            for j, (_kj, _rnj, at_j) in enumerate(items):
                if i >= j or _ki[0] == _kj[0]:
                    continue
                c1i = self._find_c1_prime(at_i)
                c1j = self._find_c1_prime(at_j)
                if c1i is not None and c1j is not None:
                    d = float(np.linalg.norm(c1j - c1i))
                    tgt = target_c1
                    lo, hi = 8.5, 12.5
                else:
                    com_i = self._base_com_from_atoms(at_i)
                    com_j = self._base_com_from_atoms(at_j)
                    d = float(np.linalg.norm(com_j - com_i))
                    tgt = target_com
                    lo, hi = 4.5, 8.5
                if lo < d < hi:
                    score = abs(d - tgt)
                else:
                    score = abs(d - tgt) + 50.0
                scored.append((score, d, at_i, at_j))
        if not scored:
            return None
        scored.sort(key=lambda t: (t[0], t[1]))
        best = scored[0]
        at_i, at_j = best[2], best[3]
        ci = self._count_base_heavy_atoms(at_i)
        cj = self._count_base_heavy_atoms(at_j)
        if ci > cj:
            return at_j, at_i
        if cj > ci:
            return at_i, at_j
        return at_i, at_j

    def _parse_template_bp_dimer_sites(self, path: Path) -> Optional[Tuple[List[TemplateAtom], List[TemplateAtom]]]:
        """Return ``(site_a, site_b)`` two residues from ``templateBP.pdb`` (geometry anchor; chemistry not implied).

        For multi-residue duplex PDBs, **do not** take the first pyrimidine and first purine in sort order:
        that can pick adjacent same-strand residues (e.g. DC A1 + DG A2) or a non-WC cross-chain pair.
        Instead, choose the cross-chain pyr×pur pair whose **C1′–C1′** separation is closest to ~10.5 Å
        (fallback: base heavy-atom COM distance ~6 Å if C1′ is missing).
        If residue names are ambiguous (e.g. both look ``pyr``), infer pur vs pyr by base heavy-atom count.
        """
        from collections import defaultdict

        groups: Dict[Tuple[str, int], List[Tuple[str, TemplateAtom]]] = defaultdict(list)
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                resname = line[17:20].strip()
                chain = (line[21:22] or "A").strip() or "A"
                resseq = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except Exception:
                continue
            atom_name = line[12:16].strip() or "X"
            elem = line[76:78].strip() or atom_name.strip()[0:1] or "C"
            atom = TemplateAtom(name=atom_name, element=elem.upper(), coord=np.array([x, y, z], dtype=np.float64))
            groups[(chain, resseq)].append((resname, atom))
        if len(groups) < 2:
            return None
        items: List[Tuple[Tuple[str, int], str, List[TemplateAtom]]] = []
        for key in sorted(groups.keys(), key=lambda k: (k[0], k[1])):
            lst = groups[key]
            rn = lst[0][0]
            atoms = [t[1] for t in lst]
            items.append((key, rn, atoms))
        pyr_items = [(k, rn, at) for k, rn, at in items if not self._is_purine_resname(rn)]
        pur_items = [(k, rn, at) for k, rn, at in items if self._is_purine_resname(rn)]
        if not pyr_items or not pur_items:
            inferred = self._infer_pyr_pur_sites_from_geometry(items)
            if inferred is None:
                self.session.logger.warning(
                    f"[BaseHunter] templateBP {path.name}: could not identify a cross-chain WC-like pair "
                    "(need two residues on different chains)."
                )
                return None
            self.session.logger.info(
                f"[BaseHunter] templateBP {path.name}: pur/pyr inferred from geometry (ambiguous resnames)."
            )
            return inferred
        target_c1 = 10.55
        target_com = 6.05
        scored: List[Tuple[float, float, List[TemplateAtom], List[TemplateAtom]]] = []
        for _kp, _rnp, at_pyr in pyr_items:
            for _kq, _rnq, at_pur in pur_items:
                if _kp[0] == _kq[0]:
                    continue
                c1p = self._find_c1_prime(at_pyr)
                c1q = self._find_c1_prime(at_pur)
                if c1p is not None and c1q is not None:
                    d = float(np.linalg.norm(c1q - c1p))
                    tgt = target_c1
                    lo, hi = 8.5, 12.5
                else:
                    com_p = self._base_com_from_atoms(at_pyr)
                    com_q = self._base_com_from_atoms(at_pur)
                    d = float(np.linalg.norm(com_q - com_p))
                    tgt = target_com
                    lo, hi = 4.5, 8.5
                if lo < d < hi:
                    score = abs(d - tgt)
                else:
                    score = abs(d - tgt) + 50.0
                scored.append((score, d, at_pyr, at_pur))
        if not scored:
            return None
        scored.sort(key=lambda t: (t[0], t[1]))
        best = scored[0]
        if best[0] >= 40.0:
            self.session.logger.warning(
                f"[BaseHunter] templateBP {path.name}: no WC-like pyr–pur pair in expected distance range; "
                f"using closest-scored pair (d={best[1]:.2f} Å). Check template."
            )
        else:
            self.session.logger.info(
                f"[BaseHunter] templateBP {path.name}: WC dimer from C1′–C1′ (or COM) distance={best[1]:.2f} Å."
            )
        return (best[2], best[3])

    def _load_template_bp_dimer_cached(self) -> Optional[Tuple[List[TemplateAtom], List[TemplateAtom]]]:
        root = Path(self.template_dir.text().strip()).expanduser()
        path = self._template_bp_path(root)
        if path is None:
            return None
        if self._template_bp_dimer_cache_path != path:
            self._template_bp_dimer_cache = None
            self._template_bp_dimer_cache_path = path
        if self._template_bp_dimer_cache is not None:
            return self._template_bp_dimer_cache
        got = self._parse_template_bp_dimer_sites(path)
        self._template_bp_dimer_cache = got
        return got

    @staticmethod
    def _base_com_from_atoms(atoms: List[TemplateAtom]) -> np.ndarray:
        pts = [
            a.coord
            for a in atoms
            if (a.element or "C").upper() != "H" and not BaseHunterInteractiveTool._is_sugar_phosphate_atom_name(a.name)
        ]
        if len(pts) < 2:
            pts = [a.coord for a in atoms if (a.element or "C").upper() != "H"]
        if not pts:
            return np.zeros(3, dtype=np.float64)
        return np.mean(np.stack(pts, axis=0), axis=0)

    @staticmethod
    def _kabsch_rigid_rows(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Least-squares rigid transform R,t with rows of P, Q as corresponding 3D points (N×3)."""
        if P.shape[0] < 2 or P.shape != Q.shape:
            return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        c_p = P.mean(axis=0)
        c_q = Q.mean(axis=0)
        pc = P - c_p
        qc = Q - c_q
        h = pc.T @ qc
        u, _, vt = np.linalg.svd(h)
        rm = vt.T @ u.T
        if np.linalg.det(rm) < 0:
            vt_adj = vt.copy()
            vt_adj[-1, :] *= -1.0
            rm = vt_adj.T @ u.T
        t = c_q - rm @ c_p
        return rm.astype(np.float64), t.astype(np.float64)

    @staticmethod
    def _backbone_target_dict(atoms: List[TemplateAtom], R: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
        """World coords for sugar/phosphate atoms keyed by normalized atom name."""
        want = {"P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"}
        out: Dict[str, np.ndarray] = {}
        for a in atoms:
            k = BaseHunterInteractiveTool._norm_atom_key(a.name)
            if k in want:
                out[k] = R @ a.coord + t
        return out

    def _kabsch_align_full_to_backbone(
        self, full: List[TemplateAtom], target_bb: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        p_rows: List[np.ndarray] = []
        q_rows: List[np.ndarray] = []
        for a in full:
            k = self._norm_atom_key(a.name)
            if k in target_bb:
                p_rows.append(a.coord.copy())
                q_rows.append(target_bb[k].copy())
        if len(p_rows) < 4:
            return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        p_mat = np.stack(p_rows, axis=0)
        q_mat = np.stack(q_rows, axis=0)
        return self._kabsch_rigid_rows(p_mat, q_mat)

    def _place_from_template_bp(
        self,
        la: str,
        lb: str,
        full_a: List[TemplateAtom],
        full_b: List[TemplateAtom],
        pur_xyz: np.ndarray,
        pyr_xyz: np.ndarray,
        a_xyz: np.ndarray,
        b_xyz: np.ndarray,
    ) -> Optional[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, List[TemplateAtom], List[TemplateAtom]]]
    ]:
        """
        Fallback seed: scale ``templateBP`` dimer to marker separation, Kabsch on base COMs,
        then align each reference nucleotide onto the transformed template backbone (sugar/phosphate).
        Returns ``(RA, tA, RB, tB, (r_d, t_d, sc_pyr, sc_pur))`` for writing step-1 debug coordinates.
        """
        dimer = self._load_template_bp_dimer_cached()
        if dimer is None:
            return None
        at_pyr, at_pur = dimer
        com_pyr = self._base_com_from_atoms(at_pyr)
        com_pur = self._base_com_from_atoms(at_pur)
        v_t = com_pur - com_pyr
        v_w = pur_xyz - pyr_xyz
        d_t = float(np.linalg.norm(v_t))
        d_w = float(np.linalg.norm(v_w))
        if d_t < 1e-6 or d_w < 1e-6:
            return None
        mid = 0.5 * (com_pyr + com_pur)
        s = d_w / d_t
        sc_pyr = [TemplateAtom(name=a.name, element=a.element, coord=mid + s * (a.coord - mid)) for a in at_pyr]
        sc_pur = [TemplateAtom(name=a.name, element=a.element, coord=mid + s * (a.coord - mid)) for a in at_pur]
        com_pyr2 = self._base_com_from_atoms(sc_pyr)
        com_pur2 = self._base_com_from_atoms(sc_pur)
        p_mat = np.stack([com_pyr2, com_pur2], axis=0)
        q_mat = np.stack([pyr_xyz, pur_xyz], axis=0)
        r_d, t_d = self._kabsch_rigid_rows(p_mat, q_mat)
        tgt_pyr = self._backbone_target_dict(sc_pyr, r_d, t_d)
        tgt_pur = self._backbone_target_dict(sc_pur, r_d, t_d)
        ra, ta = self._kabsch_align_full_to_backbone(full_a, tgt_pur if self._is_purine_letter(la) else tgt_pyr)
        rb, tb = self._kabsch_align_full_to_backbone(full_b, tgt_pyr if self._is_purine_letter(la) else tgt_pur)
        # Light nudge toward markers — full COM snap fights backbone geometry and shrinks C1'–C1'.
        cba = self._base_centroid_for_build(la, full_a)
        cbb = self._base_centroid_for_build(lb, full_b)
        wn = 0.35
        ta = ta + wn * (a_xyz - (ra @ cba + ta))
        tb = tb + wn * (b_xyz - (rb @ cbb + tb))
        return ra, ta, rb, tb, (r_d, t_d, sc_pyr, sc_pur)

    # --- Phased build: templateBP → reference pur/pyr PDBs → ChimeraX fitmap (debug PDBs per step) ---
    _SUGAR_ALIGN_NAMES = frozenset({"C1'", "C2'", "C3'", "C4'", "O4'"})
    # Non–sugar-phosphate (NSP) heavy atoms for vector/COM anchoring and inclusion metrics.
    _NSP_PURINE_ATOMS = frozenset({"C1'", "C2", "C4", "C5", "C6", "C8", "N1", "N3", "N6", "N7", "N9"})
    _NSP_PYRIMIDINE_ATOMS = frozenset({"C1'", "C2", "C4", "C5", "C6", "O2", "N1", "N3", "N4"})
    _NSP_ALL_CLASS = frozenset(_NSP_PURINE_ATOMS | _NSP_PYRIMIDINE_ATOMS)
    # Sugar–phosphate backbone (SP) for registering class templates onto ``templateBP`` geometry.
    _SP_ALIGN_NAMES = frozenset({"P", "OP1", "OP2", "C2'", "C3'", "C4'", "C5'", "O3'", "O4'", "O5'"})
    # Phase-2 refine: Kabsch rigid body uses **only** SP+sugar atoms (no NSP base-ring names).
    _SP_KABSCH_KEYS = frozenset(_SP_ALIGN_NAMES | _SUGAR_ALIGN_NAMES)
    # Fit-in-map / density-driving atoms: Phase 2 “base + C1′” lists (BASEHUNTER_INTERACTIVE_SPEC §7.4).
    # Pyrimidine note is C-centric (O2); thymine uses O4 instead of O2 — both included.
    _FITMAP_PURINE_ATOMS = frozenset({"C1'", "N9", "C4", "N3", "C2", "N1", "C6", "N6", "C5", "N7", "C8"})
    _FITMAP_PYRIMIDINE_ATOMS = frozenset({"C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6", "O4"})

    def _pair_intermediate_dir(self, root: Path, pair_id: str) -> Path:
        d = root / "basehunter_intermediates" / pair_id.replace("/", "_")
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _transform_template_atoms(atoms: List[TemplateAtom], R: np.ndarray, t: np.ndarray) -> List[TemplateAtom]:
        return [
            TemplateAtom(name=a.name, element=a.element, coord=R @ a.coord + t)
            for a in atoms
        ]

    def _sugar_dict_template_world(self, atoms: List[TemplateAtom], R: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for a in atoms:
            k = self._norm_atom_key(a.name)
            if k in self._SUGAR_ALIGN_NAMES:
                out[k] = R @ a.coord + t
        return out

    def _sp_dict_template_world(self, atoms: List[TemplateAtom], R: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
        """World coords for SP atoms (register ``templateBP-*`` onto ``templateBP`` frame)."""
        out: Dict[str, np.ndarray] = {}
        for a in atoms:
            k = self._norm_atom_key(a.name)
            if k in self._SP_ALIGN_NAMES:
                out[k] = R @ a.coord + t
        return out

    def _nsp_com(self, atoms: List[TemplateAtom], use_purine_nsp_names: bool) -> Optional[np.ndarray]:
        """Centroid of NSP atoms present in ``atoms`` (purine vs pyrimidine name set)."""
        want = self._NSP_PURINE_ATOMS if use_purine_nsp_names else self._NSP_PYRIMIDINE_ATOMS
        pts: List[np.ndarray] = []
        for a in atoms:
            if (a.element or "C").upper() == "H":
                continue
            k = self._norm_atom_key(a.name)
            if k in want:
                pts.append(a.coord.copy())
        if len(pts) < 2:
            for a in atoms:
                if (a.element or "C").upper() == "H":
                    continue
                k = self._norm_atom_key(a.name)
                if k in self._NSP_ALL_CLASS:
                    pts.append(a.coord.copy())
        if len(pts) < 1:
            return None
        return np.mean(np.stack(pts, axis=0), axis=0)

    def _kabsch_align_full_to_sp_target(
        self, full: List[TemplateAtom], target_sp: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Like backbone Kabsch but uses full SP atom name set."""
        p_rows: List[np.ndarray] = []
        q_rows: List[np.ndarray] = []
        for a in full:
            k = self._norm_atom_key(a.name)
            if k in target_sp:
                p_rows.append(a.coord.copy())
                q_rows.append(target_sp[k].copy())
        if len(p_rows) < 4:
            return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        return self._kabsch_rigid_rows(np.stack(p_rows, axis=0), np.stack(q_rows, axis=0))

    def _score_map_mean_above_thr(self, map_model, thr: float, pts: np.ndarray) -> float:
        if map_model is None or pts.shape[0] == 0:
            return 0.0
        dens = self._sample_points_density(map_model, pts)
        above = np.maximum(dens - thr, 0.0)
        return float(np.mean(above) - 0.25 * np.mean(np.maximum(thr - dens, 0.0)))

    def _rank_world_points_density(
        self, map_model: Any, thr: float, pts: np.ndarray, outside_weight: float = 1.35
    ) -> Tuple[int, float, float, float]:
        """Rank base-heavy world points for map agreement (higher tuple compares better).

        Order: count ≥ threshold, threshold score (inside − w×outside), mean density, −mean sub-threshold mass,
        all on in-bounds samples (same edge handling as joint refine).
        """
        if map_model is None or pts.shape[0] == 0:
            return (0, -1e30, 0.0, 0.0)
        dens = self._sample_points_density(map_model, pts)
        mask = self._map_atom_in_bounds_mask(map_model, pts)
        if np.any(mask):
            bar = dens[mask]
        else:
            bar = dens
        if bar.size == 0:
            return (0, -1e30, 0.0, 0.0)
        ins = int(np.sum(bar >= thr))
        outs = int(bar.size - ins)
        dens_s = float(ins - outside_weight * float(outs))
        avg = float(np.mean(bar))
        below = float(np.mean(np.maximum(thr - bar, 0.0)))
        return (ins, dens_s, avg, -below)

    def _frac_heavy_atoms_above_threshold(
        self, map_model, thr: float, world_atoms: List[TemplateAtom]
    ) -> Tuple[float, int, int]:
        """Fraction of non-H atoms with density ≥ ``thr`` (single structure in world coords)."""
        if map_model is None:
            return 0.0, 0, 0
        rows: List[np.ndarray] = []
        for a in world_atoms:
            if (a.element or "C").upper() == "H":
                continue
            rows.append(a.coord)
        if not rows:
            return 0.0, 0, 0
        pxyz = np.stack(rows, axis=0)
        dens = self._sample_points_density(map_model, pxyz)
        n_tot = int(dens.shape[0])
        n_ok = int(np.sum(dens >= thr))
        return (float(n_ok) / float(max(1, n_tot))), n_ok, n_tot

    def _scale_sites_to_marker_distance(
        self, site_a: List[TemplateAtom], site_b: List[TemplateAtom], p0: np.ndarray, p1: np.ndarray, d_target: float
    ) -> Tuple[List[TemplateAtom], List[TemplateAtom], np.ndarray, np.ndarray]:
        """Uniform scale about midpoint of NSP COMs so |COM_B−COM_A| matches marker separation."""
        d_src = float(np.linalg.norm(p1 - p0))
        if d_src < 1e-6:
            return site_a, site_b, p0, p1
        mid = 0.5 * (p0 + p1)
        s = d_target / d_src
        sc_a = [TemplateAtom(name=a.name, element=a.element, coord=mid + s * (a.coord - mid)) for a in site_a]
        sc_b = [TemplateAtom(name=a.name, element=a.element, coord=mid + s * (a.coord - mid)) for a in site_b]
        p0s = mid + s * (p0 - mid)
        p1s = mid + s * (p1 - mid)
        return sc_a, sc_b, p0s, p1s

    def _placement_tuple_from_world(
        self, map_model, thr: float, world_a: List[TemplateAtom], world_b: List[TemplateAtom]
    ) -> Tuple[float, Tuple[int, float, float, float], float, int, int]:
        """Higher is better: prioritize ≥90% in-map, then density rank tuple, then mean density."""
        merged = world_a + world_b
        frac, n_ok, n_tot = self._frac_heavy_atoms_above_threshold(map_model, thr, merged)
        nsp_pts: List[np.ndarray] = []
        for a in merged:
            if (a.element or "C").upper() == "H" or self._is_sugar_phosphate_atom_name(a.name):
                continue
            k = self._norm_atom_key(a.name)
            if k in self._NSP_ALL_CLASS:
                nsp_pts.append(a.coord)
        if len(nsp_pts) < 3:
            pxyz = np.zeros((0, 3), dtype=np.float64)
        else:
            pxyz = np.stack(nsp_pts, axis=0)
        rk = self._rank_world_points_density(map_model, thr, pxyz) if map_model is not None else (0, 0.0, 0.0, 0.0)
        mean_d = float(np.mean(self._sample_points_density(map_model, pxyz))) if pxyz.shape[0] > 0 and map_model else 0.0
        return frac, rk, mean_d, n_ok, n_tot

    def _solve_template_bp_placement(
        self,
        marker_a: np.ndarray,
        marker_b: np.ndarray,
        map_model,
        thr: float,
        dbg: Optional[Path] = None,
        resseq: int = 1,
        *,
        swap_only: Optional[bool] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, List[TemplateAtom], List[TemplateAtom], bool, str]]:
        """Rigid ``templateBP`` (geometry only; e.g. C–C) using NSP COMs + marker vectors.

        Explores **two assignments** unless ``swap_only`` is set: template chain A→marker A / B→B vs A→B / B→A.
        Scales NSP–NSP separation to |marker_b−marker_a|, Kabsch-aligns NSP COMs to chosen markers, then twists
        about the **marker** axis to optimize map overlap and **fraction of heavy atoms ≥ threshold** (target >90%).

        When ``swap_only=False``, only the branch with template chain A at **marker_a** is evaluated so builds match
        the table (Marker A → output chain A).

        Returns ``(R, t, sc_site_a, sc_site_b, swap_markers, summary)`` where ``swap_markers`` means chain A maps to
        marker B (True) or marker A (False).
        """
        dimer = self._load_template_bp_dimer_cached()
        if dimer is None:
            return None
        # Tuple order is historical (pyr_site, pur_site); for ``templateBP`` geometry both are equivalent anchors.
        site_a, site_b = dimer[0], dimer[1]
        # Double-cytosine / WC frame: use pyrimidine NSP names for NSP COM on both sides.
        p0 = self._nsp_com(site_a, False)
        p1 = self._nsp_com(site_b, False)
        if p0 is None or p1 is None:
            p0 = self._base_com_from_atoms(site_a)
            p1 = self._base_com_from_atoms(site_b)
        m_a = np.array(marker_a, dtype=np.float64)
        m_b = np.array(marker_b, dtype=np.float64)
        d_w = float(np.linalg.norm(m_b - m_a))
        if d_w < 1e-6:
            return None
        sc_a, sc_b, p0s, p1s = self._scale_sites_to_marker_distance(site_a, site_b, p0, p1, d_w)

        best_global: Optional[Tuple[Tuple, np.ndarray, np.ndarray, bool, int, str]] = None
        log_lines: List[str] = []

        swap_candidates = (False, True) if swap_only is None else (swap_only,)
        if map_model is None:
            best_nm: Optional[Tuple[float, np.ndarray, np.ndarray, bool, str]] = None
            for swap in swap_candidates:
                q0, q1 = (m_a, m_b) if not swap else (m_b, m_a)
                q_mat = np.stack([q0, q1], axis=0)
                p_mat = np.stack([p0s, p1s], axis=0)
                r0, t0 = self._kabsch_rigid_rows(p_mat, q_mat)
                w_a = self._transform_template_atoms(sc_a, r0, t0)
                w_b = self._transform_template_atoms(sc_b, r0, t0)
                ca = self._nsp_com(w_a, False) or self._base_com_from_atoms(w_a)
                cb = self._nsp_com(w_b, False) or self._base_com_from_atoms(w_b)
                rms = float(np.linalg.norm(ca - q0) + np.linalg.norm(cb - q1))
                label = "swap=False chainA→markerA" if not swap else "swap=True chainA→markerB"
                log_lines.append(f"{label} (no map): NSP-COM↔marker RMS={rms:.3f} Å")
                if best_nm is None or rms < best_nm[0]:
                    best_nm = (rms, r0.copy(), t0.copy(), swap, label)
            if best_nm is None:
                return None
            _, R_ret, t_ret, swap_ret, label_win = best_nm
            summary = "; ".join(log_lines) + f" → chosen {label_win} (no map)"
            self.session.logger.info(f"[BaseHunter] templateBP geometric placement: {summary}")
            return R_ret, t_ret, sc_a, sc_b, swap_ret, summary

        for swap in swap_candidates:
            q0, q1 = (m_a, m_b) if not swap else (m_b, m_a)
            q_mat = np.stack([q0, q1], axis=0)
            p_mat = np.stack([p0s, p1s], axis=0)
            r0, t0 = self._kabsch_rigid_rows(p_mat, q_mat)
            mid_m = 0.5 * (m_a + m_b)
            ax_m = self._normalize(m_b - m_a)
            best_swap: Optional[Tuple[Tuple, np.ndarray, np.ndarray, int]] = None
            for ideg in range(0, 360, 6):
                Rw = self._rot_matrix(ax_m, float(ideg))
                r2 = Rw @ r0
                t2 = Rw @ (t0 - mid_m) + mid_m
                w_a = self._transform_template_atoms(sc_a, r2, t2)
                w_b = self._transform_template_atoms(sc_b, r2, t2)
                frac, rk, mean_d, n_ok, n_tot = self._placement_tuple_from_world(map_model, thr, w_a, w_b)
                # Prefer ≥90% in-map, then density rank, then mean NSP density.
                key = (frac >= 0.90, frac, rk, mean_d)
                if best_swap is None or key > best_swap[0]:
                    best_swap = (key, r2.copy(), t2.copy(), ideg)
            if best_swap is None:
                continue
            _, r_best, t_best, ideg_best = best_swap
            w_a_f = self._transform_template_atoms(sc_a, r_best, t_best)
            w_b_f = self._transform_template_atoms(sc_b, r_best, t_best)
            frac_f, rk_f, mean_f, nok_f, nt_f = self._placement_tuple_from_world(map_model, thr, w_a_f, w_b_f)
            label = "swap=False chainA→markerA" if not swap else "swap=True chainA→markerB"
            log_lines.append(
                f"{label}: twist={ideg_best}° frac≥thr={frac_f:.3f} ({nok_f}/{nt_f}) "
                f"rank={rk_f} meanNSPdens={mean_f:.4f}"
            )
            gkey = (frac_f >= 0.90, frac_f, rk_f, mean_f)
            if best_global is None or gkey > best_global[0]:
                best_global = (gkey, r_best, t_best, swap, ideg_best, label)
            if dbg is not None:
                tag = "step0_swap0_chainA_to_markerA" if not swap else "step0_swap1_chainA_to_markerB"
                self._write_pdb_debug(
                    dbg / f"{tag}_templateBP_twist{ideg_best}.pdb",
                    tag,
                    [
                        ("A", resseq, "DC", sc_a, r_best, t_best),
                        ("B", resseq, "DC", sc_b, r_best, t_best),
                    ],
                )

        if best_global is None:
            return None
        _, R_ret, t_ret, swap_ret, ideg_ret, label_win = best_global
        summary = "; ".join(log_lines) + f" → chosen {label_win} (twist={ideg_ret}°)"
        self.session.logger.info(f"[BaseHunter] templateBP geometric placement: {summary}")
        return R_ret, t_ret, sc_a, sc_b, swap_ret, summary

    @staticmethod
    def _write_pdb_debug(path: Path, title: str, segments: List[Tuple[str, int, str, List[TemplateAtom], np.ndarray, np.ndarray]]) -> None:
        """Write ATOM records: segments = (chain, resseq, resname, template_atoms, R, t)."""
        lines: List[str] = []
        lines.append(f"REMARK {title[:70]}")
        serial = 1
        for chain, resseq, resname, atoms, R, t in segments:
            for a in atoms:
                pos = R @ a.coord + t
                el = (a.element or "C").upper()[:1]
                name = a.name[:4].ljust(4)
                lines.append(
                    f"ATOM  {serial:5d} {name} {resname:>3} {chain}{resseq:4d}    "
                    f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00          {el:>2s}"
                )
                serial += 1
        lines.append("END")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _session_structure_from_template(
        self,
        name: str,
        chain_id: str,
        resseq: int,
        resname: str,
        full: List[TemplateAtom],
        R: np.ndarray,
        t: np.ndarray,
    ):
        from chimerax.atomic import AtomicStructure, Element

        m = AtomicStructure(self.session, name=name)
        res = m.new_residue(resname, chain_id, resseq)
        for ta in full:
            elc = (ta.element or "C").strip().upper()[:1] or "C"
            e = Element.get_element(elc)
            try:
                at = m.new_atom(ta.name, e)
            except Exception:
                at = m.new_atom(ta.name, Element.get_element("C"))
            p = R @ ta.coord + t
            at.coord = [float(p[0]), float(p[1]), float(p[2])]
            res.add_atom(at)
        self.session.models.add([m])
        return m

    def _session_dimer_from_templates(
        self,
        name: str,
        resseq: int,
        resname_a: str,
        full_a: List[TemplateAtom],
        RA: np.ndarray,
        tA: np.ndarray,
        resname_b: str,
        full_b: List[TemplateAtom],
        RB: np.ndarray,
        tB: np.ndarray,
    ):
        """One ``AtomicStructure`` with two residues (chains A/B) for joint rigid ``fitmap``."""
        from chimerax.atomic import AtomicStructure, Element

        m = AtomicStructure(self.session, name=name)
        for chain_id, resname, full, R, t in (
            ("A", resname_a, full_a, RA, tA),
            ("B", resname_b, full_b, RB, tB),
        ):
            res = m.new_residue(resname, chain_id, resseq)
            for ta in full:
                elc = (ta.element or "C").strip().upper()[:1] or "C"
                e = Element.get_element(elc)
                try:
                    at = m.new_atom(ta.name, e)
                except Exception:
                    at = m.new_atom(ta.name, Element.get_element("C"))
                p = R @ ta.coord + t
                at.coord = [float(p[0]), float(p[1]), float(p[2])]
                res.add_atom(at)
        self.session.models.add([m])
        return m

    def _apply_fitmap_joint_to_pair_rt(
        self,
        full_a: List[TemplateAtom],
        RA: np.ndarray,
        tA: np.ndarray,
        full_b: List[TemplateAtom],
        RB: np.ndarray,
        tB: np.ndarray,
        struct,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fold joint rigid ``fitmap`` motion into per-chain (R,t), assuming same model order as :meth:`_session_dimer_from_templates`."""
        P: List[np.ndarray] = []
        for a in full_a:
            P.append(RA @ a.coord + tA)
        for a in full_b:
            P.append(RB @ a.coord + tB)
        atoms_sorted = sorted(
            struct.atoms,
            key=lambda at: (at.residue.chain_id, at.residue.number, self._norm_atom_key(at.name)),
        )
        if len(atoms_sorted) != len(P):
            return RA, tA, RB, tB
        Q = np.stack([np.array(at.coord, dtype=np.float64) for at in atoms_sorted], axis=0)
        Pm = np.stack(P, axis=0)
        Rd, td = self._kabsch_rigid_rows(Pm, Q)
        return Rd @ RA, Rd @ tA + td, Rd @ RB, Rd @ tB + td

    def _coords_by_norm_name(self, struct) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for a in struct.atoms:
            out[self._norm_atom_key(a.name)] = np.array(a.coord, dtype=np.float64)
        return out

    def _try_fitmap_subset(
        self,
        struct,
        map_model,
        thr: float,
        purine_side: bool,
        *,
        move_whole_molecules: bool = True,
        allow_shift: bool = True,
        allow_rotate: bool = True,
        atom_subset: Optional[frozenset] = None,
        envelope: bool = True,
    ) -> bool:
        """ChimeraX Fit in Map on base-heavy atoms (see ``_FITMAP_*``); overlap metric.

        ``envelope=True`` restricts optimization to map isosurface shell (good for rotation-heavy fits).
        ``envelope=False`` allows overlap gradients away from the shell so **translation-only** passes can
        still move a residue that sits just inside/outside the envelope.

        ``move_whole_molecules=True`` (joint / whole-residue passes): rigid motion follows the driving atoms.
        ``False``: only fit atoms move in ChimeraX; fold back with :meth:`_apply_fitmap_to_rt`.

        If ``atom_subset`` is set, it overrides ``purine_side`` for which atom names are fitted.
        """
        if map_model is None:
            return False
        want = (
            atom_subset
            if atom_subset is not None
            else (self._FITMAP_PURINE_ATOMS if purine_side else self._FITMAP_PYRIMIDINE_ATOMS)
        )
        try:
            from chimerax.atomic import Atoms

            alist = [a for a in struct.atoms if self._norm_atom_key(a.name) in want]
            if len(alist) < 3:
                self.session.logger.warning("[BaseHunter] fitmap: fewer than 3 fit atoms; skipping.")
                return False
            atoms_coll = Atoms(alist)
        except Exception as e:
            self.session.logger.warning(f"[BaseHunter] fitmap atom selection failed: {e}")
            return False
        try:
            from chimerax.core.objects import Objects
            from chimerax.map_fit.fitcmd import fitmap

            # fitmap expects a collection exposing .atoms (Objects), not a bare Atoms instance (ChimeraX ≥1.6).
            fit_target = Objects(atoms=atoms_coll)
            fitmap(
                self.session,
                fit_target,
                in_map=map_model,
                metric="overlap",
                envelope=envelope,
                shift=allow_shift,
                rotate=allow_rotate,
                move_whole_molecules=move_whole_molecules,
                max_steps=3000,
            )
            return True
        except Exception as e:
            self.session.logger.warning(f"[BaseHunter] fitmap() API failed ({e}); trying fitmap command line.")
            try:
                from chimerax.core.commands import run

                mid = struct.id_string
                vid = map_model.id_string
                pieces: List[str] = []
                for a in alist:
                    asp = getattr(a, "atomspec", None)
                    if callable(asp):
                        try:
                            asp = asp()
                        except Exception:
                            asp = None
                    if not asp:
                        continue
                    asp = str(asp).strip()
                    if asp.startswith("/"):
                        pieces.append(f"#{mid}{asp}")
                    else:
                        pieces.append(f"#{mid}/{asp}")
                if len(pieces) >= 3:
                    spec = " | ".join(pieces)
                    mwm = "true" if move_whole_molecules else "false"
                    shift_s = "true" if allow_shift else "false"
                    rot_s = "true" if allow_rotate else "false"
                    env_s = "true" if envelope else "false"
                    cmd = (
                        f"fitmap {spec} inMap #{vid} metric overlap envelope {env_s} shift {shift_s} rotate {rot_s} "
                        f"moveWholeMolecules {mwm} maxSteps 3000"
                    )
                    run(self.session, cmd)
                    return True
                self.session.logger.warning(
                    "[BaseHunter] fitmap: could not build per-atom spec strings for command fallback "
                    f"({len(alist)} fit atoms, {len(pieces)} specs)."
                )
                return False
            except Exception as e2:
                self.session.logger.warning(f"[BaseHunter] fitmap command fallback failed: {e2}")
                return False

    @staticmethod
    def _atoms_chain_resseq(struct, chain_id: str, resseq: int) -> List[Any]:
        """All atoms in one residue (chain + residue number) on an :class:`AtomicStructure`."""
        cid = str(chain_id).strip()
        rs = int(resseq)
        return [
            a
            for a in struct.atoms
            if str(a.residue.chain_id).strip() == cid and int(a.residue.number) == rs
        ]

    def _try_fitmap_atom_list(
        self,
        alist: List[Any],
        map_model,
        *,
        move_whole_molecules: bool = False,
        allow_shift: bool = True,
        allow_rotate: bool = True,
        envelope: bool = True,
    ) -> bool:
        """``fitmap`` on a fixed list of atoms (e.g. ``select #m/A``), mirroring the ChimeraX command line."""
        if map_model is None or len(alist) < 3:
            return False
        try:
            from chimerax.atomic import Atoms

            atoms_coll = Atoms(alist)
        except Exception as e:
            self.session.logger.warning(f"[BaseHunter] fitmap atom list failed: {e}")
            return False
        try:
            from chimerax.core.objects import Objects
            from chimerax.map_fit.fitcmd import fitmap

            fit_target = Objects(atoms=atoms_coll)
            fitmap(
                self.session,
                fit_target,
                in_map=map_model,
                metric="overlap",
                envelope=envelope,
                shift=allow_shift,
                rotate=allow_rotate,
                move_whole_molecules=move_whole_molecules,
                max_steps=3000,
            )
            return True
        except Exception as e:
            self.session.logger.warning(f"[BaseHunter] fitmap() API failed ({e}); trying fitmap command line.")
            try:
                from chimerax.core.commands import run

                struct = getattr(alist[0], "structure", None)
                if struct is None:
                    return False
                mid = struct.id_string
                vid = map_model.id_string
                pieces: List[str] = []
                for a in alist:
                    asp = getattr(a, "atomspec", None)
                    if callable(asp):
                        try:
                            asp = asp()
                        except Exception:
                            asp = None
                    if not asp:
                        continue
                    asp = str(asp).strip()
                    if asp.startswith("/"):
                        pieces.append(f"#{mid}{asp}")
                    else:
                        pieces.append(f"#{mid}/{asp}")
                if len(pieces) >= 3:
                    spec = " | ".join(pieces)
                    mwm = "true" if move_whole_molecules else "false"
                    shift_s = "true" if allow_shift else "false"
                    rot_s = "true" if allow_rotate else "false"
                    env_s = "true" if envelope else "false"
                    cmd = (
                        f"fitmap {spec} inMap #{vid} metric overlap envelope {env_s} shift {shift_s} rotate {rot_s} "
                        f"moveWholeMolecules {mwm} maxSteps 3000"
                    )
                    run(self.session, cmd)
                    return True
                self.session.logger.warning(
                    "[BaseHunter] fitmap: could not build per-atom spec strings for command fallback "
                    f"({len(alist)} fit atoms, {len(pieces)} specs)."
                )
                return False
            except Exception as e2:
                self.session.logger.warning(f"[BaseHunter] fitmap command fallback failed: {e2}")
                return False

    def _rt_from_template_to_struct_residue(
        self,
        full: List[TemplateAtom],
        struct,
        chain_id: str,
        resseq: int,
        fallback_R: np.ndarray,
        fallback_t: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rigid (R,t) mapping template coordinates into fitted structure coords for one residue (Kabsch)."""
        cid = str(chain_id).strip()
        rs = int(resseq)
        P_rows: List[np.ndarray] = []
        Q_rows: List[np.ndarray] = []
        for ta in full:
            if (ta.element or "C").upper() == "H":
                continue
            k = self._norm_atom_key(ta.name)
            found = None
            for at in struct.atoms:
                if str(at.residue.chain_id).strip() != cid:
                    continue
                if int(at.residue.number) != rs:
                    continue
                if self._norm_atom_key(at.name) != k:
                    continue
                found = at
                break
            if found is not None:
                P_rows.append(np.asarray(ta.coord, dtype=np.float64))
                Q_rows.append(np.asarray(found.coord, dtype=np.float64))
        if len(P_rows) < 3:
            return fallback_R, fallback_t
        Pm = np.stack(P_rows, axis=0)
        Qm = np.stack(Q_rows, axis=0)
        return self._kabsch_rigid_rows(Pm, Qm)

    def _fix_purine_glycosidic_bonds(self, model) -> None:
        """Drop mistaken C1′–N1 bonds on purines from :meth:`AtomicStructure.connect_structure` and ensure C1′–N9."""
        from chimerax.core.commands import run

        purine_res = frozenset({"DA", "DG", "RA", "RG"})
        for res in model.residues:
            rname = (res.name or "").strip().upper()
            if rname not in purine_res:
                continue
            by_key: Dict[str, Any] = {}
            for a in res.atoms:
                k = self._norm_atom_key(a.name)
                if k not in by_key:
                    by_key[k] = a
            c1p = by_key.get("C1'")
            n9 = by_key.get("N9")
            n1 = by_key.get("N1")
            if c1p is None or n9 is None:
                continue
            mid = model.id_string
            cid = str(res.chain_id).strip()
            nr = res.number
            asp = f"#{mid}/{cid}:{nr}"

            if n1 is not None:
                for b in list(getattr(c1p, "bonds", ())):
                    try:
                        other = b.other_atom(c1p)
                    except Exception:
                        continue
                    if other is n1:
                        try:
                            run(self.session, f"~bond {asp}@C1' {asp}@N1")
                        except Exception:
                            try:
                                b.delete()
                            except Exception:
                                pass
                        break

            bonded_n9 = False
            for b in list(getattr(c1p, "bonds", ())):
                try:
                    if b.other_atom(c1p) is n9:
                        bonded_n9 = True
                        break
                except Exception:
                    continue
            if bonded_n9:
                continue
            d_n9 = float(np.linalg.norm(np.asarray(n9.coord, dtype=np.float64) - np.asarray(c1p.coord, dtype=np.float64)))
            try:
                from chimerax.atomic.struct_edit import add_bond

                add_bond(c1p, n9)
            except Exception:
                try:
                    suf = "" if d_n9 < 2.8 else " reasonable false"
                    run(self.session, f"bond {asp}@C1' {asp}@N9{suf}")
                except Exception:
                    pass
            if d_n9 > 2.4:
                self.session.logger.warning(
                    f"[BaseHunter] Purine {cid}:{nr} ({rname}): C1'–N9 is {d_n9:.2f} Å after glycosidic repair "
                    "(geometry may still need manual adjustment)."
                )

    def _apply_fitmap_to_rt(self, full: List[TemplateAtom], R0: np.ndarray, t0: np.ndarray, struct) -> Tuple[np.ndarray, np.ndarray]:
        """Map pre-fitmap placement (R0,t0) to post-fitmap coordinates (Kabsch over all residue atoms).

        Works for both ``moveWholeMolecules`` modes: whole-molecule motion is already rigid; partial-atom motion
        is folded into one rigid update by least-squares match of full coordinates before vs after.
        """
        world = self._coords_by_norm_name(struct)
        P: List[np.ndarray] = []
        Q: List[np.ndarray] = []
        for a in full:
            k = self._norm_atom_key(a.name)
            if k in world:
                P.append(R0 @ a.coord + t0)
                Q.append(world[k])
        if len(P) < 3:
            return R0, t0
        Rd, td = self._kabsch_rigid_rows(np.stack(P, axis=0), np.stack(Q, axis=0))
        return Rd @ R0, Rd @ t0 + td

    @staticmethod
    def _atom_base_or_c1_density(atom_name: str) -> bool:
        """C1′ plus base (non–sugar/phosphate) atoms for stage-5 map optimization."""
        k = BaseHunterInteractiveTool._norm_atom_key(atom_name)
        if k == "C1'":
            return True
        return not BaseHunterInteractiveTool._is_sugar_phosphate_atom_name(atom_name)

    def _stack_base_c1_world(self, full: List[TemplateAtom], R: np.ndarray, t: np.ndarray) -> np.ndarray:
        rows: List[np.ndarray] = []
        for a in full:
            if not self._atom_base_or_c1_density(a.name):
                continue
            if (a.element or "C").upper() == "H":
                continue
            rows.append(R @ a.coord + t)
        if not rows:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack(rows, axis=0)

    def _refine_joint_translation_base_c1(
        self,
        map_model,
        thr: float,
        full_a: List[TemplateAtom],
        full_b: List[TemplateAtom],
        RA: np.ndarray,
        tA: np.ndarray,
        RB: np.ndarray,
        tB: np.ndarray,
        e1: np.ndarray,
        e2w: np.ndarray,
        e3w: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Identical translation on both chains (keeps WC pair internal distances); optimizes base+C1′ map rank."""
        if map_model is None:
            return RA, tA, RB, tB

        def stacked(t_a: np.ndarray, t_b: np.ndarray) -> np.ndarray:
            pa = self._stack_base_c1_world(full_a, RA, t_a)
            pb = self._stack_base_c1_world(full_b, RB, t_b)
            if pa.shape[0] == 0 or pb.shape[0] == 0:
                return np.zeros((0, 3), dtype=np.float64)
            return np.vstack([pa, pb])

        base = stacked(tA, tB)
        if base.shape[0] == 0:
            return RA, tA, RB, tB
        best_key = self._rank_world_points_density(map_model, thr, base)
        best_tA, best_tB = tA.copy(), tB.copy()
        scales1 = [0.0, 0.2, 0.4, -0.2, -0.4, 0.6, -0.6, 0.8, -0.8]
        scales3 = [0.0, 0.1, -0.1, 0.2, -0.2]
        e1u = self._normalize(e1)
        e2u = self._normalize(e2w)
        e3u = self._normalize(e3w)
        for s1 in scales1:
            for s2 in scales1:
                for s3 in scales3:
                    d = s1 * e1u + s2 * e2u + s3 * e3u
                    t_an = tA + d
                    t_bn = tB + d
                    bpts = stacked(t_an, t_bn)
                    if bpts.shape[0] == 0:
                        continue
                    k = self._rank_world_points_density(map_model, thr, bpts)
                    if k > best_key:
                        best_key = k
                        best_tA, best_tB = t_an.copy(), t_bn.copy()
        return RA, best_tA, RB, best_tB

    def _log_stage_geometry(
        self,
        stage: str,
        pair_id: str,
        la: str,
        lb: str,
        full_a: List[TemplateAtom],
        full_b: List[TemplateAtom],
        RA: np.ndarray,
        tA: np.ndarray,
        RB: np.ndarray,
        tB: np.ndarray,
        n_a0: np.ndarray,
        n_b0: np.ndarray,
    ) -> None:
        """WC geometry diagnostics (C1′–C1′, plane normals, P–P)."""
        _c1a = self._find_c1_prime(full_a)
        _c1b = self._find_c1_prime(full_b)
        if _c1a is None or _c1b is None:
            self.session.logger.info(f"[BaseHunter] {stage} {pair_id}: geometry QC skipped (missing C1′).")
            return
        try:
            c1a_w = RA @ _c1a + tA
            c1b_w = RB @ _c1b + tB
            d_c1 = float(np.linalg.norm(c1a_w - c1b_w))
            na_w = self._normalize(RA @ n_a0)
            nb_w = self._normalize(RB @ n_b0)
            nd = float(np.dot(na_w, nb_w))
            plane_ang = math.degrees(math.acos(min(1.0, max(0.0, abs(nd)))))
            msg = (
                f"{stage} {pair_id} ({la}/{lb}): C1′–C1′={d_c1:.2f} Å, n·n={nd:.3f}, "
                f"plane angle≈{plane_ang:.1f}° (acos|n·n|)"
            )
            pa = self._find_phosphate_p(full_a)
            pb = self._find_phosphate_p(full_b)
            if pa is not None and pb is not None:
                d_pp = float(np.linalg.norm((RA @ pa + tA) - (RB @ pb + tB)))
                msg += f", P–P={d_pp:.2f} Å"
            self.session.logger.info(f"[BaseHunter] {msg}")
        except Exception:
            self.session.logger.info(f"[BaseHunter] {stage} {pair_id}: geometry QC failed.")

    @staticmethod
    def _parse_pdb_all_atoms(path: Path) -> List[TemplateAtom]:
        """Parse every ATOM/HETATM record; coordinates are exactly as in the file."""
        out: List[TemplateAtom] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip() or "X"
            elem = line[76:78].strip()
            if not elem:
                elem = atom_name.strip()[0:1] or "C"
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except Exception:
                continue
            out.append(TemplateAtom(name=atom_name, element=elem.upper(), coord=np.array([x, y, z], dtype=np.float64)))
        return out

    def _nucleotide_template_path(self, root: Path, letter: str) -> Optional[Path]:
        """Resolve full-residue PDB for building one letter (A,G,C,T).

        Per-letter ``referencePDB-{L}.pdb`` is preferred. **Do not** use ``referencePDB-pyrimidine.pdb`` for
        thymine — that file is typically cytosine; T needs ``referencePDB-T.pdb`` or ``templateBP-T2.pdb``.
        """
        L = letter.upper()
        per = root / f"referencePDB-{L}.pdb"
        if per.is_file():
            return per
        p = root / f"templateBP-{L}2.pdb"
        if p.is_file():
            return p
        if L in {"A", "G"}:
            for fn in ("templateBP-purine.pdb", "templatebp-purine.pdb", "referencePDB-purine.pdb"):
                q = root / fn
                if q.is_file():
                    return q
            return None
        if L == "C":
            for fn in ("templateBP-pyrimidine.pdb", "templatebp-pyrimidine.pdb", "referencePDB-pyrimidine.pdb"):
                q = root / fn
                if q.is_file():
                    return q
            return None
        if L == "T":
            qt = root / "referencePDB-T.pdb"
            if qt.is_file():
                return qt
            return None
        return None

    def _base_only_template_path(self, root: Path, letter: str) -> Optional[Path]:
        L = letter.upper()
        p = root / f"templateBP-{L}2-base.pdb"
        if p.is_file():
            return p
        return None

    def _load_nucleotide_for_build(self, letter: str) -> List[TemplateAtom]:
        L = letter.upper()
        root = Path(self.template_dir.text().strip()).expanduser()
        path = self._nucleotide_template_path(root, L)
        pkey = str(path.resolve()) if path is not None and path.is_file() else ""
        key: Tuple[str, str] = (L, pkey)
        if key in self._nucleotide_build_cache:
            return self._nucleotide_build_cache[key]
        if path is None or not path.is_file():
            self._nucleotide_build_cache[key] = []
            return []
        atoms = self._parse_pdb_all_atoms(path)
        self._nucleotide_build_cache[key] = atoms
        return atoms

    def _base_centroid_for_build(self, letter: str, full_atoms: List[TemplateAtom]) -> np.ndarray:
        """Centroid of base heavy atoms: prefer templateBP-{L}2-base.pdb, else base atoms from full template."""
        root = Path(self.template_dir.text().strip()).expanduser()
        bp = self._base_only_template_path(root, letter)
        if bp is not None and bp.is_file():
            coords = []
            for a in self._parse_pdb_all_atoms(bp):
                el = (a.element or "C").upper()
                if el == "H":
                    continue
                coords.append(a.coord)
            if coords:
                return np.mean(np.stack(coords, axis=0), axis=0)
        coords = []
        for a in full_atoms:
            el = (a.element or "C").upper()
            if el == "H" or self._is_sugar_phosphate_atom_name(a.name):
                continue
            coords.append(a.coord)
        if not coords:
            return np.mean(np.stack([a.coord for a in full_atoms], axis=0), axis=0)
        return np.mean(np.stack(coords, axis=0), axis=0)

    @staticmethod
    def _find_c1_prime(full: List[TemplateAtom]) -> Optional[np.ndarray]:
        for a in full:
            s = a.name.strip().upper().replace("*", "'")
            if s == "C1'":
                return a.coord.copy()
        return None

    @staticmethod
    def _find_phosphate_p(full: List[TemplateAtom]) -> Optional[np.ndarray]:
        for a in full:
            s = a.name.strip().upper()
            if s == "P":
                return a.coord.copy()
        return None

    @staticmethod
    def _atom_coord_by_name(full: List[TemplateAtom], want: str) -> Optional[np.ndarray]:
        w = want.strip().upper().replace("*", "'")
        for a in full:
            s = a.name.strip().upper().replace("*", "'")
            if s == w:
                return a.coord.copy()
        return None

    def _wc_base_plane_normal_template(self, letter: str, full: List[TemplateAtom]) -> np.ndarray:
        """WC-oriented base normal: purines from (N9,C4,C6); pyrimidines from (N1,C2,C4); else ring SVD."""
        L = letter.upper()
        pur = L in {"A", "G"}
        if pur:
            p9 = self._atom_coord_by_name(full, "N9")
            c4 = self._atom_coord_by_name(full, "C4")
            c6 = self._atom_coord_by_name(full, "C6")
            if p9 is not None and c4 is not None and c6 is not None:
                return self._normalize(np.cross(c6 - c4, p9 - c4))
        else:
            n1 = self._atom_coord_by_name(full, "N1")
            c2 = self._atom_coord_by_name(full, "C2")
            c4 = self._atom_coord_by_name(full, "C4")
            if n1 is not None and c2 is not None and c4 is not None:
                return self._normalize(np.cross(c4 - c2, n1 - c2))
        base_pts: List[np.ndarray] = []
        for a in full:
            el = (a.element or "C").upper()
            if el == "H" or self._is_sugar_phosphate_atom_name(a.name):
                continue
            base_pts.append(a.coord)
        if len(base_pts) < 3:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return self._template_plane_normal(np.stack(base_pts, axis=0))

    def _ring_normal_and_sugar_inplane(self, letter: str, full: List[TemplateAtom]) -> Tuple[np.ndarray, np.ndarray]:
        """Ring normal from WC atom sets; in-plane 'sugar' direction = projection of C1'-centroid onto ring plane."""
        n = self._wc_base_plane_normal_template(letter, full)
        base_pts: List[np.ndarray] = []
        for a in full:
            el = (a.element or "C").upper()
            if el == "H" or self._is_sugar_phosphate_atom_name(a.name):
                continue
            base_pts.append(a.coord)
        if not base_pts:
            return n, np.array([1.0, 0.0, 0.0], dtype=np.float64)
        c_b = np.mean(np.stack(base_pts, axis=0), axis=0)
        c1 = self._find_c1_prime(full)
        if c1 is None:
            b = self._normalize(np.array([1.0, 0.0, 0.0], dtype=np.float64))
        else:
            v = c1 - c_b
            v = v - n * float(np.dot(n, v))
            nv = float(np.linalg.norm(v))
            b = self._normalize(v) if nv > 1e-9 else self._normalize(np.cross(n, np.array([0.0, 0.0, 1.0])))
        return n, b

    def _pair_world_frame(self, u_ab: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Orthonormal (e1,e2,e3): e1 along A→B, e3 perpendicular to e1 (pair-plane normal), e2 = e3×e1."""
        e1 = self._normalize(u_ab)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(e1, up))) > 0.92:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        e3 = self._normalize(np.cross(up, e1))
        e2 = self._normalize(np.cross(e3, e1))
        return e1, e2, e3

    def _rotation_pair_place(
        self,
        n_t: np.ndarray,
        b_t: np.ndarray,
        e1: np.ndarray,
        e2w: np.ndarray,
        e3w: np.ndarray,
        normal_sign: float,
        sugar_sign: float,
    ) -> np.ndarray:
        """Rigid rotation: ring normal → ±e3, sugar in-plane direction → ±e2 (twist about e3)."""
        target_n = float(normal_sign) * e3w
        R1 = self._rotation_from_to(n_t, target_n)
        b1 = R1 @ b_t
        b1 = self._normalize(b1)
        tgt = float(sugar_sign) * self._normalize(e2w)
        x = float(np.dot(b1, tgt))
        y = float(np.dot(np.cross(b1, tgt), e3w))
        phi = math.degrees(math.atan2(y, x))
        R2 = self._rot_matrix(e3w, phi)
        return R2 @ R1

    @staticmethod
    def _twist_Rt_about_point(R: np.ndarray, t: np.ndarray, pivot: np.ndarray, axis: np.ndarray, degrees: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply rotation about axis through pivot: x' = Rk @ x + tk with Rk = R_axis @ R."""
        Rk = BaseHunterInteractiveTool._rot_matrix(axis, degrees)
        return Rk @ R, Rk @ (t - pivot) + pivot

    def _refine_rotation_about_e3_at_marker(
        self,
        map_model,
        coords_template: np.ndarray,
        R0: np.ndarray,
        t0: np.ndarray,
        marker: np.ndarray,
        e3w: np.ndarray,
        thr: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """±5° about pair normal e3 through marker (keeps base centroid on marker)."""
        if map_model is None or coords_template.shape[0] == 0:
            return R0, t0
        pivot = np.array(marker, dtype=np.float64)
        axis = self._normalize(e3w)
        best_R, best_t = R0, t0
        best_score = -1e30
        for ddeg in (-5.0, 0.0, 5.0):
            R_try, t_try = self._twist_Rt_about_point(R0, t0, pivot, axis, ddeg)
            pts = (coords_template @ R_try.T) + t_try[None, :]
            dens = self._sample_points_density(map_model, pts)
            above = np.maximum(dens - thr, 0.0)
            below = np.maximum(thr - dens, 0.0)
            score = float(np.mean(above) - 0.25 * np.mean(below))
            if score > best_score:
                best_score = score
                best_R, best_t = R_try, t_try
        return best_R, best_t

    def _joint_pair_twist_refine(
        self,
        map_model,
        full_a: List[TemplateAtom],
        full_b: List[TemplateAtom],
        ca: np.ndarray,
        cb: np.ndarray,
        n_a0: np.ndarray,
        n_b0: np.ndarray,
        c1_a0: np.ndarray,
        c1_b0: np.ndarray,
        p_a0: Optional[np.ndarray],
        p_b0: Optional[np.ndarray],
        RA0: np.ndarray,
        tA0: np.ndarray,
        RB0: np.ndarray,
        tB0: np.ndarray,
        a_xyz: np.ndarray,
        b_xyz: np.ndarray,
        e3w: np.ndarray,
        thr: float,
        w_c1: float = 0.35,
        w_pp: float = 0.02,
        w_plane: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Twist A and B about e3 through respective markers to target C1'–C1' ~10.5 Å, map fit, soft P–P."""
        axis = self._normalize(e3w)
        target_c1 = 10.5
        target_pp = 18.75
        cos5 = math.cos(math.radians(5.0))

        def score_pair(RA: np.ndarray, tA: np.ndarray, RB: np.ndarray, tB: np.ndarray) -> float:
            c1a = (RA @ c1_a0) + tA
            c1b = (RB @ c1_b0) + tB
            d_c1 = float(np.linalg.norm(c1a - c1b))
            na = self._normalize(RA @ n_a0)
            nb = self._normalize(RB @ n_b0)
            nd = float(np.dot(na, nb))
            # Coplanar WC: |n·n| ≈ 1 (parallel or anti-parallel ring normals).
            plane_pen = max(0.0, cos5 - abs(nd))
            c1_pen = abs(d_c1 - target_c1)
            pp_pen = 0.0
            if p_a0 is not None and p_b0 is not None:
                pa = (RA @ p_a0) + tA
                pb = (RB @ p_b0) + tB
                pp_pen = abs(float(np.linalg.norm(pa - pb)) - target_pp)
            map_term = 0.0
            if map_model is not None:
                pts = np.vstack([(ca @ RA.T) + tA[None, :], (cb @ RB.T) + tB[None, :]])
                dens = self._sample_points_density(map_model, pts)
                above = np.maximum(dens - thr, 0.0)
                below = np.maximum(thr - dens, 0.0)
                map_term = float(np.mean(above) - 0.25 * np.mean(below))
            return map_term - w_c1 * c1_pen - w_plane * plane_pen - w_pp * pp_pen

        s0 = score_pair(RA0, tA0, RB0, tB0)
        best = (s0, RA0, tA0, RB0, tB0)
        # Coarse joint twist (preserves normals vs e3 when both were aligned to e3).
        for da in (-20.0, -10.0, 0.0, 10.0, 20.0):
            RA, tA = self._twist_Rt_about_point(RA0, tA0, a_xyz, axis, da)
            for db in range(-60, 61, 5):
                RB, tB = self._twist_Rt_about_point(RB0, tB0, b_xyz, axis, float(db))
                s = score_pair(RA, tA, RB, tB)
                if s > best[0]:
                    best = (s, RA, tA, RB, tB)
        _, RA, tA, RB, tB = best
        best = (score_pair(RA, tA, RB, tB), RA, tA, RB, tB)
        # Local polish ±5° on each axis at best.
        for _ in range(2):
            improved = False
            for da in (-5.0, 0.0, 5.0):
                for db in (-5.0, 0.0, 5.0):
                    RA2, tA2 = self._twist_Rt_about_point(RA, tA, a_xyz, axis, da)
                    RB2, tB2 = self._twist_Rt_about_point(RB, tB, b_xyz, axis, db)
                    s = score_pair(RA2, tA2, RB2, tB2)
                    if s > best[0]:
                        best = (s, RA2, tA2, RB2, tB2)
                        improved = True
            _, RA, tA, RB, tB = best
            if not improved:
                break
        return RA, tA, RB, tB

    def _pair_score_build(
        self,
        map_model,
        ca: np.ndarray,
        cb: np.ndarray,
        RA: np.ndarray,
        tA: np.ndarray,
        RB: np.ndarray,
        tB: np.ndarray,
        n_a0: np.ndarray,
        n_b0: np.ndarray,
        c1_a0: np.ndarray,
        c1_b0: np.ndarray,
        p_a0: Optional[np.ndarray],
        p_b0: Optional[np.ndarray],
        thr: float,
        w_c1: float,
        w_pp: float,
        w_plane: float,
        w_marker: float,
        a_xyz: np.ndarray,
        b_xyz: np.ndarray,
        cba: np.ndarray,
        cbb: np.ndarray,
        plane_mode: str = "wc",
    ) -> float:
        """Higher is better: map fit minus distance / planarity / marker penalties."""
        target_c1 = 10.55
        target_pp = 18.75
        cos5 = math.cos(math.radians(5.0))
        c1a = (RA @ c1_a0) + tA
        c1b = (RB @ c1_b0) + tB
        d_c1 = float(np.linalg.norm(c1a - c1b))
        na = self._normalize(RA @ n_a0)
        nb = self._normalize(RB @ n_b0)
        nd = float(np.dot(na, nb))
        if plane_mode == "coplanar":
            # Drive |n·n| → 1 (parallel or anti-parallel ring normals).
            plane_pen = 1.0 - abs(nd)
        else:
            plane_pen = max(0.0, cos5 - abs(nd))
        c1_pen = abs(d_c1 - target_c1)
        pp_pen = 0.0
        if p_a0 is not None and p_b0 is not None:
            pa = (RA @ p_a0) + tA
            pb = (RB @ p_b0) + tB
            pp_pen = abs(float(np.linalg.norm(pa - pb)) - target_pp)
        m_a = float(np.linalg.norm((RA @ cba + tA) - a_xyz))
        m_b = float(np.linalg.norm((RB @ cbb + tB) - b_xyz))
        map_term = 0.0
        if map_model is not None:
            pts = np.vstack([(ca @ RA.T) + tA[None, :], (cb @ RB.T) + tB[None, :]])
            dens = self._sample_points_density(map_model, pts)
            above = np.maximum(dens - thr, 0.0)
            below = np.maximum(thr - dens, 0.0)
            map_term = float(np.mean(above) - 0.25 * np.mean(below))
        return (
            map_term
            - w_c1 * c1_pen
            - w_pp * pp_pen
            - w_plane * plane_pen
            - w_marker * (m_a + m_b)
        )

    def _refine_separation_along_e1(
        self,
        map_model,
        ca: np.ndarray,
        cb: np.ndarray,
        RA: np.ndarray,
        tA: np.ndarray,
        RB: np.ndarray,
        tB: np.ndarray,
        n_a0: np.ndarray,
        n_b0: np.ndarray,
        c1_a0: np.ndarray,
        c1_b0: np.ndarray,
        p_a0: Optional[np.ndarray],
        p_b0: Optional[np.ndarray],
        e1_axis: np.ndarray,
        thr: float,
        a_xyz: np.ndarray,
        b_xyz: np.ndarray,
        cba: np.ndarray,
        cbb: np.ndarray,
        step: float,
        lim: float,
        emphasis: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Translate A along −e1 and B along +e1 to open C1'–C1' and P–P toward B-DNA targets."""
        e1u = self._normalize(e1_axis)
        w_c1 = 1.45 if emphasis else 1.0
        w_pp = 0.24 if emphasis else 0.16
        w_plane = 2.0
        w_marker = 0.1 if emphasis else 0.14
        best = (
            self._pair_score_build(
                map_model,
                ca,
                cb,
                RA,
                tA,
                RB,
                tB,
                n_a0,
                n_b0,
                c1_a0,
                c1_b0,
                p_a0,
                p_b0,
                thr,
                w_c1,
                w_pp,
                w_plane,
                w_marker,
                a_xyz,
                b_xyz,
                cba,
                cbb,
                "wc",
            ),
            RA,
            tA,
            RB,
            tB,
        )
        da = -lim
        nstep = int(round((2.0 * lim) / max(1e-6, step))) + 1
        for _ in range(max(1, nstep)):
            tA2 = tA - e1u * da
            tB2 = tB + e1u * da
            s = self._pair_score_build(
                map_model,
                ca,
                cb,
                RA,
                tA2,
                RB,
                tB2,
                n_a0,
                n_b0,
                c1_a0,
                c1_b0,
                p_a0,
                p_b0,
                thr,
                w_c1,
                w_pp,
                w_plane,
                w_marker,
                a_xyz,
                b_xyz,
                cba,
                cbb,
                "wc",
            )
            if s > best[0]:
                best = (s, RA, tA2, RB, tB2)
            da += step
        return best[1], best[2], best[3], best[4]

    def _refine_roll_about_ab_axis(
        self,
        map_model,
        ca: np.ndarray,
        cb: np.ndarray,
        RA: np.ndarray,
        tA: np.ndarray,
        RB: np.ndarray,
        tB: np.ndarray,
        n_a0: np.ndarray,
        n_b0: np.ndarray,
        c1_a0: np.ndarray,
        c1_b0: np.ndarray,
        p_a0: Optional[np.ndarray],
        p_b0: Optional[np.ndarray],
        a_xyz: np.ndarray,
        b_xyz: np.ndarray,
        thr: float,
        cba: np.ndarray,
        cbb: np.ndarray,
        from_at: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Rotate whole dimer about line A–B through midpoint (fixes ~90° errors from 2-point AT registration)."""
        m = 0.5 * (a_xyz + b_xyz)
        e1 = self._normalize(b_xyz - a_xyz)
        w_c1, w_pp, w_marker = 0.55, 0.1, 0.11
        w_plane = 6.0 if from_at else 3.5
        mode = "coplanar" if from_at else "wc"

        def sc(RAa: np.ndarray, tAa: np.ndarray, RBb: np.ndarray, tBb: np.ndarray) -> float:
            return self._pair_score_build(
                map_model,
                ca,
                cb,
                RAa,
                tAa,
                RBb,
                tBb,
                n_a0,
                n_b0,
                c1_a0,
                c1_b0,
                p_a0,
                p_b0,
                thr,
                w_c1,
                w_pp,
                w_plane,
                w_marker,
                a_xyz,
                b_xyz,
                cba,
                cbb,
                mode,
            )

        ra0, ta0, rb0, tb0 = RA, tA, RB, tB
        best_deg = 0.0
        best = (sc(ra0, ta0, rb0, tb0), ra0, ta0, rb0, tb0, 0.0)
        for ideg in range(0, 360, 12):
            deg = float(ideg)
            rg = self._rot_matrix(e1, deg)
            ra2 = rg @ ra0
            ta2 = rg @ (ta0 - m) + m
            rb2 = rg @ rb0
            tb2 = rg @ (tb0 - m) + m
            s = sc(ra2, ta2, rb2, tb2)
            if s > best[0]:
                best = (s, ra2, ta2, rb2, tb2, deg)
        best_deg = float(best[5])
        for ddel in range(-10, 11, 2):
            deg = (best_deg + float(ddel)) % 360.0
            rg = self._rot_matrix(e1, deg)
            ra2 = rg @ ra0
            ta2 = rg @ (ta0 - m) + m
            rb2 = rg @ rb0
            tb2 = rg @ (tb0 - m) + m
            s = sc(ra2, ta2, rb2, tb2)
            if s > best[0]:
                best = (s, ra2, ta2, rb2, tb2, deg)
        _, RA, tA, RB, tB, best_deg = best
        if from_at and abs(best_deg) > 0.5:
            self.session.logger.info(f"[BaseHunter] Roll about A–B axis: {best_deg:.1f}° (map + coplanarity).")
        return RA, tA, RB, tB

    @staticmethod
    def _flip_rt_about_ab_axis(
        RA: np.ndarray,
        tA: np.ndarray,
        RB: np.ndarray,
        tB: np.ndarray,
        a_xyz: np.ndarray,
        b_xyz: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """180° rotation about the line A–B through its midpoint (both residues)."""
        m = 0.5 * (a_xyz + b_xyz)
        e1 = BaseHunterInteractiveTool._normalize(b_xyz - a_xyz)
        rf = BaseHunterInteractiveTool._rot_matrix(e1, 180.0)
        ta2 = rf @ (tA - m) + m
        tb2 = rf @ (tB - m) + m
        return rf @ RA, ta2, rf @ RB, tb2

    @staticmethod
    def _fallback_voxel_step(di: int, dj: int, dk: int, data: Any) -> np.ndarray:
        s = getattr(data, "step", None) if data is not None else None
        if s is not None and len(s) >= 3:
            return np.array(
                [float(di) * float(s[0]), float(dj) * float(s[1]), float(dk) * float(s[2])],
                dtype=np.float64,
            )
        return np.array([float(di), float(dj), float(dk)], dtype=np.float64)

    def _voxel_offset_vectors_scene(self, map_model: Any, ref_xyz: np.ndarray, voxel_radius: int = 1) -> List[np.ndarray]:
        """Scene-space offsets for integer voxel steps (di,dj,dk) with max |·| ≤ voxel_radius on the map grid."""
        data = getattr(map_model, "data", None)
        offs: List[np.ndarray] = []
        ref = np.asarray(ref_xyz, dtype=np.float64).ravel()[:3]
        r = max(0, int(voxel_radius))
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                for dk in range(-r, r + 1):
                    dv = None
                    if data is not None and hasattr(data, "xyz_to_ijk") and hasattr(data, "ijk_to_xyz"):
                        try:
                            ijk0 = np.array(data.xyz_to_ijk(list(ref)), dtype=np.float64).ravel()[:3]
                            p0 = np.array(data.ijk_to_xyz(ijk0), dtype=np.float64).ravel()[:3]
                            p1 = np.array(
                                data.ijk_to_xyz(ijk0 + np.array([float(di), float(dj), float(dk)], dtype=np.float64)),
                                dtype=np.float64,
                            ).ravel()[:3]
                            dv = p1 - p0
                        except Exception:
                            dv = None
                    if dv is None:
                        dv = self._fallback_voxel_step(di, dj, dk, data)
                    offs.append(np.asarray(dv, dtype=np.float64).ravel()[:3])
        return offs

    def _density_threshold_score(
        self,
        map_model: Any,
        thr: float,
        ca: np.ndarray,
        RA: np.ndarray,
        tA: np.ndarray,
        cb: np.ndarray,
        RB: np.ndarray,
        tB: np.ndarray,
        outside_weight: float = 1.35,
    ) -> Tuple[float, int, int]:
        """Score = n_inside − w*n_out at threshold (higher is better)."""
        pts = np.vstack([(ca @ RA.T) + tA, (cb @ RB.T) + tB])
        dens = self._sample_points_density(map_model, pts)
        ins = int(np.sum(dens >= thr))
        out = int(np.sum(dens < thr))
        return float(ins - outside_weight * float(out)), ins, out

    @staticmethod
    def _dimer_com_world(
        ca: np.ndarray,
        RA: np.ndarray,
        tA: np.ndarray,
        cb: np.ndarray,
        RB: np.ndarray,
        tB: np.ndarray,
    ) -> np.ndarray:
        """Geometric center of all template atoms (world), ChimeraX Fit-in-Map rotation pivot."""
        pa = (ca @ RA.T) + tA
        pb = (cb @ RB.T) + tB
        allp = np.vstack([pa, pb])
        return np.mean(allp, axis=0)

    @staticmethod
    def _apply_joint_rigid(
        RA: np.ndarray,
        tA: np.ndarray,
        RB: np.ndarray,
        tB: np.ndarray,
        Rout: np.ndarray,
        dt: np.ndarray,
        pivot: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Rigid motion of the dimer: x' = Rout @ (x - pivot) + pivot + dt (preserves P–P, C1′–C1′, etc.)."""
        p = np.asarray(pivot, dtype=np.float64).ravel()[:3]
        d = np.asarray(dt, dtype=np.float64).ravel()[:3]
        RA2 = Rout @ RA
        tA2 = Rout @ (tA - p) + p + d
        RB2 = Rout @ RB
        tB2 = Rout @ (tB - p) + p + d
        return RA2, tA2, RB2, tB2

    def _map_atom_in_bounds_mask(self, map_model: Any, pts_xyz: np.ndarray, margin_vox: float = 1.0) -> np.ndarray:
        """Atoms away from map grid faces (ChimeraX omits near-edge points from fit average)."""
        n = int(pts_xyz.shape[0])
        mask = np.ones(n, dtype=bool)
        data = getattr(map_model, "data", None)
        if data is None or not hasattr(data, "xyz_to_ijk"):
            return mask
        try:
            full = data.full_matrix()
            size = getattr(data, "size", None)
            if size is not None and len(size) >= 3:
                nx, ny, nz = int(size[0]), int(size[1]), int(size[2])
            else:
                nz, ny, nx = int(full.shape[0]), int(full.shape[1]), int(full.shape[2])
            m = float(margin_vox)
            for i in range(n):
                ijk = np.array(data.xyz_to_ijk(list(pts_xyz[i])), dtype=np.float64).ravel()[:3]
                ii, jj, kk = float(ijk[0]), float(ijk[1]), float(ijk[2])
                if size is not None and len(size) >= 3:
                    ok = m <= ii < nx - m and m <= jj < ny - m and m <= kk < nz - m
                else:
                    ok = m <= ii < nx - m and m <= jj < ny - m and m <= kk < nz - m
                if not ok:
                    mask[i] = False
        except Exception:
            return np.ones(n, dtype=bool)
        return mask

    def _fitmap_style_scores(
        self,
        map_model: Any,
        thr: float,
        ca: np.ndarray,
        RA: np.ndarray,
        tA: np.ndarray,
        cb: np.ndarray,
        RB: np.ndarray,
        tB: np.ndarray,
        edge_vox: float = 1.0,
    ) -> Dict[str, float]:
        """Atoms-in-map style: mean interpolated density over in-bounds atoms (ChimeraX Fit in Map)."""
        pts = np.vstack([(ca @ RA.T) + tA, (cb @ RB.T) + tB])
        dens = self._sample_points_density(map_model, pts)
        mask = self._map_atom_in_bounds_mask(map_model, pts, edge_vox)
        if np.any(mask):
            bar = dens[mask]
            n_ib = int(np.sum(mask))
        else:
            bar = dens
            n_ib = int(bar.shape[0])
        avg = float(np.mean(bar)) if bar.size else 0.0
        inside = int(np.sum(bar >= thr))
        below = float(np.mean(np.maximum(thr - bar, 0.0))) if bar.size else 0.0
        ins_all = int(np.sum(dens >= thr))
        outs_all = int(np.sum(dens < thr))
        return {
            "avg": avg,
            "inside": float(inside),
            "below": below,
            "n_ib": float(n_ib),
            "ins_all": float(ins_all),
            "outs_all": float(outs_all),
        }

    def _density_fit_rank_key(
        self,
        map_model: Any,
        thr: float,
        ca: np.ndarray,
        RA: np.ndarray,
        tA: np.ndarray,
        cb: np.ndarray,
        RB: np.ndarray,
        tB: np.ndarray,
        w_out: float = 1.35,
    ) -> Tuple[int, float, float, float]:
        """Lexicographic ranking: prefer more atoms ≥ thr, then thr-score, fitmap mean, then lower sub-thr mass."""
        dens_s, ins, _outs = self._density_threshold_score(map_model, thr, ca, RA, tA, cb, RB, tB, w_out)
        fs = self._fitmap_style_scores(map_model, thr, ca, RA, tA, cb, RB, tB)
        return (int(ins), float(dens_s), float(fs["avg"]), float(-fs["below"]))

    @staticmethod
    def _angle_grid_deg(ang_min: float, ang_max: float, step: float) -> List[float]:
        if step <= 0:
            return [float(ang_min)]
        n = int(math.floor((ang_max - ang_min) / step + 1e-9)) + 1
        n = max(1, n)
        out = [ang_min + i * step for i in range(n) if ang_min + i * step <= ang_max + 1e-6]
        if not out:
            return [float(ang_min)]
        return out

    def _geometry_penalties_build(
        self,
        RA: np.ndarray,
        tA: np.ndarray,
        RB: np.ndarray,
        tB: np.ndarray,
        n_a0: np.ndarray,
        n_b0: np.ndarray,
        c1_a0: np.ndarray,
        c1_b0: np.ndarray,
        p_a0: Optional[np.ndarray],
        p_b0: Optional[np.ndarray],
        plane_mode: str,
    ) -> Dict[str, float]:
        """C1′–C1′, P–P, and base-plane penalties (same targets as _pair_score_build)."""
        target_c1 = 10.55
        target_pp = 18.75
        cos5 = math.cos(math.radians(5.0))
        c1a = (RA @ c1_a0) + tA
        c1b = (RB @ c1_b0) + tB
        d_c1 = float(np.linalg.norm(c1a - c1b))
        c1_pen = abs(d_c1 - target_c1)
        na = self._normalize(RA @ n_a0)
        nb = self._normalize(RB @ n_b0)
        nd = float(np.dot(na, nb))
        if plane_mode == "coplanar":
            plane_pen = 1.0 - abs(nd)
        else:
            plane_pen = max(0.0, cos5 - abs(nd))
        pp_pen = 0.0
        if p_a0 is not None and p_b0 is not None:
            pa = (RA @ p_a0) + tA
            pb = (RB @ p_b0) + tB
            pp_pen = abs(float(np.linalg.norm(pa - pb)) - target_pp)
        return {"c1_pen": c1_pen, "pp_pen": pp_pen, "plane_pen": plane_pen, "abs_nd": abs(nd)}

    def _refine_local_density_hierarchical(
        self,
        map_model: Any,
        ca: np.ndarray,
        cb: np.ndarray,
        RA: np.ndarray,
        tA: np.ndarray,
        RB: np.ndarray,
        tB: np.ndarray,
        c1_a0: np.ndarray,
        c1_b0: np.ndarray,
        n_a0: np.ndarray,
        n_b0: np.ndarray,
        p_a0: Optional[np.ndarray],
        p_b0: Optional[np.ndarray],
        e1w: np.ndarray,
        e2w: np.ndarray,
        e3w: np.ndarray,
        thr: float,
        ref_xyz: np.ndarray,
        plane_mode: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Joint rigid-body refine (like ChimeraX Fit in Map): same translation on both residues, rotations about dimer COM.

        Ranks poses by: atoms ≥ map threshold, then thr-score, fitmap mean, then sub-threshold mass (see ChimeraX fitmap).
        Internal geometry (P–P, C1′–C1′, ring normals) is unchanged because the dimer moves as one rigid body.
        """
        w_out = 1.35
        ax1 = self._normalize(e1w)
        ax2 = self._normalize(e2w)
        ax3 = self._normalize(e3w)
        axis_by_name = {"e1": ax1, "e2": ax2, "e3": ax3}
        I3 = np.eye(3, dtype=np.float64)
        z3 = np.zeros(3, dtype=np.float64)

        def fit_key(ra: np.ndarray, ta: np.ndarray, rb: np.ndarray, tb: np.ndarray) -> Tuple[int, float, float, float]:
            return self._density_fit_rank_key(map_model, thr, ca, ra, ta, cb, rb, tb, w_out)

        best_key = fit_key(RA, tA, RB, tB)
        best_RA, best_tA, best_RB, best_tB = RA.copy(), tA.copy(), RB.copy(), tB.copy()

        def accept(ra: np.ndarray, ta: np.ndarray, rb: np.ndarray, tb: np.ndarray) -> None:
            nonlocal best_key, best_RA, best_tA, best_RB, best_tB
            k = fit_key(ra, ta, rb, tb)
            if k > best_key:
                best_key = k
                best_RA, best_tA, best_RB, best_tB = ra.copy(), ta.copy(), rb.copy(), tb.copy()

    # (voxel_radius, list of (axis_name, ang_min, ang_max, ang_step)); axis_name e1|e2|e3 in pair frame.
        stage_axes: List[Tuple[int, List[Tuple[str, float, float, float]]]] = [
            (2, [("e3", -15.0, 15.0, 1.0)]),
            (1, [("e3", -5.0, 5.0, 0.5), ("e1", -2.5, 2.5, 0.5), ("e2", -2.5, 2.5, 0.5)]),
            (1, [("e3", -2.0, 2.0, 0.2), ("e1", -1.2, 1.2, 0.2), ("e2", -1.2, 1.2, 0.2)]),
        ]
        angs_cache: Dict[Tuple[float, float, float], List[float]] = {}

        for ir, (vox_r, ax_specs) in enumerate(stage_axes):
            offs = self._voxel_offset_vectors_scene(map_model, ref_xyz, voxel_radius=vox_r)
            for _ in range(2):
                ra_s, ta_s, rb_s, tb_s = best_RA.copy(), best_tA.copy(), best_RB.copy(), best_tB.copy()
                for dv in offs:
                    ra2, ta2, rb2, tb2 = self._apply_joint_rigid(ra_s, ta_s, rb_s, tb_s, I3, dv, z3)
                    accept(ra2, ta2, rb2, tb2)
                for aname, amin, amax, astep in ax_specs:
                    ax_u = axis_by_name[aname]
                    key = (amin, amax, astep)
                    if key not in angs_cache:
                        angs_cache[key] = self._angle_grid_deg(amin, amax, astep)
                    angles = angs_cache[key]
                    ra0, ta0, rb0, tb0 = best_RA.copy(), best_tA.copy(), best_RB.copy(), best_tB.copy()
                    piv = self._dimer_com_world(ca, ra0, ta0, cb, rb0, tb0)
                    for ang in angles:
                        Rk = self._rot_matrix(ax_u, float(ang))
                        ra2, ta2, rb2, tb2 = self._apply_joint_rigid(ra0, ta0, rb0, tb0, Rk, z3, piv)
                        accept(ra2, ta2, rb2, tb2)

            fs = self._fitmap_style_scores(map_model, thr, ca, best_RA, best_tA, cb, best_RB, best_tB)
            dens_s, ins, outs = self._density_threshold_score(map_model, thr, ca, best_RA, best_tA, cb, best_RB, best_tB, w_out)
            g = self._geometry_penalties_build(best_RA, best_tA, best_RB, best_tB, n_a0, n_b0, c1_a0, c1_b0, p_a0, p_b0, plane_mode)
            ax_desc = ", ".join(f"{nm}{lo:g}…{hi:g}°/{st:g}" for nm, lo, hi, st in ax_specs)
            self.session.logger.info(
                f"[BaseHunter] Density refine stage {ir + 1}/3 (joint rigid, vox±{vox_r}, axes {ax_desc}): "
                f"fitmap avg {fs['avg']:.5f}, in-bounds {int(fs['n_ib'])}/{int(ca.shape[0] + cb.shape[0])}, "
                f"≥thr {int(fs['inside'])}, sub-thr mean {fs['below']:.4f}; "
                f"thr-score {dens_s:.1f} (≥thr {ins}, <thr {outs}); "
                f"C1′-pen {g['c1_pen']:.3f} Å, PP-pen {g['pp_pen']:.3f} Å"
            )

        fs = self._fitmap_style_scores(map_model, thr, ca, best_RA, best_tA, cb, best_RB, best_tB)
        dens_s, ins, outs = self._density_threshold_score(map_model, thr, ca, best_RA, best_tA, cb, best_RB, best_tB, w_out)
        self.session.logger.info(
            f"[BaseHunter] Local density refine done (joint rigid / fitmap-style): avg {fs['avg']:.5f}, "
            f"thr-score {dens_s:.1f} (≥thr {ins}, <thr {outs})."
        )
        return best_RA, best_tA, best_RB, best_tB

    def _refine_micro_twist_e3_each_c1_prime(
        self,
        map_model: Any,
        ca: np.ndarray,
        cb: np.ndarray,
        RA: np.ndarray,
        tA: np.ndarray,
        RB: np.ndarray,
        tB: np.ndarray,
        c1_a0: np.ndarray,
        c1_b0: np.ndarray,
        n_a0: np.ndarray,
        n_b0: np.ndarray,
        p_a0: Optional[np.ndarray],
        p_b0: Optional[np.ndarray],
        e1w: np.ndarray,
        e2w: np.ndarray,
        e3w: np.ndarray,
        thr: float,
        plane_mode: str,
        ang_max_deg: float = 2.5,
        ang_step_deg: float = 0.25,
        n_passes: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tiny per-residue rotations through each C1′ (no translation): e3 then small e2/e1 in the pair frame.

        e2/e1 swings backbone/sugar toward density without large independent buckles. Tight geometry caps vs. entry pose.
        Ranking matches joint refine (≥thr count first, then thr-score, fitmap avg, sub-thr mass).
        """
        ax1 = self._normalize(e1w)
        ax2 = self._normalize(e2w)
        ax3 = self._normalize(e3w)
        w_out = 1.35
        has_pp = p_a0 is not None and p_b0 is not None
        m0 = self._geometry_penalties_build(RA, tA, RB, tB, n_a0, n_b0, c1_a0, c1_b0, p_a0, p_b0, plane_mode)
        slack_c1, slack_pp, slack_plane = 0.11, 0.14, 0.05
        pp_cap = min(m0["pp_pen"] + slack_pp, max(0.36, m0["pp_pen"])) if has_pp else float("inf")
        c1_cap = min(m0["c1_pen"] + slack_c1, max(0.32, m0["c1_pen"]))

        def feasible(m: Dict[str, float]) -> bool:
            if m["c1_pen"] > c1_cap + 1e-9:
                return False
            if has_pp and m["pp_pen"] > pp_cap + 1e-9:
                return False
            if m["plane_pen"] > m0["plane_pen"] + slack_plane:
                return False
            return True

        def fit_key(ra: np.ndarray, ta: np.ndarray, rb: np.ndarray, tb: np.ndarray) -> Optional[Tuple[int, float, float, float]]:
            g = self._geometry_penalties_build(ra, ta, rb, tb, n_a0, n_b0, c1_a0, c1_b0, p_a0, p_b0, plane_mode)
            if not feasible(g):
                return None
            return self._density_fit_rank_key(map_model, thr, ca, ra, ta, cb, rb, tb, w_out)

        k0 = fit_key(RA, tA, RB, tB)
        if k0 is None:
            return RA, tA, RB, tB
        best_key = k0
        best_RA, best_tA, best_RB, best_tB = RA.copy(), tA.copy(), RB.copy(), tB.copy()

        def consider(ra: np.ndarray, ta: np.ndarray, rb: np.ndarray, tb: np.ndarray) -> None:
            nonlocal best_key, best_RA, best_tA, best_RB, best_tB
            k = fit_key(ra, ta, rb, tb)
            if k is not None and k > best_key:
                best_key = k
                best_RA, best_tA, best_RB, best_tB = ra.copy(), ta.copy(), rb.copy(), tb.copy()

    # (axis vector, ±max deg, step deg) — e3 largest; e2/e1 small backbone tilt.
        axis_rounds: List[Tuple[np.ndarray, float, float]] = [
            (ax3, float(ang_max_deg), float(ang_step_deg)),
            (ax2, 1.25, 0.25),
            (ax1, 1.25, 0.25),
        ]
        for _ in range(int(n_passes)):
            for ax_u, amax, astep in axis_rounds:
                angles = self._angle_grid_deg(-float(amax), float(amax), float(astep))
                ra0, ta0 = best_RA.copy(), best_tA.copy()
                c1p_a = ra0 @ c1_a0 + ta0
                for ang in angles:
                    Rk = self._rot_matrix(ax_u, float(ang))
                    ra2 = Rk @ ra0
                    ta2 = Rk @ (ta0 - c1p_a) + c1p_a
                    consider(ra2, ta2, best_RB, best_tB)
                rb0, tb0 = best_RB.copy(), best_tB.copy()
                c1p_b = rb0 @ c1_b0 + tb0
                for ang in angles:
                    Rk = self._rot_matrix(ax_u, float(ang))
                    rb2 = Rk @ rb0
                    tb2 = Rk @ (tb0 - c1p_b) + c1p_b
                    consider(best_RA, best_tA, rb2, tb2)

        fs = self._fitmap_style_scores(map_model, thr, ca, best_RA, best_tA, cb, best_RB, best_tB)
        dens_s, ins, outs = self._density_threshold_score(map_model, thr, ca, best_RA, best_tA, cb, best_RB, best_tB, w_out)
        g = self._geometry_penalties_build(best_RA, best_tA, best_RB, best_tB, n_a0, n_b0, c1_a0, c1_b0, p_a0, p_b0, plane_mode)
        self.session.logger.info(
            f"[BaseHunter] Micro C1′ refine (e3 ±{ang_max_deg}°/{ang_step_deg}°, e2/e1 ±1.25°/0.25°, no shift): "
            f"fitmap avg {fs['avg']:.5f}, thr-score {dens_s:.1f} (≥thr {ins}, <thr {outs}); "
            f"C1′-pen {g['c1_pen']:.3f} Å, PP-pen {g['pp_pen']:.3f} Å, plane-pen {g['plane_pen']:.4f}"
        )
        return best_RA, best_tA, best_RB, best_tB

    @staticmethod
    def _parse_pdb_coords(path: Path) -> np.ndarray:
        coords: List[List[float]] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            an = line[12:16].strip().upper()
        # Skip backbone/sugar only (primed names, P/OP*). Do **not** exclude base ring C5/C4/O2 etc.
            if BaseHunterInteractiveTool._is_sugar_phosphate_atom_name(an):
                continue
            elem = line[76:78].strip().upper()
            if elem in {"H", "D"}:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except Exception:
                continue
            coords.append([x, y, z])
        if not coords:
            return np.zeros((0, 3), dtype=np.float64)
        arr = np.array(coords, dtype=np.float64)
        return arr - arr.mean(axis=0, keepdims=True)

    def _load_templates(self) -> Dict[str, List[np.ndarray]]:
        if self._template_cache:
            return self._template_cache
        root = Path(self.template_dir.text().strip()).expanduser()
        purine_files: List[Path] = []
        pyr_files: List[Path] = []
        txt = root / "templates.txt"
        if txt.is_file():
            for e in _parse_templates_txt(txt):
                fn = str(e["filename"])
                p = root / fn
                if p.suffix.lower() != ".pdb" or not p.exists():
                    continue
                lfn = fn.lower()
                if "base" not in lfn:
                    continue
                if "purine" in lfn or "-a" in lfn or "-g" in lfn or "a2" in lfn or "g2" in lfn:
                    purine_files.append(p)
                if "pyrimidine" in lfn or "-c" in lfn or "-t" in lfn or "c2" in lfn or "t2" in lfn:
                    pyr_files.append(p)
        # Fallback filename scan for common patterns.
        if not purine_files:
            purine_files = sorted(root.glob("*A*base*.pdb")) + sorted(root.glob("*G*base*.pdb"))
        if not pyr_files:
            pyr_files = sorted(root.glob("*C*base*.pdb")) + sorted(root.glob("*T*base*.pdb"))
        pur_t = [self._parse_pdb_coords(p) for p in purine_files]
        pyr_t = [self._parse_pdb_coords(p) for p in pyr_files]
        pur_t = [x for x in pur_t if x.shape[0] > 0]
        pyr_t = [x for x in pyr_t if x.shape[0] > 0]
        self._template_cache = {"purine": pur_t, "pyrimidine": pyr_t}
        return self._template_cache

    def _discover_pur_pyr_template_paths(self, root: Path) -> Tuple[List[Path], List[Path]]:
        """Same discovery rules as :meth:`_load_templates`, but return paths for full-atom parsing."""
        purine_files: List[Path] = []
        pyr_files: List[Path] = []
        txt = root / "templates.txt"
        if txt.is_file():
            for e in _parse_templates_txt(txt):
                fn = str(e["filename"])
                p = root / fn
                if p.suffix.lower() != ".pdb" or not p.exists():
                    continue
                lfn = fn.lower()
                if "base" not in lfn:
                    continue
                if "purine" in lfn or "-a" in lfn or "-g" in lfn or "a2" in lfn or "g2" in lfn:
                    purine_files.append(p)
                if "pyrimidine" in lfn or "-c" in lfn or "-t" in lfn or "c2" in lfn or "t2" in lfn:
                    pyr_files.append(p)
        if not purine_files:
            purine_files = sorted(root.glob("*A*base*.pdb")) + sorted(root.glob("*G*base*.pdb"))
        if not pyr_files:
            pyr_files = sorted(root.glob("*C*base*.pdb")) + sorted(root.glob("*T*base*.pdb"))
        return purine_files, pyr_files

    def _load_class_atom_template_lists(self) -> Tuple[List[List[TemplateAtom]], List[List[TemplateAtom]]]:
        """Class-shape PDBs for post-build R/Y refine (purine vs pyrimidine), parsed to :class:`TemplateAtom`.

        When ``referencePDB-purine.pdb`` and ``referencePDB-pyrimidine.pdb`` exist in the template directory,
        **only** those two files are used for phase-2 (no per-base ``*A*base*`` / ``templateBP-*`` discovery).
        Otherwise discovery follows :meth:`_discover_pur_pyr_template_paths` (``templates.txt`` + filename heuristics).
        """
        cached = getattr(self, "_class_atom_templates_lists_cache", None)
        if cached is not None:
            return cached
        root = Path(self.template_dir.text().strip()).expanduser()

        ref_pur = root / "referencePDB-purine.pdb"
        ref_pyr = root / "referencePDB-pyrimidine.pdb"
        if ref_pur.is_file() and ref_pyr.is_file():
            pur_atoms = self._template_atoms_non_h(ref_pur)
            pyr_atoms = self._template_atoms_non_h(ref_pyr)
            if pur_atoms and pyr_atoms:
                self._class_atom_templates_lists_cache = ([pur_atoms], [pyr_atoms])
                return self._class_atom_templates_lists_cache

        pur_paths, pyr_paths = self._discover_pur_pyr_template_paths(root)
        pur_lists = [t for p in pur_paths if (t := self._template_atoms_non_h(p))]
        pyr_lists = [t for p in pyr_paths if (t := self._template_atoms_non_h(p))]
        self._class_atom_templates_lists_cache = (pur_lists, pyr_lists)
        return self._class_atom_templates_lists_cache

    def _template_atoms_non_h(self, path: Path) -> List[TemplateAtom]:
        """All ATOM/HETATM from ``path`` as :class:`TemplateAtom`, excluding hydrogen/deuterium."""
        return [
            a
            for a in self._parse_pdb_all_atoms(path)
            if (a.element or "C").strip().upper() not in {"H", "D"}
        ]

    def _built_atom_coords_by_norm(self, residue) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for atom in residue.atoms:
            try:
                k = self._norm_atom_key(atom.name)
            except Exception:
                continue
            # ChimeraX exposes ``atom.element`` as an Element object, not a plain string.
            if self._chimerax_atom_element_letter(atom) in {"H", "D"}:
                continue
            c = getattr(atom, "coord", None)
            if c is None:
                continue
            out[k] = np.asarray(c, dtype=np.float64).reshape(3)
        return out

    def _kabsch_align_template_to_built_sp(
        self,
        tpl: List[TemplateAtom],
        built_map: Dict[str, np.ndarray],
        *,
        align_scope: str = "all",
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Rigid body aligning ``tpl`` to built residue using shared atom names (≥3 matches).

        ``align_scope``:
        - ``"sp"`` — sugar–phosphate + sugar ring only (:attr:`_SP_KABSCH_KEYS`); no NSP base atoms.
        - ``"all"`` — SP + NSP (legacy / broader overlap when SP-only has too few points).
        """
        if align_scope == "sp":
            keys = self._SP_KABSCH_KEYS
        else:
            keys = self._SP_KABSCH_KEYS | self._NSP_ALL_CLASS
        p_rows: List[np.ndarray] = []
        q_rows: List[np.ndarray] = []
        for a in tpl:
            if (a.element or "C").upper() in {"H", "D"}:
                continue
            k = self._norm_atom_key(a.name)
            if k not in keys:
                continue
            if k not in built_map:
                continue
            p_rows.append(np.asarray(a.coord, dtype=np.float64).reshape(3))
            q_rows.append(built_map[k].copy())
        if len(p_rows) < 3:
            return None
        return self._kabsch_rigid_rows(np.stack(p_rows, axis=0), np.stack(q_rows, axis=0))

    def _scalar_map_score_nsp(
        self, map_model, thr: float, tpl: List[TemplateAtom], R: np.ndarray, t: np.ndarray
    ) -> float:
        """Single scalar from NSP-only map agreement (inclusion + threshold mass + mean density)."""
        rows: List[np.ndarray] = []
        for a in tpl:
            if (a.element or "C").upper() in {"H", "D"}:
                continue
            k = self._norm_atom_key(a.name)
            if k not in self._NSP_ALL_CLASS:
                continue
            rows.append(R @ np.asarray(a.coord, dtype=np.float64).reshape(3) + t)
        if len(rows) < 2:
            return -1e9
        pxyz = np.stack(rows, axis=0)
        ins, dens_s, avg, neg_below = self._rank_world_points_density(map_model, thr, pxyz)
        return float(ins * 12.0 + dens_s + 0.35 * avg + neg_below)

    def _emd1d_sorted_arrays(self, u: np.ndarray, v: np.ndarray) -> float:
        """Earth mover's distance for equal-weight 1D samples (sorted marginal samples)."""
        if u.size == 0 or v.size == 0:
            return 1e3
        su = np.sort(np.ravel(u))
        sv = np.sort(np.ravel(v))
        n = min(su.size, sv.size)
        if n < 1:
            return 1e3
        su = su[:n]
        sv = sv[:n]
        return float(np.mean(np.abs(su - sv)))

    def _best_refine_class_score(
        self,
        map_model,
        thr: float,
        class_templates: List[List[TemplateAtom]],
        built_res,
        max_templates: int,
    ) -> float:
        built_map = self._built_atom_coords_by_norm(built_res)
        best = -1e9
        for i, tpl in enumerate(class_templates):
            if i >= max_templates:
                break
            rt = self._kabsch_align_template_to_built_sp(tpl, built_map, align_scope="sp")
            if rt is None:
                continue
            R, t = rt
            best = max(best, self._scalar_map_score_nsp(map_model, thr, tpl, R, t))
        return best

    def _template_world_points_all_class_atoms(
        self, tpl: List[TemplateAtom], R: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """All non-hydrogen template atoms in world frame (every atom in ``tpl`` after ``R,t``)."""
        rows: List[np.ndarray] = []
        for a in tpl:
            if (a.element or "C").upper() in {"H", "D"}:
                continue
            rows.append(R @ np.asarray(a.coord, dtype=np.float64).reshape(3) + t)
        if not rows:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack(rows, axis=0)

    def _all_atom_map_fit_stats(
        self, map_model, thr: float, world_pts: np.ndarray, *, is_purine_class: bool
    ) -> Tuple[float, int, int, float]:
        """Mean sampled density, out-of-contour count, total atoms, and a scalar score (higher = better pur vs pyr)."""
        if map_model is None or world_pts.size == 0:
            return float("nan"), 0, 0, -1e9
        dens = self._sample_points_density(map_model, world_pts)
        if dens.size == 0:
            return float("nan"), 0, 0, -1e9
        thr_f = float(thr)
        mean_d = float(np.mean(dens))
        n_tot = int(dens.size)
        n_out = int(np.sum(dens < thr_f))
        frac_out = n_out / max(n_tot, 1)
        out_w = 6.5 if is_purine_class else 2.4
        score = mean_d - out_w * frac_out
        return mean_d, n_out, n_tot, score

    def _best_refine_atom_contour_class_score(
        self,
        map_model,
        thr: float,
        class_templates: List[List[TemplateAtom]],
        built_res,
        max_templates: int,
        *,
        is_purine_class: bool,
    ) -> float:
        """Best template vs built: SP-only Kabsch, score from all non-H atoms vs map threshold."""
        s, _, _, _ = self._best_refine_atom_contour_class_fit(map_model, thr, class_templates, built_res, max_templates, is_purine_class=is_purine_class)
        return s

    def _best_refine_atom_contour_class_fit(
        self,
        map_model,
        thr: float,
        class_templates: List[List[TemplateAtom]],
        built_res,
        max_templates: int,
        *,
        is_purine_class: bool,
    ) -> Tuple[float, float, int, int]:
        """Like :meth:`_best_refine_atom_contour_class_score` but returns ``(score, mean_d, n_out, n_tot)`` for the best template."""
        built_map = self._built_atom_coords_by_norm(built_res)
        best_s = -1e9
        best_mean, best_out, best_tot = float("nan"), 0, 0
        for i, tpl in enumerate(class_templates):
            if i >= max_templates:
                break
            rt = self._kabsch_align_template_to_built_sp(tpl, built_map, align_scope="sp")
            if rt is None:
                continue
            R, t = rt
            pts = self._template_world_points_all_class_atoms(tpl, R, t)
            if pts.shape[0] < 1:
                continue
            mean_d, n_out, n_tot, score = self._all_atom_map_fit_stats(map_model, thr, pts, is_purine_class=is_purine_class)
            if score > best_s:
                best_s, best_mean, best_out, best_tot = score, mean_d, n_out, n_tot
        return best_s, best_mean, best_out, best_tot

    def _refine_correlation_nsp(
        self, map_model, tpl: List[TemplateAtom], R: np.ndarray, t: np.ndarray
    ) -> float:
        """Pearson r between map samples at aligned NSP sites and a crude radial weight (higher toward base)."""
        rows: List[np.ndarray] = []
        wts: List[float] = []
        com = np.zeros(3, dtype=np.float64)
        ncom = 0
        for a in tpl:
            if (a.element or "C").upper() in {"H", "D"}:
                continue
            k = self._norm_atom_key(a.name)
            if k not in self._NSP_ALL_CLASS:
                continue
            p = R @ np.asarray(a.coord, dtype=np.float64).reshape(3) + t
            rows.append(p)
            com += p
            ncom += 1
        if len(rows) < 3:
            return 0.0
        com /= max(1, ncom)
        for p in rows:
            wts.append(float(np.linalg.norm(p - com)))
        pxyz = np.stack(rows, axis=0)
        dens = self._sample_points_density(map_model, pxyz)
        w = np.asarray(wts, dtype=np.float64)
        w = w - float(np.mean(w))
        d = dens - float(np.mean(dens))
        sd = float(np.std(d)) + 1e-9
        sw = float(np.std(w)) + 1e-9
        return float(np.dot(d / sd, w / sw) / max(len(rows), 1))

    def _refine_pur_pyr_probabilities_from_pair_model(
        self, p: Pair, map_model, thr: float, model, thorough: bool
    ) -> Optional[Dict[str, float]]:
        """Re-score R/Y from the built dimer: SP-only Kabsch alignment, NSP scalar + all-atom contour vs threshold."""
        r0 = p.result
        if r0 is None:
            return None
        pur_lists, pyr_lists = self._load_class_atom_template_lists()
        if not pur_lists or not pyr_lists:
            return None
        root = Path(self.template_dir.text().strip()).expanduser()
        ref_pur_path = root / "referencePDB-purine.pdb"
        ref_pyr_path = root / "referencePDB-pyrimidine.pdb"
        pur_log = "referencePDB-purine" if ref_pur_path.is_file() else "purine-class-template"
        pyr_log = "referencePDB-pyrimidine" if ref_pyr_path.is_file() else "pyrimidine-class-template"
        ra = self._find_residue_by_chain_number(model, "A", 1)
        rb = self._find_residue_by_chain_number(model, "B", 1)
        if ra is None or rb is None:
            return None
        max_tpl = max(1, len(pur_lists)) if thorough else min(2, len(pur_lists))
        max_tpy = max(1, len(pyr_lists)) if thorough else min(2, len(pyr_lists))

        def nsp_points_aligned(tpl: List[TemplateAtom], R: np.ndarray, t: np.ndarray) -> Optional[np.ndarray]:
            rows: List[np.ndarray] = []
            for a in tpl:
                if (a.element or "C").upper() in {"H", "D"}:
                    continue
                k = self._norm_atom_key(a.name)
                if k not in self._NSP_ALL_CLASS:
                    continue
                rows.append(R @ np.asarray(a.coord, dtype=np.float64).reshape(3) + t)
            if len(rows) < 3:
                return None
            return np.stack(rows, axis=0)

        def corr_pur_minus_pyr(res) -> float:
            bm = self._built_atom_coords_by_norm(res)
            cpp = cyy = None
            for tpl in pur_lists[:max_tpl]:
                rt = self._kabsch_align_template_to_built_sp(tpl, bm, align_scope="sp")
                if rt is None:
                    continue
                cpp = self._refine_correlation_nsp(map_model, tpl, rt[0], rt[1])
                break
            for tpl in pyr_lists[:max_tpy]:
                rt = self._kabsch_align_template_to_built_sp(tpl, bm, align_scope="sp")
                if rt is None:
                    continue
                cyy = self._refine_correlation_nsp(map_model, tpl, rt[0], rt[1])
                break
            if cpp is None or cyy is None:
                return 0.0
            return float(cpp - cyy)

        def mean_dens_pur_minus_pyr(res) -> float:
            bm = self._built_atom_coords_by_norm(res)
            mp = my = None
            for tpl in pur_lists[:max_tpl]:
                rt = self._kabsch_align_template_to_built_sp(tpl, bm, align_scope="sp")
                if rt is None:
                    continue
                pxyz = nsp_points_aligned(tpl, rt[0], rt[1])
                if pxyz is None:
                    continue
                mp = float(np.mean(self._sample_points_density(map_model, pxyz)))
                break
            for tpl in pyr_lists[:max_tpy]:
                rt = self._kabsch_align_template_to_built_sp(tpl, bm, align_scope="sp")
                if rt is None:
                    continue
                pxyz = nsp_points_aligned(tpl, rt[0], rt[1])
                if pxyz is None:
                    continue
                my = float(np.mean(self._sample_points_density(map_model, pxyz)))
                break
            if mp is None or my is None:
                return 0.0
            return float(mp - my)

        def emd1d_pur_vs_pyr_on_res(res) -> float:
            bm = self._built_atom_coords_by_norm(res)
            pxyz_pur = pxyz_pyr = None
            for tpl in pur_lists[:max_tpl]:
                rt = self._kabsch_align_template_to_built_sp(tpl, bm, align_scope="sp")
                if rt is None:
                    continue
                pxyz_pur = nsp_points_aligned(tpl, rt[0], rt[1])
                if pxyz_pur is not None:
                    break
            for tpl in pyr_lists[:max_tpy]:
                rt = self._kabsch_align_template_to_built_sp(tpl, bm, align_scope="sp")
                if rt is None:
                    continue
                pxyz_pyr = nsp_points_aligned(tpl, rt[0], rt[1])
                if pxyz_pyr is not None:
                    break
            if pxyz_pur is None or pxyz_pyr is None:
                return 0.0
            d_pur = self._sample_points_density(map_model, pxyz_pur)
            d_pyr = self._sample_points_density(map_model, pxyz_pyr)
            return float(-self._emd1d_sorted_arrays(d_pur, d_pyr))

        sA_pur = self._best_refine_class_score(map_model, thr, pur_lists, ra, max_tpl)
        sA_pyr = self._best_refine_class_score(map_model, thr, pyr_lists, ra, max_tpy)
        sB_pur = self._best_refine_class_score(map_model, thr, pur_lists, rb, max_tpl)
        sB_pyr = self._best_refine_class_score(map_model, thr, pyr_lists, rb, max_tpy)
        atomA_pur, mAp, oAp, tAp = self._best_refine_atom_contour_class_fit(
            map_model, thr, pur_lists, ra, max_tpl, is_purine_class=True
        )
        atomA_pyr, mAy, oAy, tAy = self._best_refine_atom_contour_class_fit(
            map_model, thr, pyr_lists, ra, max_tpy, is_purine_class=False
        )
        atomB_pur, mBp, oBp, tBp = self._best_refine_atom_contour_class_fit(
            map_model, thr, pur_lists, rb, max_tpl, is_purine_class=True
        )
        atomB_pyr, mBy, oBy, tBy = self._best_refine_atom_contour_class_fit(
            map_model, thr, pyr_lists, rb, max_tpy, is_purine_class=False
        )
        for ch, refn, mean_d, n_out, n_tot in (
            ("A", pur_log, mAp, oAp, tAp),
            ("A", pyr_log, mAy, oAy, tAy),
            ("B", pur_log, mBp, oBp, tBp),
            ("B", pyr_log, mBy, oBy, tBy),
        ):
            if math.isfinite(mean_d) and n_tot > 0:
                self.session.logger.info(
                    f"[BaseHunter] Position {p.pair_id} Marker chain {ch}  fit with {refn} "
                    f"{mean_d:.4f} {n_out} of {n_tot} atoms outside contour"
                )
        da_atom = float(atomA_pur - atomA_pyr)
        db_atom = float(atomB_pur - atomB_pyr)
        # NSP scalar + all-atom contour + EMD only; NSP correlation and NSP mean-density delta disabled.
        w_c = 0.0
        w_m = 0.0
        w_e = 0.06 if thorough else 0.03
        w_atom = 0.52 if thorough else 0.40
        dA = (
            (sA_pur - sA_pyr)
            + w_atom * da_atom
            + w_c * corr_pur_minus_pyr(ra)
            + w_m * mean_dens_pur_minus_pyr(ra)
            + w_e * emd1d_pur_vs_pyr_on_res(ra)
        )
        dB = (
            (sB_pur - sB_pyr)
            + w_atom * db_atom
            + w_c * corr_pur_minus_pyr(rb)
            + w_m * mean_dens_pur_minus_pyr(rb)
            + w_e * emd1d_pur_vs_pyr_on_res(rb)
        )
        t_ref = 0.52
        pA = float(1.0 / (1.0 + math.exp(-dA / t_ref)))
        pA_pyr = 1.0 - pA
        pB_pur = float(1.0 / (1.0 + math.exp(-dB / t_ref)))
        pB_pyr = 1.0 - pB_pur
        clash = float(r0.get("clash_metric", 0.0))
        joint = max(0.0, min(0.99, 0.5 * (pA + pB_pyr) - 0.20 * clash))
        den_a = self._sample_local_density(map_model, p.marker_a, threshold=thr)
        den_b = self._sample_local_density(map_model, p.marker_b, threshold=thr)
        planarity = min(2.0, 0.12 + 0.22 * abs(den_a - den_b))
        conf = max(0.0, min(0.99, 0.5 * (abs(pA - pA_pyr) + abs(pB_pyr - pB_pur))))
        call = "A-T" if (pA + pB_pyr) >= (pA_pyr + pB_pur) else "G-C"
        return {
            "pA_purine": float(pA),
            "pA_pyrimidine": float(pA_pyr),
            "pB_purine": float(pB_pur),
            "pB_pyrimidine": float(pB_pyr),
            "joint_wc": float(joint),
            "planarity_rms": float(planarity),
            "clash_metric": float(clash),
            "confidence": float(conf),
            "call": call,
            "pA_purine_phase1": float(r0.get("pA_purine", 0.5)),
            "pA_pyrimidine_phase1": float(r0.get("pA_pyrimidine", 0.5)),
            "pB_purine_phase1": float(r0.get("pB_purine", 0.5)),
            "pB_pyrimidine_phase1": float(r0.get("pB_pyrimidine", 0.5)),
            "refine_scores_A_pur_pyr": (float(sA_pur), float(sA_pyr)),
            "refine_scores_B_pur_pyr": (float(sB_pur), float(sB_pyr)),
            "refine_atom_scores_A_pur_pyr": (float(atomA_pur), float(atomA_pyr)),
            "refine_atom_scores_B_pur_pyr": (float(atomB_pur), float(atomB_pyr)),
            "refine_atom_discriminant_AB": (float(da_atom), float(db_atom)),
            "refine_discriminant_AB": (float(dA), float(dB)),
            "refine_mode": "thorough" if thorough else "balanced",
        }

    @staticmethod
    def _parse_base_letter(path: Path) -> Optional[str]:
        stem = path.stem.upper()
        for b in ("A", "G", "C", "T"):
            if f"-{b}" in stem or f"_{b}" in stem or f"{b}2" in stem or stem.endswith(b):
                return b
        if "ADEN" in stem:
            return "A"
        if "GUAN" in stem:
            return "G"
        if "CYTO" in stem:
            return "C"
        if "THYM" in stem:
            return "T"
        return None

    @staticmethod
    def _parse_template_atoms(path: Path) -> List[Tuple[str, str, np.ndarray]]:
        out: List[Tuple[str, str, np.ndarray]] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip() or "X"
            elem = line[76:78].strip()
            if not elem:
                elem = atom_name[0]
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except Exception:
                continue
            an = atom_name.upper()
            if BaseHunterInteractiveTool._is_sugar_phosphate_atom_name(atom_name):
                continue
            if (elem or "").strip().upper() in {"H", "D"}:
                continue
            out.append((atom_name, elem.upper(), np.array([x, y, z], dtype=np.float64)))
        if not out:
            return out
        ctr = np.mean(np.array([x[2] for x in out], dtype=np.float64), axis=0)
        return [(n, e, c - ctr) for n, e, c in out]

    def _load_base_templates(self) -> Dict[str, List[TemplateAtom]]:
        if self._base_template_cache:
            return self._base_template_cache
        root = Path(self.template_dir.text().strip()).expanduser()
        candidates: List[Path] = []
        txt = root / "templates.txt"
        if txt.is_file():
            for e in _parse_templates_txt(txt):
                fn = str(e["filename"])
                p = root / fn
                if p.suffix.lower() != ".pdb" or not p.exists():
                    continue
                if "base" not in fn.lower():
                    continue
                candidates.append(p)
        if not candidates:
            candidates = sorted(root.glob("*base*.pdb"))
        out: Dict[str, List[TemplateAtom]] = {"A": [], "G": [], "C": [], "T": []}
        for p in candidates:
            base = self._parse_base_letter(p)
            if base is None or base not in out:
                continue
            parsed = self._parse_template_atoms(p)
            if not parsed:
                continue
            out[base] = [TemplateAtom(name=n, element=e, coord=c) for n, e, c in parsed]
        self._base_template_cache = out
        return out

    def _sample_points_density(self, map_model, points_xyz: np.ndarray) -> np.ndarray:
        vals = None
        try:
            vals = map_model.interpolated_values(points_xyz.tolist())
        except Exception:
            vals = None
        if vals is not None:
            try:
                arr = np.array(vals, dtype=np.float64)
                if arr.size == points_xyz.shape[0]:
                    return arr
            except Exception:
                pass
        # Fallback per-point nearest/linear fallback path.
        out = np.zeros(points_xyz.shape[0], dtype=np.float64)
        for i, p in enumerate(points_xyz):
            out[i] = self._sample_density(map_model, [float(p[0]), float(p[1]), float(p[2])])
        return out

    def _score_template_class(
        self,
        map_model,
        marker_xyz: List[float],
        axis_hint: np.ndarray,
        templates: List[np.ndarray],
        threshold: float,
    ) -> float:
        if not templates:
            return -1e6
        m = np.array(marker_xyz, dtype=np.float64)
        n = self._normalize(axis_hint)
        # Build an orthogonal axis for tilt.
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(n, up))) > 0.92:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = self._normalize(np.cross(n, up))
        best = -1e9
        for tpl in templates:
            # Reduced DOF search: rotation about inter-marker axis + small tilt.
            for ang in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0):
                r1 = self._rot_matrix(n, ang)
                for tilt in (-20.0, 0.0, 20.0):
                    r = self._rot_matrix(u, tilt) @ r1
                    pts = (tpl @ r.T) + m[None, :]
                    dens = self._sample_points_density(map_model, pts)
                    above = np.maximum(dens - threshold, 0.0)
                    below = np.maximum(threshold - dens, 0.0)
                    # Per-point mean: purine templates have more heavy atoms than pyrimidine; summing biased
                    # both marginals toward purine (raw A/B both "purine") and broke WC display/build.
                    npts = int(pts.shape[0])
                    if npts < 1:
                        continue
                    score = float(np.mean(above) - 0.25 * np.mean(below))
                    if score > best:
                        best = score
        return best

    def _clash_metric(self, xyz_a: List[float], xyz_b: List[float]) -> float:
        model = self._current_structure_model()
        if model is None:
            return 0.0
        min_d = None
        try:
            atoms = getattr(model, "atoms", None)
            if atoms is None:
                return 0.0
            for atom in atoms:
                pos = getattr(atom, "scene_coord", None) or getattr(atom, "coord", None)
                if pos is None:
                    continue
                p = [float(pos[0]), float(pos[1]), float(pos[2])]
                d = min(self._dist(xyz_a, p), self._dist(xyz_b, p))
                min_d = d if min_d is None else min(min_d, d)
        except Exception:
            return 0.0
        if min_d is None:
            return 0.0
        # Positive penalty if closer than ~2.5A.
        return max(0.0, (2.5 - float(min_d)) / 2.5)

    def _template_quality_boost(self) -> float:
        root = Path(self.template_dir.text().strip()).expanduser()
        txt = root / "templates.txt"
        if not txt.is_file():
            return 0.0
        entries = _parse_templates_txt(txt)
        names = {str(e["filename"]).lower() for e in entries}
        has_pur = any("purine" in n for n in names)
        has_pyr = any("pyrimidine" in n for n in names)
        return 0.08 if (has_pur and has_pyr) else 0.0

    def _compute_result(self, p: Pair) -> Dict[str, float]:
        """Template-based map scoring for purine/pyrimidine at both markers."""
        map_model = self._current_map_model()
        if map_model is None or p.marker_a is None or p.marker_b is None:
            raise RuntimeError("Missing map or markers.")
        templates = self._load_templates()
        if not templates.get("purine") or not templates.get("pyrimidine"):
            raise RuntimeError("Missing purine/pyrimidine base templates (.pdb) in template directory.")

        thr = self._threshold_value()
        a = np.array(p.marker_a, dtype=np.float64)
        b = np.array(p.marker_b, dtype=np.float64)
        # Same line direction a→b for both markers so rotation grids are not mirrored about −n.
        ab = self._normalize(b - a)

        sA_pur = self._score_template_class(map_model, p.marker_a, ab, templates["purine"], thr)
        sA_pyr = self._score_template_class(map_model, p.marker_a, ab, templates["pyrimidine"], thr)
        sB_pur = self._score_template_class(map_model, p.marker_b, ab, templates["purine"], thr)
        sB_pyr = self._score_template_class(map_model, p.marker_b, ab, templates["pyrimidine"], thr)

        # Softmax over two classes for each side (temperature with mean-based scores; lower → sharper).
        t = 0.32
        ea_pur = math.exp(sA_pur / t)
        ea_pyr = math.exp(sA_pyr / t)
        eb_pur = math.exp(sB_pur / t)
        eb_pyr = math.exp(sB_pyr / t)
        pA = ea_pur / max(1e-9, ea_pur + ea_pyr)
        pA_pyr = 1.0 - pA
        pB_pur = eb_pur / max(1e-9, eb_pur + eb_pyr)
        pB_pyr = 1.0 - pB_pur

        clash = self._clash_metric(p.marker_a, p.marker_b)
        # Pair-level WC consistency prefers purine on one side and pyrimidine on the other.
        joint = max(0.0, min(0.99, 0.5 * (pA + pB_pyr) - 0.20 * clash))
        # Approximate planarity proxy from marker geometry + local density asymmetry.
        den_a = self._sample_local_density(map_model, p.marker_a, threshold=thr)
        den_b = self._sample_local_density(map_model, p.marker_b, threshold=thr)
        planarity = min(2.0, 0.12 + 0.22 * abs(den_a - den_b))
        conf = max(0.0, min(0.99, 0.5 * (abs(pA - pA_pyr) + abs(pB_pyr - pB_pur))))
        call = "A-T" if (pA + pB_pyr) >= (pA_pyr + pB_pur) else "G-C"
        out = {
            "pA_purine": float(pA),
            "pA_pyrimidine": float(pA_pyr),
            "pB_purine": float(pB_pur),
            "pB_pyrimidine": float(pB_pyr),
            "joint_wc": float(joint),
            "planarity_rms": float(planarity),
            "clash_metric": float(clash),
            "confidence": float(conf),
            "call": call,
        }
        raw_a = "purine" if pA >= 0.5 else "pyrimidine"
        raw_b = "purine" if pB_pur >= 0.5 else "pyrimidine"
        d_show = self._display_decision(p, result_override=out)
        da = str(d_show["class_a"]) if d_show else "?"
        db = str(d_show["class_b"]) if d_show else "?"
        self.session.logger.info(
            f"[BaseHunter] Compute {p.pair_id}: markers {self._marker_atom_label(p.marker_atom_a)} | "
            f"{self._marker_atom_label(p.marker_atom_b)} | "
            f"scores A pur/pyr={sA_pur:.3f}/{sA_pyr:.3f} B pur/pyr={sB_pur:.3f}/{sB_pyr:.3f} | "
            f"Psoft A pur/pyr={pA:.3f}/{pA_pyr:.3f} B pur/pyr={pB_pur:.3f}/{pB_pyr:.3f} | "
            f"raw_class A={raw_a} B={raw_b} | display A={da} B={db} | "
            f"call={call} joint_wc={joint:.3f} clash={clash:.3f} conf={conf:.3f}"
        )
        return out

    def _quality_uses_post_build_refine(self) -> bool:
        return self.quality.currentText().strip().lower() in ("balanced", "thorough")

    def _quality_is_thorough(self) -> bool:
        return self.quality.currentText().strip().lower() == "thorough"

    def _post_compute_build_and_refine(self, p: Pair) -> bool:
        """Phase-2 R/Y from built dimer; closes temporary build model. Returns False if refine skipped."""
        if not self._quality_uses_post_build_refine():
            return True
        map_model = self._current_map_model()
        thr = self._threshold_value()
        thorough = self._quality_is_thorough()
        timer_was = self._model_refresh_timer.isActive() if hasattr(self, "_model_refresh_timer") else False
        if timer_was:
            self._model_refresh_timer.stop()
        self._in_build = True
        try:
            self.progress.setValue(55)
            m = self._build_pairs([p], title=f"BaseHunter refine {p.pair_id}")
            if m is None:
                self.session.logger.warning(f"[BaseHunter] {p.pair_id}: refine skipped (build failed).")
                return False
            refined = self._refine_pur_pyr_probabilities_from_pair_model(p, map_model, thr, m, thorough)
            if refined is None:
                self.session.logger.warning(f"[BaseHunter] {p.pair_id}: refine skipped (no template alignment).")
            else:
                p.result = refined
                self._apply_marker_colors(p)
                self.session.logger.info(
                    f"[BaseHunter] {p.pair_id}: post-build refine ({refined.get('refine_mode')}) "
                    f"P(A) {refined.get('pA_purine_phase1', 0):.3f}→{refined.get('pA_purine', 0):.3f}, "
                    f"P(B) {refined.get('pB_purine_phase1', 0):.3f}→{refined.get('pB_purine', 0):.3f}, call={refined.get('call')}"
                )
            try:
                from chimerax.core.commands import run

                run(self.session, f"close #{m.id_string}")
            except Exception:
                pass
            return refined is not None
        finally:
            self._in_build = False
            if timer_was:
                self._model_refresh_timer.start()
            self._refresh_models(force=True)

    def _compute_selected(self):
        p = self._find_pair(self._selected_pair_id)
        if p is None:
            self.compute_msg.setText("No pair selected.")
            return
        if p.marker_a is None or p.marker_b is None:
            self.compute_msg.setText(f"{p.pair_id} is not ready.")
            return
        self._strand_align_pair_vs_table_previous(p)
        self._sync_table()
        thr_now, thr_src = self._threshold_value_with_source()
        self.status.setText(f"Compute using threshold {thr_now:.6g} ({thr_src}).")
        self._maybe_log_scene_and_map_frame(self._current_map_model())
        self._log_intra_pair_marker_geometry(p)
        self.progress.setValue(25)
        try:
            p.result = self._compute_result(p)
            p.status = "computed"
            self._apply_marker_colors(p)
        except Exception as e:
            p.status = "error"
            self.compute_msg.setText(f"Compute failed for {p.pair_id}: {e}")
            self._sync_table()
            return
        if self._quality_uses_post_build_refine():
            self._post_compute_build_and_refine(p)
        self.progress.setValue(100)
        msg = f"Computed {p.pair_id}"
        if self._quality_uses_post_build_refine():
            msg += " (phase-1 + build + NSP refine)."
        else:
            msg += " (phase-1)."
        self.compute_msg.setText(msg)
        self._sync_table()

    def _compute_all(self):
        ready = [p for p in self._pairs if p.marker_a is not None and p.marker_b is not None]
        if not ready:
            self.compute_msg.setText("No ready pairs.")
            return
        if self._canonicalize_marker_strands_sequential(ready):
            self._sync_table()
        thr_now, thr_src = self._threshold_value_with_source()
        self.status.setText(f"Compute-all using threshold {thr_now:.6g} ({thr_src}).")
        self._maybe_log_scene_and_map_frame(self._current_map_model())
        total = len(ready)
        for i, p in enumerate(ready, start=1):
            self.progress.setValue(int((i - 1) * 100 / max(total, 1)))
            try:
                self._log_intra_pair_marker_geometry(p)
                p.result = self._compute_result(p)
                p.status = "computed"
                self._apply_marker_colors(p)
                if self._quality_uses_post_build_refine():
                    self._post_compute_build_and_refine(p)
                    self._sync_table()
            except Exception:
                p.status = "error"
            self.compute_msg.setText(f"Computing {p.pair_id} ({i}/{total}) ...")
        self.progress.setValue(100)
        suf = " (phase-1 + refine)" if self._quality_uses_post_build_refine() else ""
        self.compute_msg.setText(f"Computed {total} pair(s){suf}.")
        self._sync_table()

    @staticmethod
    def _purine_preferred_at_marker_a(res: Dict[str, Any]) -> bool:
        """Whether phase scores designate marker **A** as the purine side (vs pyrimidine)."""
        return float(res.get("pA_purine", 0.5)) >= float(res.get("pA_pyrimidine", 0.5))

    def _finalize_nsp_refine_decision(
        self, refined: Dict[str, Any], phase1_snap: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phase-2 vs phase-1: tier disagreement from phase-2 strength only; optionally apply phase-2 call."""
        phase1_call = str(phase1_snap.get("call", refined.get("call", "A-T")))
        call_p2 = str(refined.get("call", phase1_call))
        pur_a1 = self._purine_preferred_at_marker_a(phase1_snap)
        pur_a2 = self._purine_preferred_at_marker_a(refined)
        agree = pur_a1 == pur_a2

        disc = refined.get("refine_discriminant_AB", (0.0, 0.0))
        try:
            da, db = float(disc[0]), float(disc[1])
        except (TypeError, ValueError, IndexError):
            da, db = 0.0, 0.0
        strength = abs(da) + abs(db)
        # Phase-2-only confidence from blended discriminants (NSP+correlation+**all-atom contour**).
        conf_p2 = float(max(0.0, min(0.99, self._sigmoid(strength / 22.0))))

        tier: str
        reason: str
        applied = False
        if agree:
            tier = "agreed"
            reason = "phase-2 pur@A matches phase-1 (marker roles unchanged)"
            out = {**refined}
            out["call"] = phase1_call
            applied = True
        else:
            ad = refined.get("refine_atom_discriminant_AB", (0.0, 0.0))
            try:
                ada, adb = float(ad[0]), float(ad[1])
            except (TypeError, ValueError, IndexError):
                ada, adb = 0.0, 0.0
            reason = (
                f"phase-2 pur@A={pur_a2} vs phase-1 {pur_a1} "
                f"(call {call_p2} vs {phase1_call}; dA/dB={da:.2f}/{db:.2f}; "
                f"atomΔ={ada:.2f}/{adb:.2f})"
            )
            if conf_p2 < 0.40:
                tier = "cautious"
            elif conf_p2 < 0.65:
                tier = "suspicious"
            else:
                tier = "suspect"
            if tier == "suspect":
                out = {**refined}
                applied = True
            else:
                out = {**refined}
                for k in (
                    "call",
                    "pA_purine",
                    "pA_pyrimidine",
                    "pB_purine",
                    "pB_pyrimidine",
                    "joint_wc",
                    "confidence",
                    "planarity_rms",
                    "clash_metric",
                ):
                    if k in phase1_snap:
                        out[k] = phase1_snap[k]
                applied = False

        out["call_phase1"] = phase1_call
        out["call_refined"] = call_p2
        out["refine_discriminant_AB"] = (float(da), float(db))
        out["refine_strength"] = float(strength)
        out["refine_confidence_phase2"] = conf_p2
        out["refine_agrees_phase1"] = agree
        out["refine_tier"] = tier
        out["refine_applied_phase2"] = applied
        out["refine_reason"] = reason
        out["refine_enabled"] = True
        # Back-compat keys for logs / older readers
        out["refine_status"] = tier
        out["refine_confidence"] = conf_p2
        out["refine_delta"] = float(da + db)
        out["refine_score_h0"] = float(da)
        out["refine_score_h1"] = float(db)
        out["refine_pur_at_marker_a_phase1"] = bool(pur_a1)
        out["refine_pur_at_marker_a_phase2"] = bool(pur_a2)
        return out

    def _run_nsp_refine_for_pairs(self, pairs: List[Pair]) -> int:
        """Build each pair with fixed table chains, then re-score NSP-only assignment."""
        if not pairs:
            return 0
        map_model = self._current_map_model()
        if map_model is None:
            self.status.setText("Refine assignments requires a selected map.")
            return 0
        thr, thr_src = self._threshold_value_with_source()
        thorough = self._quality_is_thorough()
        timer_was = self._model_refresh_timer.isActive() if hasattr(self, "_model_refresh_timer") else False
        if timer_was:
            self._model_refresh_timer.stop()
        self._in_build = True
        changed = 0
        total = len(pairs)
        try:
            for i, p in enumerate(pairs, start=1):
                self.compute_msg.setText(f"NSP refine {p.pair_id} ({i}/{total}) ...")
                self.progress.setValue(int((i - 1) * 100 / max(total, 1)))
                phase1_snap = copy.deepcopy(p.result or {})
                m = self._build_pairs([p], title=f"BaseHunter NSP refine {p.pair_id}")
                if m is None:
                    continue
                try:
                    refined = self._refine_pur_pyr_probabilities_from_pair_model(p, map_model, thr, m, thorough)
                    if refined is None:
                        continue
                    before = str(phase1_snap.get("call", "A-T"))
                    refined = self._finalize_nsp_refine_decision(refined, phase1_snap)
                    p.result = refined
                    tier = str(refined.get("refine_tier", "agreed"))
                    p.status = "computed" if tier == "agreed" else f"computed/{tier}"
                    self._apply_marker_colors(p)
                    after = str(refined.get("call", before))
                    changed += 1 if after != before else 0
                    self.session.logger.info(
                        f"[BaseHunter] NSP refine {p.pair_id}: call {before} → {after}; "
                        f"dA/dB={float(refined.get('refine_score_h0', 0.0)):.3f}/{float(refined.get('refine_score_h1', 0.0)):.3f} "
                        f"strength={float(refined.get('refine_strength', 0.0)):.3f} "
                        f"tier={tier} conf_p2={float(refined.get('refine_confidence_phase2', 0.0)):.2f} "
                        f"applied={bool(refined.get('refine_applied_phase2', False))}."
                    )
                finally:
                    try:
                        from chimerax.core.commands import run
                        run(self.session, f"close #{m.id_string}")
                    except Exception:
                        pass
        finally:
            self._in_build = False
            if timer_was:
                self._model_refresh_timer.start()
            self._refresh_models(force=True)
        self.progress.setValue(100)
        self.status.setText(f"NSP refine used threshold {thr:.6g} ({thr_src}); updated {changed}/{total} call(s).")
        self.compute_msg.setText(f"NSP refinement complete for {total} pair(s).")
        self._sync_table()
        return changed

    def _refine_assignments_selected(self):
        p = self._find_pair(self._selected_pair_id)
        if p is None:
            self.compute_msg.setText("No pair selected.")
            return
        if p.result is None or p.marker_a is None or p.marker_b is None:
            self.compute_msg.setText(f"{p.pair_id} must be computed before NSP refine.")
            return
        self._run_nsp_refine_for_pairs([p])

    def _refine_assignments_all(self):
        pairs = [p for p in self._pairs if p.result is not None and p.marker_a is not None and p.marker_b is not None]
        if not pairs:
            self.compute_msg.setText("No computed pairs available for NSP refine.")
            return
        self._run_nsp_refine_for_pairs(pairs)

    def _update_results(self):
        """Fill the results panel with one block per pair (computed pairs), plus selected-pair emphasis."""
        lines: List[str] = []
        sel = self._selected_pair_id
        for p in self._pairs:
            if p.result is None:
                lines.append(f"{p.pair_id}: not computed ({p.status}).")
                continue
            r = p.result
            d = self._display_decision(p)
            raw = self._raw_density_classes(p)
            ra, rb = (raw or ("?", "?"))
            call = self._assignment_for(p)
            la, lb = self._letters_at_markers(p, call) if call else ("?", "?")
            if d is None:
                lines.append(f"{p.pair_id}: no display decision.")
                continue
            class_a = "Pur" if str(d["class_a"]) == "purine" else "Pyr"
            class_b = "Pur" if str(d["class_b"]) == "purine" else "Pyr"
            mark = "  ← selected" if p.pair_id == sel else ""
            lines.append(
                f"{p.pair_id}{mark}: disp A={class_a} B={class_b} | raw A={ra} B={rb} | "
                f"P(A pur/pyr)={float(d['pA_pur']):.2f}/{float(d['pA_pyr']):.2f} "
                f"P(B pur/pyr)={float(d['pB_pur']):.2f}/{float(d['pB_pyr']):.2f} | "
                f"call={r.get('call')} build→A={la} B={lb} | WC={float(r['joint_wc']):.2f} "
                f"plan={float(r['planarity_rms']):.2f} clash={float(r['clash_metric']):.2f}"
            )
            if bool(r.get("refine_enabled", False)):
                raa = r.get("refine_atom_discriminant_AB", (0.0, 0.0))
                try:
                    aada, aadb = float(raa[0]), float(raa[1])
                except (TypeError, ValueError, IndexError):
                    aada, aadb = 0.0, 0.0
                lines.append(
                    f"    refine: pur@A phase1={bool(r.get('refine_pur_at_marker_a_phase1'))} "
                    f"phase2={bool(r.get('refine_pur_at_marker_a_phase2'))} | "
                    f"call ph1={r.get('call_phase1', '?')} ph2={r.get('call_refined', '?')} active={r.get('call', '?')} | "
                    f"dA/dB={float(r.get('refine_score_h0', 0.0)):.2f}/{float(r.get('refine_score_h1', 0.0)):.2f} "
                    f"atomΔ={aada:.2f}/{aadb:.2f} str={float(r.get('refine_strength', 0.0)):.2f} "
                    f"conf_p2={float(r.get('refine_confidence_phase2', 0.0)):.2f} tier={r.get('refine_tier', '?')} "
                    f"applied={bool(r.get('refine_applied_phase2', False))} | {r.get('refine_reason', '?')}"
                )
        if not lines:
            self.results_text.setPlainText("No pairs.")
        else:
            self.results_text.setPlainText("\n".join(lines))

    def _assignment_for(self, p: Pair) -> str:
        if p.result is not None:
            return str(p.result.get("call", "A-T"))
        return "A-T"

    def _swap_selected_chains(self):
        """Marker A ↔ B + score swap; keep the same WC call string (table strand IDs only)."""
        row = self.pair_table.currentRow()
        if 0 <= row < self.pair_table.rowCount() and self.pair_table.item(row, 0) is not None:
            self._selected_pair_id = self.pair_table.item(row, 0).text()
        p = self._find_pair(self._selected_pair_id)
        if p is None:
            self.status.setText("No pair selected.")
            return
        if p.marker_a is None or p.marker_b is None:
            self.status.setText("Selected pair is missing markers.")
            return
        saved_call = str(p.result.get("call", "A-T")) if p.result is not None else None
        self._swap_marker_positions_and_atoms(p)
        if p.result is not None:
            self._swap_marker_axis_scores_in_result(p.result, freeze_call=saved_call)
        self._apply_marker_colors(p)
        self._sync_table()
        self.status.setText(
            f"Swapped chains for {p.pair_id} (Marker A/B and scores; call stays {saved_call or 'n/a'})."
        )

    def _swap_selected_assignment(self):
        """Swap R/Y identity for the selected pair (markers and chain IDs fixed)."""
        row = self.pair_table.currentRow()
        if 0 <= row < self.pair_table.rowCount() and self.pair_table.item(row, 0) is not None:
            self._selected_pair_id = self.pair_table.item(row, 0).text()
        p = self._find_pair(self._selected_pair_id)
        if p is None:
            self.status.setText("No pair selected.")
            return
        if p.result is None:
            self.status.setText("Compute the pair first, then swap assignment.")
            return
        self._invert_marker_class_scores_in_result(p.result)
        self._apply_marker_colors(p)
        self._sync_table()
        self.status.setText(f"{p.pair_id}: swapped assignment R↔Y on this pair (markers unchanged).")

    def _build_selected(self):
        p = self._find_pair(self._selected_pair_id)
        if p is None:
            self.status.setText("No pair selected.")
            return
        call = self._assignment_for(p)
        if not call:
            return
        if p.marker_a is None or p.marker_b is None:
            self.status.setText("Selected pair has missing marker(s).")
            return
        timer_was_active = self._model_refresh_timer.isActive() if hasattr(self, "_model_refresh_timer") else False
        self._in_build = True
        if timer_was_active:
            self._model_refresh_timer.stop()
        try:
            self._build_pairs([p], title=f"BaseHunter {p.pair_id}")
        finally:
            self._in_build = False
            if timer_was_active:
                self._model_refresh_timer.start()
        self._refresh_models(force=True)
        la, lb = self._letters_at_markers(p, call)
        self.status.setText(f"Build complete for {p.pair_id}: markers A/B ← {la}, {lb} (assignment {call})")

    def _build_all(self):
        computed = [p for p in self._pairs if p.result is not None]
        if not computed:
            self.status.setText("No computed pairs available.")
            return
        buildable = [p for p in computed if p.marker_a is not None and p.marker_b is not None and self._assignment_for(p)]
        if not buildable:
            self.status.setText("No buildable computed pairs.")
            return
        # Same duplex order and marker strand labels as a multi-pair ``_build_pairs`` run; required so merge
        # C1′ stride and per-marker probabilities stay aligned with helix neighbors (table order can differ).
        ordered = self._order_pairs_along_duplex(buildable)
        # Do not re-run strand canonicalization here: ``Compute all`` already aligns markers along the duplex,
        # and repeating can fight manual ``Swap assignment`` / per-pair refine. Merge uses duplex order + dual C1′ hops.
        timer_was_active = self._model_refresh_timer.isActive() if hasattr(self, "_model_refresh_timer") else False
        self._in_build = True
        if timer_was_active:
            self._model_refresh_timer.stop()
        built: List[Tuple[Pair, Any]] = []
        try:
            for p in ordered:
                m = self._build_pairs([p], title=f"BaseHunter {p.pair_id}")
                if m is not None:
                    built.append((p, m))
        finally:
            self._in_build = False
            if timer_was_active:
                self._model_refresh_timer.start()
        if not built:
            self.status.setText("Build all failed: no pair models were produced.")
            return
        merged = self._merge_pair_models(built, title="BaseHunter Build")
        if merged is None:
            self.status.setText("Build all failed while merging pair models.")
            return
        self._refresh_models(force=True)
        self.status.setText(f"Build complete for {len(built)} pair(s), merged in duplex order (chains A/B).")

    @staticmethod
    def _chimerax_atom_element_letter(atom) -> str:
        el = getattr(atom, "element", None)
        if el is None:
            return "C"
        name = getattr(el, "name", None)
        if name:
            s = str(name).strip().upper()
            return s[:1] if s else "C"
        s = str(el).strip().upper()
        return s[:1] if s else "C"

    def _find_residue_by_chain_number(self, struct, chain_id: str, resnum: int):
        cid = str(chain_id).strip()
        for r in struct.residues:
            if str(getattr(r, "chain_id", "")).strip() != cid:
                continue
            try:
                if int(getattr(r, "number", -1)) == int(resnum):
                    return r
            except Exception:
                continue
        return None

    def _c1_prime_coord_on_residue(self, struct, chain_id: str, resnum: int) -> Optional[np.ndarray]:
        res = self._find_residue_by_chain_number(struct, chain_id, resnum)
        if res is None:
            return None
        for a in res.atoms:
            if self._norm_atom_key(a.name) == "C1'":
                return np.asarray(getattr(a, "coord", None), dtype=np.float64).reshape(3)
        return None

    @staticmethod
    def _duplex_chain_swap_from_quad_distances(
        pa: np.ndarray,
        pb: np.ndarray,
        c1: np.ndarray,
        c2: np.ndarray,
        *,
        distinct_margin: float = 0.25,
        min_max: float = 3.0,
    ) -> Tuple[bool, str, float, float, float, float]:
        """Use the **largest** of four cross-pair distances to label strands (robust to helix bend).

        ``pa`` / ``pb`` are opposite-strand anchors (previous pair marker A/B or merged C1′ sites).
        ``c1`` / ``c2`` are the two new sites in default order (e.g. first/second click → A/B).
        The winning distance pairs **opposite** strands; if it is (pa,c1) or (pb,c2), default order is wrong
        and the caller should **swap** so ``c1`` maps to the ``pb`` side and ``c2`` to the ``pa`` side.

        Returns ``(swap, key, d11, d12, d21, d22)`` with keys 11=pa–c1, 12=pa–c2, 21=pb–c1, 22=pb–c2.
        """
        d11 = float(np.linalg.norm(pa - c1))
        d12 = float(np.linalg.norm(pa - c2))
        d21 = float(np.linalg.norm(pb - c1))
        d22 = float(np.linalg.norm(pb - c2))
        vals = (d11, d12, d21, d22)
        keys = ("11", "12", "21", "22")
        imax = int(np.argmax(vals))
        max_d = vals[imax]
        sorted_v = sorted(vals, reverse=True)
        second_d = sorted_v[1] if len(sorted_v) > 1 else max_d
        if max_d < min_max:
            return False, "--", d11, d12, d21, d22
        if max_d - second_d < distinct_margin:
            return False, f"{keys[imax]}?", d11, d12, d21, d22
        swap = keys[imax] in ("11", "22")
        return swap, keys[imax], d11, d12, d21, d22

    def _merge_c1_strand_swaps(self, built: List[Tuple[Pair, Any]]) -> List[bool]:
        """Per pair, map source A/B → merged A/B using quad-distance **maximum** (strand vs bend)."""
        n = len(built)
        swaps = [False] * n
        if n < 2:
            return swaps
        c1_per: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = []
        for _p, m in built:
            c1_per.append(
                (
                    self._c1_prime_coord_on_residue(m, "A", 1),
                    self._c1_prime_coord_on_residue(m, "B", 1),
                )
            )
        for i in range(1, n):
            ca_prev, cb_prev = c1_per[i - 1]
            ca_cur, cb_cur = c1_per[i]
            if ca_prev is None or cb_prev is None or ca_cur is None or cb_cur is None:
                if ca_cur is None or cb_cur is None:
                    self.session.logger.warning(
                        f"[BaseHunter] Merge: missing C1′ on pair index {i + 1}; strand alignment skipped for that step."
                    )
                continue
            pa = ca_prev if not swaps[i - 1] else cb_prev
            pb = cb_prev if not swaps[i - 1] else ca_prev
            swap, win, d11, d12, d21, d22 = self._duplex_chain_swap_from_quad_distances(
                pa, pb, ca_cur, cb_cur, distinct_margin=0.25, min_max=3.0
            )
            pid = built[i][0].pair_id
            if win == "--":
                self.session.logger.warning(
                    f"[BaseHunter] Merge: {pid}: quad C1′ distances too small "
                    f"(11/12/21/22={d11:.2f},{d12:.2f},{d21:.2f},{d22:.2f} Å); strand swap skipped."
                )
                continue
            if win.endswith("?"):
                self.session.logger.warning(
                    f"[BaseHunter] Merge: {pid}: ambiguous quad maximum ({win} "
                    f"11/12/21/22={d11:.2f},{d12:.2f},{d21:.2f},{d22:.2f} Å); strand swap skipped."
                )
                continue
            if swap:
                swaps[i] = True
                self.session.logger.info(
                    f"[BaseHunter] Merge: {pid}: swapped source A/B (quad max={win}, "
                    f"11/12/21/22={d11:.2f},{d12:.2f},{d21:.2f},{d22:.2f} Å)."
                )
        return swaps

    def _merge_copy_intra_residue_bonds(self, src_res, old_to_new: Dict[Any, Any]) -> int:
        from chimerax.atomic.struct_edit import add_bond

        nadd = 0
        seen: set[Tuple[int, int]] = set()
        for atom in src_res.atoms:
            for b in getattr(atom, "bonds", ()) or ():
                try:
                    other = b.other_atom(atom)
                except Exception:
                    continue
                if other is None or other.residue is not atom.residue:
                    continue
                na = old_to_new.get(atom)
                nb = old_to_new.get(other)
                if na is None or nb is None:
                    continue
                i1, i2 = id(na), id(nb)
                key = (i1, i2) if i1 <= i2 else (i2, i1)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    if hasattr(na, "connects_to") and na.connects_to(nb):
                        continue
                except Exception:
                    pass
                try:
                    add_bond(na, nb)
                    nadd += 1
                except Exception:
                    pass
        return nadd

    def _merge_try_phosphodiester_bonds(self, out) -> int:
        """Add O3′(i)–P(i+1) only when atoms are already within a covalent bond length (independent pair builds)."""
        from chimerax.atomic.struct_edit import add_bond

        nadd = 0
        for cid in ("A", "B"):
            residues = [r for r in out.residues if str(getattr(r, "chain_id", "")).strip() == cid]
            try:
                residues.sort(key=lambda r: int(getattr(r, "number", 0)))
            except Exception:
                continue
            for j in range(len(residues) - 1):
                r1, r2 = residues[j], residues[j + 1]
                o3 = None
                p_at = None
                for a in r1.atoms:
                    if self._norm_atom_key(a.name) == "O3'":
                        o3 = a
                        break
                for a in r2.atoms:
                    if self._norm_atom_key(a.name) == "P":
                        p_at = a
                        break
                if o3 is None or p_at is None:
                    continue
                d = float(
                    np.linalg.norm(np.asarray(o3.coord, dtype=np.float64) - np.asarray(p_at.coord, dtype=np.float64))
                )
                if d > 2.55:
                    continue
                try:
                    if hasattr(o3, "connects_to") and o3.connects_to(p_at):
                        continue
                except Exception:
                    pass
                try:
                    add_bond(o3, p_at)
                    nadd += 1
                except Exception:
                    pass
        return nadd

    def _merge_pair_models(self, built: List[Tuple[Pair, Any]], title: str):
        """Merge single-pair build outputs into one A/B model with sequential residue numbers.

        Per-pair builds always use residue number ``1`` on chains A/B. We renumber to ``1…N`` here.

        **Do not** call :meth:`AtomicStructure.connect_structure` on the merged model: ChimeraX can
        mis-infer bonds across distant bases on the same chain (native crash). Instead we copy each
        source residue's existing bonds onto the new atoms, and optionally add O3′–P links between
        consecutive residues only when those atoms are already within ~2.55 Å.
        """
        from chimerax.atomic import AtomicStructure, Element
        from chimerax.core.commands import run

        self.session.logger.info(f"[BaseHunter] Merge: combining {len(built)} pair model(s) into «{title}» …")
        out = AtomicStructure(self.session, name=title)
        # Table is the source of truth for chain IDs: preserve per-pair chain A/B in merge.
        swaps = [False] * len(built)
        bonds_copied = 0
        for idx, ((_p, m), flip) in enumerate(zip(built, swaps), start=1):
            chain_map = [("A", "B"), ("B", "A")] if flip else [("A", "A"), ("B", "B")]
            for dst_chain, src_chain in chain_map:
                src_res = self._find_residue_by_chain_number(m, src_chain, 1)
                if src_res is None:
                    self.session.logger.warning(
                        f"[BaseHunter] Merge: missing chain {src_chain}:1 on model #{getattr(m, 'id_string', '?')}; skipped."
                    )
                    continue
                dst = out.new_residue(str(getattr(src_res, "name", "UNK")), dst_chain, idx)
                old_to_new: Dict[Any, Any] = {}
                for atom in src_res.atoms:
                    elc = self._chimerax_atom_element_letter(atom)
                    e = Element.get_element(elc)
                    aname = str(getattr(atom, "name", "X")).strip()
                    a_new = out.new_atom(aname, e)
                    pos = np.array(getattr(atom, "coord"), dtype=np.float64)
                    a_new.coord = [float(pos[0]), float(pos[1]), float(pos[2])]
                    a_new.radius = float(getattr(atom, "radius", 0.45) or 0.45)
                    dst.add_atom(a_new)
                    old_to_new[atom] = a_new
                bonds_copied += self._merge_copy_intra_residue_bonds(src_res, old_to_new)
        if bonds_copied:
            self.session.logger.info(f"[BaseHunter] Merge: copied {bonds_copied} bond(s) from per-pair models (intra-residue).")
        self.session.models.add([out])
        try:
            pd_n = self._merge_try_phosphodiester_bonds(out)
            if pd_n:
                self.session.logger.info(
                    f"[BaseHunter] Merge: added {pd_n} O3′–P bond(s) between consecutive residues (≤ 2.55 Å)."
                )
        except Exception:
            pass
        try:
            self._fix_purine_glycosidic_bonds(out)
        except Exception:
            pass
        try:
            run(self.session, f"style {out.id_string} stick")
        except Exception:
            pass
        for _p, m in built:
            try:
                run(self.session, f"close #{m.id_string}")
            except Exception:
                pass
        self.session.logger.info(
            f"[BaseHunter] Merged {len(built)} per-pair models into {out.id_string} "
            f"(chains A/B, residues 1…{len(built)})."
        )
        return out

    # Expected marker–marker distances (Å): same strand between consecutive pairs ~3–4; WC pair (two chains) ~6–10.
    _MARKER_SAME_STRAND_LO = 2.3
    _MARKER_SAME_STRAND_HI = 5.0
    _MARKER_WC_PAIR_LO = 5.0
    _MARKER_WC_PAIR_HI = 11.5

    def _swap_marker_positions_and_atoms(self, p: Pair) -> None:
        """Exchange marker coordinates and marker atoms only (no ``p.result`` change)."""
        p.marker_a, p.marker_b = p.marker_b, p.marker_a
        p.marker_atom_a, p.marker_atom_b = p.marker_atom_b, p.marker_atom_a
        if p.marker_atom_a is not None and p.marker_a is not None:
            p.marker_atom_a.coord = p.marker_a
        if p.marker_atom_b is not None and p.marker_b is not None:
            p.marker_atom_b.coord = p.marker_b

    def _strand_align_pair_vs_table_previous(self, p: Pair) -> None:
        """Optionally swap Marker A/B vs the previous **ready** pair in table order (quad-max rule).

        Invoked from **Compute** only so click order stays untouched until the user runs scoring.
        """
        if p.marker_a is None or p.marker_b is None:
            return
        cand = [q for q in self._pairs if q.marker_a is not None and q.marker_b is not None]
        try:
            idx = cand.index(p)
        except ValueError:
            return
        if idx <= 0:
            return
        prev = cand[idx - 1]
        if prev.marker_a is None or prev.marker_b is None:
            return
        pa = np.array(prev.marker_a, dtype=np.float64)
        pb = np.array(prev.marker_b, dtype=np.float64)
        a1 = np.array(p.marker_a, dtype=np.float64)
        b1 = np.array(p.marker_b, dtype=np.float64)
        swap, win, d11, d12, d21, d22 = self._duplex_chain_swap_from_quad_distances(
            pa, pb, a1, b1, distinct_margin=0.25, min_max=3.0
        )
        if win == "--":
            self.session.logger.warning(
                f"[BaseHunter] {p.pair_id}: marker strand auto-order skipped vs {prev.pair_id} "
                f"(quad 11/12/21/22={d11:.2f},{d12:.2f},{d21:.2f},{d22:.2f} Å too small)."
            )
            return
        if win.endswith("?"):
            self.session.logger.warning(
                f"[BaseHunter] {p.pair_id}: ambiguous quad max ({win}) vs {prev.pair_id} "
                f"(11/12/21/22={d11:.2f},{d12:.2f},{d21:.2f},{d22:.2f} Å); click order kept."
            )
            return
        if swap:
            self._swap_marker_positions_and_atoms(p)
            if p.result is not None:
                self._swap_marker_axis_scores_in_result(p.result)
                self._apply_marker_colors(p)
            self.session.logger.info(
                f"[BaseHunter] {p.pair_id}: marker A/B reordered on compute using quad max={win} "
                f"(11/12/21/22={d11:.2f},{d12:.2f},{d21:.2f},{d22:.2f} Å) for continuity with {prev.pair_id}."
            )

    def _swap_pair_markers_in_place(self, p: Pair) -> None:
        """Exchange marker A/B so chain A in the built model tracks one physical strand across pairs."""
        self._swap_marker_positions_and_atoms(p)
        if p.result is not None:
            self._swap_marker_axis_scores_in_result(p.result)
        self._apply_marker_colors(p)

    @staticmethod
    def _marker_pair_separation(p: Pair) -> Optional[float]:
        if p.marker_a is None or p.marker_b is None:
            return None
        a = np.array(p.marker_a, dtype=np.float64)
        b = np.array(p.marker_b, dtype=np.float64)
        return float(np.linalg.norm(a - b))

    def _log_intra_pair_marker_geometry(self, p: Pair) -> None:
        d = self._marker_pair_separation(p)
        if d is None:
            return
        lo_s, hi_s = self._MARKER_SAME_STRAND_LO, self._MARKER_SAME_STRAND_HI
        lo_w, hi_w = self._MARKER_WC_PAIR_LO, self._MARKER_WC_PAIR_HI
        if lo_s <= d <= hi_s:
            self.session.logger.warning(
                f"[BaseHunter] {p.pair_id}: marker A–B distance = {d:.2f} Å (looks like same-strand spacing ~3–4 Å). "
                "Markers for one WC pair should sit on **opposite** strands (~6–11 Å apart for base/COM picks)."
            )
        elif lo_w <= d <= hi_w:
            self.session.logger.info(f"[BaseHunter] {p.pair_id}: marker A–B distance = {d:.2f} Å (typical WC pair span).")
        else:
            self.session.logger.warning(
                f"[BaseHunter] {p.pair_id}: marker A–B distance = {d:.2f} Å (unusual vs ~6–11 Å WC or ~3–4 Å same-strand)."
            )

    def _canonicalize_marker_strands_sequential(self, ordered: List[Pair]) -> int:
        """Align marker A/B vs the previous **ready** pair in table (creation) order using the same quad-max rule as
        :meth:`_strand_align_pair_vs_table_previous`.

        ``ordered`` only restricts which pairs participate (e.g. duplex-sorted batch for compute); neighbor links
        follow ``self._pairs`` so this pass cannot fight add-marker orient or undo swaps when helix sort permutes BP
        order vs raster order.
        """
        if len(ordered) < 2:
            return 0
        allowed_ids = {id(p) for p in ordered}
        cand = [
            p
            for p in self._pairs
            if id(p) in allowed_ids and p.marker_a is not None and p.marker_b is not None
        ]
        if len(cand) < 2:
            return 0
        swaps = 0
        for i in range(1, len(cand)):
            prev, cur = cand[i - 1], cand[i]
            pa = np.array(prev.marker_a, dtype=np.float64)
            pb = np.array(prev.marker_b, dtype=np.float64)
            ca = np.array(cur.marker_a, dtype=np.float64)
            cb = np.array(cur.marker_b, dtype=np.float64)
            swap, win, d11, d12, d21, d22 = self._duplex_chain_swap_from_quad_distances(
                pa, pb, ca, cb, distinct_margin=0.25, min_max=3.0
            )
            if win == "--":
                self.session.logger.warning(
                    f"[BaseHunter] {cur.pair_id}: strand canonicalize skipped vs {prev.pair_id} "
                    f"(quad 11/12/21/22={d11:.2f},{d12:.2f},{d21:.2f},{d22:.2f} Å too small)."
                )
                continue
            if win.endswith("?"):
                self.session.logger.warning(
                    f"[BaseHunter] {cur.pair_id}: strand canonicalize ambiguous ({win}) vs {prev.pair_id} "
                    f"(11/12/21/22={d11:.2f},{d12:.2f},{d21:.2f},{d22:.2f} Å); markers unchanged."
                )
                continue
            if swap:
                self._swap_pair_markers_in_place(cur)
                swaps += 1
                self.session.logger.info(
                    f"[BaseHunter] {cur.pair_id}: swapped marker A/B (quad max={win}, "
                    f"11/12/21/22={d11:.2f},{d12:.2f},{d21:.2f},{d22:.2f} Å) vs {prev.pair_id} for compute/build."
                )
        return swaps

    def _maybe_log_scene_and_map_frame(self, map_model) -> None:
        if self._scene_coord_note_logged or map_model is None:
            return
        self._scene_coord_note_logged = True
        self.session.logger.info(
            "[BaseHunter] Map values are sampled in **scene coordinates** (same frame as marker positions from "
            "``atom.scene_coord`` when available). The selected volume must be superposed with the structure that "
            "supplied the markers; otherwise density at built atoms will look wrong even if a template PDB/MRC pair "
            "matches on disk."
        )
        root = Path(self.template_dir.text().strip()).expanduser()
        self._optional_log_template_pdb_map_sample_stats(root)

    def _optional_log_template_pdb_map_sample_stats(self, root: Path) -> None:
        """If template dir has co-registered 1BNA.pdb + map, sample the map at PDB base atoms (file coordinates)."""
        pdb = root / "1BNA.pdb"
        mpath = None
        for ext in (".mrc", ".map", ".ccp4"):
            cand = root / f"1BNA{ext}"
            if cand.is_file():
                mpath = cand
                break
        if not pdb.is_file() or mpath is None:
            return
        try:
            import gemmi  # type: ignore

            from cryomodel.io.mrc import read_map  # type: ignore
        except Exception:
            return
        try:
            vol = read_map(mpath)
            g = vol.grid
            if g is None:
                return
        except Exception:
            return
        coords: List[Tuple[float, float, float]] = []
        for line in pdb.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            an = line[12:16].strip().upper().replace("*", "'")
            if "'" in an or an.startswith("P") or an in {"OP1", "OP2"}:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            coords.append((x, y, z))
            if len(coords) >= 200:
                break
        if len(coords) < 8:
            return
        vals: List[float] = []
        for x, y, z in coords:
            try:
                vals.append(float(g.interpolate_value(gemmi.Position(x, y, z))))
            except Exception:
                continue
        if len(vals) < 8:
            return
        arr = np.array(vals, dtype=np.float64)
        self.session.logger.info(
            f"[BaseHunter] Template-dir co-check {pdb.name} + {mpath.name}: interpolated map at "
            f"{len(vals)} base-heavy-atom sites — min/median/max = {float(np.min(arr)):.4f} / "
            f"{float(np.median(arr)):.4f} / {float(np.max(arr)):.4f} (same Å frame as PDB xyz in the files)."
        )

    def _duplex_step_direction_for_pairs(self, pairs: List[Pair]) -> np.ndarray:
        """Unit vector along the duplex step used to order pairs.

        When ``p.result`` exists, uses purine_marker − pyrimidine_marker from the WC assignment.
        Before any scores exist, ``_assignment_for`` would default to A–T for every pair, which
        biases ordering and duplex rise; in that case use marker_B − marker_A (same convention
        as scoring axis ``b − a`` in :meth:`_compute_result`).
        """
        dirs: List[np.ndarray] = []
        for p in pairs:
            if p.marker_a is None or p.marker_b is None:
                continue
            a = np.array(p.marker_a, dtype=np.float64)
            b = np.array(p.marker_b, dtype=np.float64)
            call = self._assignment_for(p)
            if call and p.result is not None:
                la, lb = self._letters_at_markers(p, call)
                if self._is_purine_letter(la):
                    pur, pyr = a, b
                else:
                    pur, pyr = b, a
                v = pur - pyr
            else:
                v = b - a
            n = float(np.linalg.norm(v))
            if n > 1e-9:
                dirs.append(v / n)
        if not dirs:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return self._normalize(np.mean(np.stack(dirs, axis=0), axis=0))

    def _class_template_paths(self, root: Path) -> Tuple[Optional[Path], Optional[Path]]:
        """``templateBP-purine.pdb`` / ``templateBP-pyrimidine.pdb`` preferred; fall back to ``referencePDB-*.pdb``."""
        pur: Optional[Path] = None
        pyr: Optional[Path] = None
        for fn in ("templateBP-purine.pdb", "templatebp-purine.pdb", "referencePDB-purine.pdb"):
            p = root / fn
            if p.is_file():
                pur = p
                break
        for fn in ("templateBP-pyrimidine.pdb", "templatebp-pyrimidine.pdb", "referencePDB-pyrimidine.pdb"):
            p = root / fn
            if p.is_file():
                pyr = p
                break
        return pur, pyr

    def _order_pairs_along_duplex(self, pairs: List[Pair]) -> List[Pair]:
        """Sort pairs along mean WC step so residue numbers increase 5'→3' on each strand (dsDNA, antiparallel partners)."""
        work: List[Pair] = []
        for p in pairs:
            call = self._assignment_for(p)
            if not call or p.marker_a is None or p.marker_b is None:
                continue
            work.append(p)
        if len(work) <= 1:
            return work
        rise = self._duplex_step_direction_for_pairs(work)
        p0 = work[0]
        a0 = np.array(p0.marker_a, dtype=np.float64)
        b0 = np.array(p0.marker_b, dtype=np.float64)
        mid0 = 0.5 * (a0 + b0)
        scored: List[Tuple[float, str, Pair]] = []
        for p in work:
            a = np.array(p.marker_a, dtype=np.float64)
            b = np.array(p.marker_b, dtype=np.float64)
            mid = 0.5 * (a + b)
            s = float(np.dot(mid - mid0, rise))
            scored.append((s, p.pair_id, p))
        scored.sort(key=lambda t: (t[0], t[1]))
        return [t[2] for t in scored]

    def _build_pairs(self, pairs: List[Pair], title: str):
        """Build WC pairs (chains A/B). ``templateBP.pdb`` is **geometry-only** (e.g. C–C WC frame); class shapes from
        ``templateBP-purine`` / ``templateBP-pyrimidine`` (else ``referencePDB-purine`` / ``referencePDB-pyrimidine``).
        BaseHunter scores map pur vs pyr to markers; placement explores **both** chain↔marker directions (see
        ``step0_*`` under ``basehunter_intermediates``), then joint NSP ``fitmap`` to preserve the dimer rigidly.

        Debug: ``step0_*``, ``step1_templateBP_selected.pdb``, ``step2_class_templates_on_geometry.pdb``,
        ``step3_after_per_chain_fitmap.pdb`` (one dimer model like step 2; ``fitmap`` chain A then B with
        ``moveWholeMolecules false``, then rigid (R,t) from template→fitted coords per chain),
        ``step4``–``step6`` files only when ``_BASEHUNTER_RUN_STAGES_AFTER_3`` is True (joint / extra polish).
        """
        from chimerax.atomic import AtomicStructure, Element
        from chimerax.core.commands import run

        root = Path(self.template_dir.text().strip()).expanduser()
        pur_path, pyr_path = self._class_template_paths(root)
        if pur_path is None or pyr_path is None:
            self.status.setText(
                "Build requires templateBP-purine.pdb and templateBP-pyrimidine.pdb "
                "(or referencePDB-purine.pdb / referencePDB-pyrimidine.pdb) in the template directory."
            )
            self.session.logger.warning("[BaseHunter] Missing purine/pyrimidine class template PDBs.")
            return None
        if self._template_bp_path(root) is None:
            self.status.setText(
                "Build requires templateBP.pdb in the template directory."
            )
            self.session.logger.warning("[BaseHunter] Missing templateBP.pdb for phased build.")
            return None
        ref_pur_atoms = self._parse_pdb_all_atoms(pur_path)
        ref_pyr_atoms = self._parse_pdb_all_atoms(pyr_path)
        if len(ref_pur_atoms) < 8 or len(ref_pyr_atoms) < 8:
            self.status.setText("Purine/pyrimidine class templates could not be read or are too small.")
            return None

        map_model = self._current_map_model()
        thr = self._threshold_value()

        ordered_pairs = self._order_pairs_along_duplex(pairs)
        n_strand_swaps = self._canonicalize_marker_strands_sequential(ordered_pairs)
        if n_strand_swaps:
            self._sync_table()
        self._maybe_log_scene_and_map_frame(map_model)
        if len(ordered_pairs) > 1:
            self.session.logger.info(
                "[BaseHunter] dsDNA: "
                f"{len(ordered_pairs)} pairs along duplex (chain A = marker_a, chain B = marker_b, antiparallel); "
                f"order {', '.join(p.pair_id for p in ordered_pairs)}"
            )
        elif len(ordered_pairs) == 1:
            self.session.logger.info(
                "[BaseHunter] dsDNA: single pair; chain A ← marker_a, chain B ← marker_b (duplex partners)."
            )

        model = AtomicStructure(self.session, name=title)
        for idx, p in enumerate(ordered_pairs, start=1):
            call = self._assignment_for(p)
            if not call or p.marker_a is None or p.marker_b is None:
                continue
            try:
                self._log_intra_pair_marker_geometry(p)
                la, lb = self._letters_at_markers(p, call)
                d_show = self._display_decision(p)
                raw = self._raw_density_classes(p)
                ra, rb = (raw or ("?", "?"))
                da = str(d_show["class_a"]) if d_show else "?"
                db = str(d_show["class_b"]) if d_show else "?"
                self.session.logger.info(
                    f"[BaseHunter] Build pair {p.pair_id} resIdx={idx}: assignment_call={call} → "
                    f"letters markerA/markerB={la},{lb} | raw_density A={ra} B={rb} | display_class A={da} B={db} | "
                    f"markers {self._marker_atom_label(p.marker_atom_a)} | {self._marker_atom_label(p.marker_atom_b)}"
                )
                a_xyz = np.array(p.marker_a, dtype=np.float64)
                b_xyz = np.array(p.marker_b, dtype=np.float64)
                # Pair frame: e1 from purine marker toward pyrimidine marker (WC geometry for ±e3 conventions).
                if self._is_purine_letter(la):
                    pur_xyz, pyr_xyz = a_xyz, b_xyz
                else:
                    pur_xyz, pyr_xyz = b_xyz, a_xyz
                e1, e2w, e3w = self._pair_world_frame(pyr_xyz - pur_xyz)

                dbg = self._pair_intermediate_dir(root, p.pair_id)
                bp_sol = self._solve_template_bp_placement(
                    a_xyz, b_xyz, map_model, thr, dbg, resseq=idx, swap_only=False
                )

                if bp_sol is not None:
                    r_at, t_at, sc_a, sc_b, swap_markers, bp_summary = bp_sol
                    s1_title = ("step1 " + bp_summary.replace(";", ","))[:70]
                    self._write_pdb_debug(
                        dbg / "step1_templateBP_selected.pdb",
                        s1_title,
                        [
                            ("A", idx, "DC", sc_a, r_at, t_at),
                            ("B", idx, "DC", sc_b, r_at, t_at),
                        ],
                    )
                    class_a = str(d_show.get("class_a", "")).strip().lower() if d_show else ""
                    marker_a_is_pur = (
                        class_a == "purine" if class_a in {"purine", "pyrimidine"} else self._is_purine_letter(la)
                    )
                    pur_on_site_first = marker_a_is_pur == (not swap_markers)
                    if pur_on_site_first:
                        tgt_pur_sp = self._sp_dict_template_world(sc_a, r_at, t_at)
                        tgt_pyr_sp = self._sp_dict_template_world(sc_b, r_at, t_at)
                        Rpp, tpp = self._kabsch_align_full_to_sp_target(ref_pur_atoms, tgt_pur_sp)
                        Ryy, tyy = self._kabsch_align_full_to_sp_target(ref_pyr_atoms, tgt_pyr_sp)
                        full_a, RA0, tA0 = ref_pur_atoms, Rpp, tpp
                        full_b, RB0, tB0 = ref_pyr_atoms, Ryy, tyy
                    else:
                        tgt_pur_sp = self._sp_dict_template_world(sc_b, r_at, t_at)
                        tgt_pyr_sp = self._sp_dict_template_world(sc_a, r_at, t_at)
                        Rpp, tpp = self._kabsch_align_full_to_sp_target(ref_pur_atoms, tgt_pur_sp)
                        Ryy, tyy = self._kabsch_align_full_to_sp_target(ref_pyr_atoms, tgt_pyr_sp)
                        full_a, RA0, tA0 = ref_pyr_atoms, Ryy, tyy
                        full_b, RB0, tB0 = ref_pur_atoms, Rpp, tpp
                    c_base_a = self._base_centroid_for_build(la, full_a)
                    c_base_b = self._base_centroid_for_build(lb, full_b)
                    self._write_pdb_debug(
                        dbg / "step2_class_templates_on_geometry.pdb",
                        "step2",
                        [
                            ("A", idx, self._dna_three_letter(la), full_a, RA0, tA0),
                            ("B", idx, self._dna_three_letter(lb), full_b, RB0, tB0),
                        ],
                    )
                else:
                    full_a = self._load_nucleotide_for_build(la)
                    full_b = self._load_nucleotide_for_build(lb)
                    if not full_a or not full_b:
                        continue
                    c_base_a = self._base_centroid_for_build(la, full_a)
                    c_base_b = self._base_centroid_for_build(lb, full_b)
                    placed = self._place_from_template_bp(la, lb, full_a, full_b, pur_xyz, pyr_xyz, a_xyz, b_xyz)
                    if placed is not None:
                        RA0, tA0, RB0, tB0, seed5 = placed
                        rs, ts, scy, scu = seed5
                        self._write_pdb_debug(
                            dbg / "step1_templateBP_fallback.pdb",
                            "step1_fallback",
                            [
                                ("A", idx, "PYM", scy, rs, ts),
                                ("B", idx, "PUR", scu, rs, ts),
                            ],
                        )
                        self.session.logger.info(
                            "[BaseHunter] Initial placement: templateBP dimer (scaled) + backbone Kabsch + base-COM nudge."
                        )
                        self._write_pdb_debug(
                            dbg / "step2_class_templates_on_geometry.pdb",
                            "step2",
                            [
                                ("A", idx, self._dna_three_letter(la), full_a, RA0, tA0),
                                ("B", idx, self._dna_three_letter(lb), full_b, RB0, tB0),
                            ],
                        )
                    else:
                        RA0 = self._rot_place_by_chemistry(la, full_a, e1, e2w, e3w, self._is_purine_letter(la))
                        RB0 = self._rot_place_by_chemistry(lb, full_b, e1, e2w, e3w, self._is_purine_letter(lb))
                        tA0 = a_xyz - RA0 @ c_base_a
                        tB0 = b_xyz - RB0 @ c_base_b

                if not full_a or not full_b:
                    continue
                if d_show is not None:
                    self.session.logger.info(
                        f"[BaseHunter] Pair {p.pair_id}: built residue letters (chains A/B) = {la}, {lb}; "
                        f"display classes marker_a={d_show['class_a']}, marker_b={d_show['class_b']}; "
                        f"assignment call={call}"
                    )

                ca = np.stack([x.coord for x in full_a], axis=0)
                cb = np.stack([x.coord for x in full_b], axis=0)

                n_a0 = self._wc_base_plane_normal_template(la, full_a)
                n_b0 = self._wc_base_plane_normal_template(lb, full_b)
                c1_a0 = self._find_c1_prime(full_a)
                c1_b0 = self._find_c1_prime(full_b)
                # Keep stage-2 seeded placement stable; avoid aggressive map-driven joint searches here.
                RA, tA, RB, tB = RA0, tA0, RB0, tB0
                if p.build_flip:
                    RA, tA, RB, tB = self._flip_rt_about_ab_axis(RA, tA, RB, tB, a_xyz, b_xyz)

                # Stage 3: per-chain ``fitmap`` on one dimer model (on by default; disable via ``_BASEHUNTER_RUN_STAGE3_FITMAP``).
                if map_model is not None:
                    if _BASEHUNTER_RUN_STAGE3_FITMAP:
                        m_s3 = self._session_dimer_from_templates(
                            f"BaseHunter step3_dimer {p.pair_id}",
                            idx,
                            self._dna_three_letter(la),
                            full_a,
                            RA,
                            tA,
                            self._dna_three_letter(lb),
                            full_b,
                            RB,
                            tB,
                        )
                        try:
                            atoms_a = self._atoms_chain_resseq(m_s3, "A", idx)
                            atoms_b = self._atoms_chain_resseq(m_s3, "B", idx)
                            if len(atoms_a) >= 3:
                                self.session.logger.info(
                                    f"[BaseHunter] Stage3 fitmap chain A ({len(atoms_a)} atoms), moveWholeMolecules false — {p.pair_id}"
                                )
                                self._try_fitmap_atom_list(
                                    atoms_a,
                                    map_model,
                                    move_whole_molecules=False,
                                    allow_shift=True,
                                    allow_rotate=True,
                                    envelope=True,
                                )
                                RA, tA = self._rt_from_template_to_struct_residue(full_a, m_s3, "A", idx, RA, tA)
                            if len(atoms_b) >= 3:
                                self.session.logger.info(
                                    f"[BaseHunter] Stage3 fitmap chain B ({len(atoms_b)} atoms), moveWholeMolecules false — {p.pair_id}"
                                )
                                self._try_fitmap_atom_list(
                                    atoms_b,
                                    map_model,
                                    move_whole_molecules=False,
                                    allow_shift=True,
                                    allow_rotate=True,
                                    envelope=True,
                                )
                                RB, tB = self._rt_from_template_to_struct_residue(full_b, m_s3, "B", idx, RB, tB)
                        finally:
                            try:
                                run(self.session, f"close #{m_s3.id_string}")
                            except Exception:
                                pass
                        self._log_stage_geometry(
                            "Stage3 per-chain fitmap on dimer (moveWholeMolecules false; shift+rotate)",
                            p.pair_id,
                            la,
                            lb,
                            full_a,
                            full_b,
                            RA,
                            tA,
                            RB,
                            tB,
                            n_a0,
                            n_b0,
                        )
                    else:
                        self.session.logger.info(
                            f"[BaseHunter] Stage3 fitmap skipped (bases frozen at stage-2 placement) — {p.pair_id}"
                        )
                        self._log_stage_geometry(
                            "Stage3 frozen (no fitmap; stage-2 pose kept)",
                            p.pair_id,
                            la,
                            lb,
                            full_a,
                            full_b,
                            RA,
                            tA,
                            RB,
                            tB,
                            n_a0,
                            n_b0,
                        )

                    self._write_pdb_debug(
                        dbg / "step3_after_per_chain_fitmap.pdb",
                        "step3_per_chain_fitmap",
                        [
                            ("A", idx, self._dna_three_letter(la), full_a, RA, tA),
                            ("B", idx, self._dna_three_letter(lb), full_b, RB, tB),
                        ],
                    )

                    def _log_frac_in_map() -> None:
                        world_heavy: List[TemplateAtom] = []
                        for a in full_a:
                            if (a.element or "C").upper() != "H":
                                world_heavy.append(
                                    TemplateAtom(name=a.name, element=a.element, coord=RA @ a.coord + tA)
                                )
                        for a in full_b:
                            if (a.element or "C").upper() != "H":
                                world_heavy.append(
                                    TemplateAtom(name=a.name, element=a.element, coord=RB @ a.coord + tB)
                                )
                        frac_hi, n_ok, n_tot = self._frac_heavy_atoms_above_threshold(map_model, thr, world_heavy)
                        self.session.logger.info(
                            f"[BaseHunter] {p.pair_id}: heavy atoms ≥ map threshold = {frac_hi:.1%} ({n_ok}/{n_tot}) "
                            f"(target >90% when map is available)."
                        )

                    if not _BASEHUNTER_RUN_STAGES_AFTER_3:
                        _log_frac_in_map()

                    if _BASEHUNTER_RUN_STAGES_AFTER_3:
                        nsp_joint = frozenset(self._NSP_ALL_CLASS)
                        nsp_a = frozenset(
                            self._NSP_PURINE_ATOMS if self._is_purine_letter(la) else self._NSP_PYRIMIDINE_ATOMS
                        )
                        nsp_b = frozenset(
                            self._NSP_PURINE_ATOMS if self._is_purine_letter(lb) else self._NSP_PYRIMIDINE_ATOMS
                        )
                        self._write_pdb_debug(
                            dbg / "step4_before_joint_fitmap.pdb",
                            "step4_before_joint",
                            [
                                ("A", idx, self._dna_three_letter(la), full_a, RA, tA),
                                ("B", idx, self._dna_three_letter(lb), full_b, RB, tB),
                            ],
                        )
                        m_bp4 = self._session_dimer_from_templates(
                            f"BaseHunter joint4 {p.pair_id}",
                            idx,
                            self._dna_three_letter(la),
                            full_a,
                            RA,
                            tA,
                            self._dna_three_letter(lb),
                            full_b,
                            RB,
                            tB,
                        )
                        if self._try_fitmap_subset(
                            m_bp4,
                            map_model,
                            thr,
                            False,
                            move_whole_molecules=True,
                            allow_shift=False,
                            allow_rotate=True,
                            atom_subset=nsp_joint,
                        ):
                            RA, tA, RB, tB = self._apply_fitmap_joint_to_pair_rt(
                                full_a, RA, tA, full_b, RB, tB, m_bp4
                            )
                        try:
                            run(self.session, f"close #{m_bp4.id_string}")
                        except Exception:
                            pass
                        self._write_pdb_debug(
                            dbg / "step4_after_joint_fitmap.pdb",
                            "step4_joint",
                            [
                                ("A", idx, self._dna_three_letter(la), full_a, RA, tA),
                                ("B", idx, self._dna_three_letter(lb), full_b, RB, tB),
                            ],
                        )
                        self._log_stage_geometry(
                            "Stage4 joint NSP fitmap (single rigid dimer, rotation-only)",
                            p.pair_id,
                            la,
                            lb,
                            full_a,
                            full_b,
                            RA,
                            tA,
                            RB,
                            tB,
                            n_a0,
                            n_b0,
                        )
                        m_bp5 = self._session_dimer_from_templates(
                            f"BaseHunter joint5 {p.pair_id}",
                            idx,
                            self._dna_three_letter(la),
                            full_a,
                            RA,
                            tA,
                            self._dna_three_letter(lb),
                            full_b,
                            RB,
                            tB,
                        )
                        if self._try_fitmap_subset(
                            m_bp5,
                            map_model,
                            thr,
                            False,
                            move_whole_molecules=True,
                            allow_shift=False,
                            allow_rotate=True,
                            atom_subset=nsp_joint,
                        ):
                            RA, tA, RB, tB = self._apply_fitmap_joint_to_pair_rt(
                                full_a, RA, tA, full_b, RB, tB, m_bp5
                            )
                        try:
                            run(self.session, f"close #{m_bp5.id_string}")
                        except Exception:
                            pass
                        self._write_pdb_debug(
                            dbg / "step5_after_joint_fitmap_rot.pdb",
                            "step5_joint_rot",
                            [
                                ("A", idx, self._dna_three_letter(la), full_a, RA, tA),
                                ("B", idx, self._dna_three_letter(lb), full_b, RB, tB),
                            ],
                        )
                        self._log_stage_geometry(
                            "Stage5 joint NSP fitmap (rotation-only)",
                            p.pair_id,
                            la,
                            lb,
                            full_a,
                            full_b,
                            RA,
                            tA,
                            RB,
                            tB,
                            n_a0,
                            n_b0,
                        )
                        m_r6a = self._session_structure_from_template(
                            f"BaseHunter rot_only A {p.pair_id}",
                            "A",
                            idx,
                            self._dna_three_letter(la),
                            full_a,
                            RA,
                            tA,
                        )
                        if self._try_fitmap_subset(
                            m_r6a,
                            map_model,
                            thr,
                            self._is_purine_letter(la),
                            move_whole_molecules=True,
                            allow_shift=False,
                            allow_rotate=True,
                            atom_subset=nsp_a,
                        ):
                            RA, tA = self._apply_fitmap_to_rt(full_a, RA, tA, m_r6a)
                        try:
                            run(self.session, f"close #{m_r6a.id_string}")
                        except Exception:
                            pass
                        m_r6b = self._session_structure_from_template(
                            f"BaseHunter rot_only B {p.pair_id}",
                            "B",
                            idx,
                            self._dna_three_letter(lb),
                            full_b,
                            RB,
                            tB,
                        )
                        if self._try_fitmap_subset(
                            m_r6b,
                            map_model,
                            thr,
                            self._is_purine_letter(lb),
                            move_whole_molecules=True,
                            allow_shift=False,
                            allow_rotate=True,
                            atom_subset=nsp_b,
                        ):
                            RB, tB = self._apply_fitmap_to_rt(full_b, RB, tB, m_r6b)
                        try:
                            run(self.session, f"close #{m_r6b.id_string}")
                        except Exception:
                            pass
                        self._write_pdb_debug(
                            dbg / "step6_after_per_residue_rotate_only.pdb",
                            "step6_rot_only",
                            [
                                ("A", idx, self._dna_three_letter(la), full_a, RA, tA),
                                ("B", idx, self._dna_three_letter(lb), full_b, RB, tB),
                            ],
                        )
                        self._log_stage_geometry(
                            "Stage6 per-residue rotate-only (NSP, whole residue)",
                            p.pair_id,
                            la,
                            lb,
                            full_a,
                            full_b,
                            RA,
                            tA,
                            RB,
                            tB,
                            n_a0,
                            n_b0,
                        )
                        _log_frac_in_map()
                else:
                    self._write_pdb_debug(
                        dbg / "step6_pair_fragment.pdb",
                        "step6_no_map",
                        [("A", idx, self._dna_three_letter(la), full_a, RA, tA), ("B", idx, self._dna_three_letter(lb), full_b, RB, tB)],
                    )

                # Final geometry summary (same metrics as Stage3/5 logs).
                self._log_stage_geometry(
                    "Build final",
                    p.pair_id,
                    la,
                    lb,
                    full_a,
                    full_b,
                    RA,
                    tA,
                    RB,
                    tB,
                    n_a0,
                    n_b0,
                )

                r1 = model.new_residue(self._dna_three_letter(la), "A", idx)
                for atom in full_a:
                    elc = (atom.element or "C").strip().upper()[:1] or "C"
                    e = Element.get_element(elc)
                    a1 = model.new_atom(atom.name, e)
                    pos = RA @ atom.coord + tA
                    a1.coord = [float(pos[0]), float(pos[1]), float(pos[2])]
                    a1.radius = 0.45
                    r1.add_atom(a1)

                r2 = model.new_residue(self._dna_three_letter(lb), "B", idx)
                for atom in full_b:
                    elc = (atom.element or "C").strip().upper()[:1] or "C"
                    e = Element.get_element(elc)
                    a2 = model.new_atom(atom.name, e)
                    pos = RB @ atom.coord + tB
                    a2.coord = [float(pos[0]), float(pos[1]), float(pos[2])]
                    a2.radius = 0.45
                    r2.add_atom(a2)

            except Exception:
                self.session.logger.error(
                    f"[BaseHunter] Build pair {p.pair_id} resIdx={idx} failed:\n{traceback.format_exc()}"
                )
                continue
        self.session.models.add([model])
        self._refresh_models(prefer_structure=model)
        try:
            model.connect_structure(bond_length_tolerance=0.40)
        except Exception:
            pass
        try:
            self._fix_purine_glycosidic_bonds(model)
        except Exception:
            pass
        try:
            run(self.session, f"style {model.id_string} stick")
        except Exception:
            pass
        self.session.logger.info(
            f"[BaseHunter] Built model {model.id_string} with {len(ordered_pairs)} pair(s) "
            f"on chains A/B (residue numbers 1…{len(ordered_pairs)} per chain)."
        )
        return model

    @staticmethod
    def _json_safe_export(obj: Any) -> Any:
        """Recursively convert results (numpy, tuples, nested dicts) for JSON export."""
        if obj is None:
            return None
        if isinstance(obj, (str, int, bool)):
            return obj
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return str(obj)
            return float(obj)
        if isinstance(obj, np.ndarray):
            return [float(x) for x in np.ravel(obj).tolist()]
        if isinstance(obj, np.floating):
            x = float(obj)
            if math.isnan(x) or math.isinf(x):
                return str(x)
            return x
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {str(k): BaseHunterInteractiveTool._json_safe_export(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [BaseHunterInteractiveTool._json_safe_export(v) for v in obj]
        if isinstance(obj, set):
            return [BaseHunterInteractiveTool._json_safe_export(v) for v in sorted(obj, key=lambda x: str(x))]
        try:
            return str(obj)
        except Exception:
            return None

    def _export_slot_model_record(self, *, kind: str) -> Dict[str, Any]:
        if kind == "map":
            combo, models = self.map_combo, self._map_models
        else:
            combo, models = self.model_combo, self._structure_models
        i = combo.currentIndex()
        label = combo.currentText()
        if i < 0 or i >= len(models):
            return {"combo_index": i, "combo_text": label, "id_string": None, "disk_path": None}
        m = models[i]
        did = getattr(m, "id_string", None)
        return {"combo_index": i, "combo_text": label, "id_string": str(did) if did is not None else None, "disk_path": _disk_path_for_model(m)}

    def _export_calculations_json(self):
        path, _ = QFileDialog.getSaveFileName(
            self.tool_window.ui_area,
            "Export BaseHunter calculations",
            str(Path.home() / "basehunter_calculations.json"),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        thr, thr_src = self._threshold_value_with_source()
        vol_thr: Optional[float] = None
        try:
            vm = self._current_map_model()
            if vm is not None:
                vol_thr = self._volume_display_threshold(vm)
        except Exception:
            pass
        pairs_out: List[Dict[str, Any]] = []
        for p in self._pairs:
            pairs_out.append(
                {
                    "pair_id": p.pair_id,
                    "status": p.status,
                    "build_flip": bool(getattr(p, "build_flip", False)),
                    "marker_a_xyz": p.marker_a,
                    "marker_b_xyz": p.marker_b,
                    "marker_atom_a": self._marker_atom_label(p.marker_atom_a),
                    "marker_atom_b": self._marker_atom_label(p.marker_atom_b),
                    "result": p.result,
                }
            )
        payload: Dict[str, Any] = {
            "export_version": 1,
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "quality_setting": self.quality.currentText().strip(),
            "post_build_refine_auto": self._quality_uses_post_build_refine(),
            "threshold": {"value": float(thr), "source": thr_src, "volume_viewer_level": vol_thr},
            "inherit_volume_threshold": bool(self.inherit_threshold.isChecked()),
            "manual_threshold_text": self.threshold_edit.text().strip(),
            "map": self._export_slot_model_record(kind="map"),
            "working_model": self._export_slot_model_record(kind="structure"),
            "template_dir": self.template_dir.text().strip(),
            "selected_pair_id": self._selected_pair_id,
            "session_counter": self._counter,
            "pairs": pairs_out,
        }
        safe = self._json_safe_export(payload)
        Path(path).write_text(json.dumps(safe, indent=2), encoding="utf-8")
        self.status.setText(f"Calculations exported: {path}")
        self.session.logger.info(f"[BaseHunter] Calculations JSON written to {path}")

    def _save_session(self):
        path, _ = QFileDialog.getSaveFileName(
            self.tool_window.ui_area,
            "Save BaseHunter session",
            str(Path.home() / "basehunter_session.json"),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        payload = {
            "pairs": [
                {
                    "pair_id": p.pair_id,
                    "marker_a": p.marker_a,
                    "marker_b": p.marker_b,
                    "status": p.status,
                    "result": p.result,
                    "build_flip": bool(getattr(p, "build_flip", False)),
                }
                for p in self._pairs
            ],
            "selected_pair_id": self._selected_pair_id,
            "counter": self._counter,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.status.setText(f"Session saved: {path}")

    def _load_session(self):
        path, _ = QFileDialog.getOpenFileName(
            self.tool_window.ui_area,
            "Load BaseHunter session",
            str(Path.home()),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self._pairs = []
        for x in payload.get("pairs", []):
            self._pairs.append(
                Pair(
                    pair_id=str(x.get("pair_id")),
                    marker_a=x.get("marker_a"),
                    marker_b=x.get("marker_b"),
                    status=str(x.get("status", "new")),
                    result=x.get("result"),
                    build_flip=bool(x.get("build_flip", False)),
                )
            )
        self._selected_pair_id = payload.get("selected_pair_id")
        self._counter = int(payload.get("counter", len(self._pairs)))
        self._sync_table()
        self.status.setText(f"Session loaded: {path}")

