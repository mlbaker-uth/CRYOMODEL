from __future__ import annotations

import csv
import math
import tempfile
import traceback
from pathlib import Path

from Qt.QtCore import Qt
from Qt.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QPushButton,
    QPlainTextEdit, QSpinBox, QDoubleSpinBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget
)

from chimerax.atomic import selected_atoms
from chimerax.core.commands import run as cxrun
from chimerax.core.errors import UserError
from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow

from .domain_model import DomainModel, DomainRow
from .pdbcom_engine import PDBComEngine


def _distance(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _plane_from_points(p1, p2, p3):
    u = [p2[i] - p1[i] for i in range(3)]
    v = [p3[i] - p1[i] for i in range(3)]
    n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
    norm = math.sqrt(sum(x * x for x in n))
    if norm == 0:
        raise ValueError("Reference plane points are collinear")
    n = [x / norm for x in n]
    return p1, n


def _signed_distance_to_plane(point, plane_point, plane_normal):
    return sum((point[i] - plane_point[i]) * plane_normal[i] for i in range(3))


class CryoModelDomainCOMTool(ToolInstance):
    SESSION_ENDURING = False
    SESSION_SAVE = False

    COL_USE = 0
    COL_DOMAIN = 1
    COL_CHAIN = 2
    COL_START = 3
    COL_END = 4
    COL_COLOR = 5

    def __init__(self, session, tool_name="CryoModel Domain COM"):
        super().__init__(session, tool_name)
        self.display_name = "CryoModel Domain COM"
        self.tw = MainToolWindow(self)
        self.domain_model = DomainModel()
        self._com_rows = []
        self._com_pdb_path = None

        parent = self.tw.ui_area
        root = QVBoxLayout(parent)
        self._build_top_controls(root)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        self._build_domains_tab()
        self._build_com_tab()
        self._build_log_panel(root)
        self._refresh_structure_menu()
        self.tw.manage(None)

    def _build_top_controls(self, parent_layout):
        box = QGroupBox("General")
        grid = QGridLayout(box)
        self.structure_combo = QComboBox()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh_structure_menu)
        self.exe_edit = QLineEdit("cryomodel")
        exe_btn = QPushButton("Browse…")
        exe_btn.clicked.connect(self._browse_executable)
        self.prefix_edit = QLineEdit("domains")
        self.workdir_edit = QLineEdit("")
        wd_btn = QPushButton("Browse…")
        wd_btn.clicked.connect(self._browse_workdir)

        grid.addWidget(QLabel("Structure"), 0, 0)
        grid.addWidget(self.structure_combo, 0, 1)
        grid.addWidget(refresh, 0, 2)
        grid.addWidget(QLabel("CryoModel executable"), 1, 0)
        grid.addWidget(self.exe_edit, 1, 1)
        grid.addWidget(exe_btn, 1, 2)
        grid.addWidget(QLabel("Output prefix"), 2, 0)
        grid.addWidget(self.prefix_edit, 2, 1)
        grid.addWidget(QLabel("Working directory"), 3, 0)
        grid.addWidget(self.workdir_edit, 3, 1)
        grid.addWidget(wd_btn, 3, 2)
        parent_layout.addWidget(box)

    def _build_domains_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        source_box = QGroupBox("Domain Source")
        source_grid = QGridLayout(source_box)
        self.domain_file_edit = QLineEdit("")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_domain_file)
        load = QPushButton("Load")
        load.clicked.connect(self._load_domain_file)
        self.chain_combo = QComboBox()
        self.n_domains = QSpinBox(); self.n_domains.setRange(0, 100); self.n_domains.setValue(0)
        infer_btn = QPushButton("Infer Domains")
        infer_btn.clicked.connect(self._infer_domains)

        source_grid.addWidget(QLabel("Domain file"), 0, 0)
        source_grid.addWidget(self.domain_file_edit, 0, 1)
        source_grid.addWidget(browse, 0, 2)
        source_grid.addWidget(load, 0, 3)
        source_grid.addWidget(QLabel("Chain"), 1, 0)
        source_grid.addWidget(self.chain_combo, 1, 1)
        source_grid.addWidget(QLabel("# Domains (0=auto)"), 1, 2)
        source_grid.addWidget(self.n_domains, 1, 3)
        source_grid.addWidget(infer_btn, 2, 3)
        layout.addWidget(source_box)

        adv = QGroupBox("pdbdomain Advanced")
        adv_grid = QGridLayout(adv)
        self.merge_distance = QDoubleSpinBox(); self.merge_distance.setRange(0.0, 1000.0); self.merge_distance.setValue(25.0)
        self.seed_size = QSpinBox(); self.seed_size.setRange(1, 1000); self.seed_size.setValue(20)
        self.min_domain_res = QSpinBox(); self.min_domain_res.setRange(0, 10000); self.min_domain_res.setValue(50)
        self.prefer_gaps = QCheckBox("Prefer gaps"); self.prefer_gaps.setChecked(True)
        self.gaps_only = QCheckBox("Gaps only")
        self.gap_window = QSpinBox(); self.gap_window.setRange(0, 1000); self.gap_window.setValue(10)
        self.sse_source = QComboBox(); self.sse_source.addItems(["header", "dssp", "auto", "none"])
        self.sse_window = QSpinBox(); self.sse_window.setRange(0, 1000); self.sse_window.setValue(10)
        self.write_pdb = QCheckBox("Write PDB"); self.write_pdb.setChecked(True)
        adv_grid.addWidget(QLabel("Merge distance"), 0, 0); adv_grid.addWidget(self.merge_distance, 0, 1)
        adv_grid.addWidget(QLabel("Seed size"), 0, 2); adv_grid.addWidget(self.seed_size, 0, 3)
        adv_grid.addWidget(QLabel("Min domain residues"), 1, 0); adv_grid.addWidget(self.min_domain_res, 1, 1)
        adv_grid.addWidget(self.prefer_gaps, 1, 2); adv_grid.addWidget(self.gaps_only, 1, 3)
        adv_grid.addWidget(QLabel("Gap window"), 2, 0); adv_grid.addWidget(self.gap_window, 2, 1)
        adv_grid.addWidget(QLabel("SSE source"), 2, 2); adv_grid.addWidget(self.sse_source, 2, 3)
        adv_grid.addWidget(QLabel("SSE window"), 3, 0); adv_grid.addWidget(self.sse_window, 3, 1)
        adv_grid.addWidget(self.write_pdb, 3, 2)
        layout.addWidget(adv)

        editor_box = QGroupBox("Domain Definitions")
        editor_layout = QVBoxLayout(editor_box)
        self.domain_table = QTableWidget(0, 6)
        self.domain_table.setHorizontalHeaderLabels(["Use", "Domain", "Chain", "Start", "End", "Color"])
        hdr = self.domain_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Stretch)
        hdr.setSectionResizeMode(self.COL_USE, QHeaderView.ResizeToContents)
        editor_layout.addWidget(self.domain_table)
        btn_row = QHBoxLayout()
        for label, fn in [
            ("Add Row", self._add_domain_row),
            ("Delete Row", self._delete_selected_rows),
            ("Rename Domain", self._rename_selected_domain),
            ("Join Selected", self._join_selected_rows),
            ("Split Selected", self._split_selected_row),
            ("Sort", self._sort_rows),
        ]:
            b = QPushButton(label); b.clicked.connect(fn); btn_row.addWidget(b)
        btn_row.addStretch(1)
        editor_layout.addLayout(btn_row)
        layout.addWidget(editor_box, 1)

        vis_box = QGroupBox("Visualization / Save")
        vis_row = QHBoxLayout(vis_box)
        for label, fn in [
            ("Auto-color domains", self._autocolor_domains),
            ("Clear colors", self._clear_domain_colors),
            ("Save JSON", self._save_domains_json),
            ("Save TXT", self._save_domains_txt),
        ]:
            b = QPushButton(label); b.clicked.connect(fn); vis_row.addWidget(b)
        vis_row.addStretch(1)
        layout.addWidget(vis_box)

        self.tabs.addTab(tab, "Domains")

    def _build_com_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        run_box = QGroupBox("pdbcom Options")
        run_grid = QGridLayout(run_box)
        self.mass_weighted = QCheckBox("Mass-weighted COM"); self.mass_weighted.setChecked(True)
        self.atoms_mode = QComboBox(); self.atoms_mode.addItems(["all", "backbone", "CA"])
        run_btn = QPushButton("Run pdbcom")
        run_btn.clicked.connect(self._run_pdbcom)
        run_grid.addWidget(self.mass_weighted, 0, 0)
        run_grid.addWidget(QLabel("Atoms"), 0, 1)
        run_grid.addWidget(self.atoms_mode, 0, 2)
        run_grid.addWidget(run_btn, 0, 3)
        layout.addWidget(run_box)

        ref_box = QGroupBox("Reference Point")
        ref_grid = QGridLayout(ref_box)
        self.use_ref_point = QCheckBox("Use reference point")
        self.ref_point_mode = QComboBox(); self.ref_point_mode.addItems(["Domain COM", "Atom spec", "Selected atom"])
        self.ref_point_domain = QComboBox()
        self.ref_point_atom = QLineEdit("")
        ref_atom_btn = QPushButton("Use selected atom")
        ref_atom_btn.clicked.connect(lambda: self._fill_atom_lineedit(self.ref_point_atom))
        ref_grid.addWidget(self.use_ref_point, 0, 0)
        ref_grid.addWidget(self.ref_point_mode, 0, 1)
        ref_grid.addWidget(self.ref_point_domain, 0, 2)
        ref_grid.addWidget(self.ref_point_atom, 1, 1, 1, 2)
        ref_grid.addWidget(ref_atom_btn, 1, 3)
        layout.addWidget(ref_box)

        plane_box = QGroupBox("Reference Plane")
        plane_grid = QGridLayout(plane_box)
        self.use_plane = QCheckBox("Use reference plane")
        plane_grid.addWidget(self.use_plane, 0, 0)
        self.plane_modes = []
        self.plane_domains = []
        self.plane_atoms = []
        for i in range(3):
            mode = QComboBox(); mode.addItems(["Domain COM", "Atom spec", "Selected atom"])
            domain = QComboBox()
            atom = QLineEdit("")
            btn = QPushButton("Use selected atom")
            btn.clicked.connect(lambda _, le=atom: self._fill_atom_lineedit(le))
            self.plane_modes.append(mode); self.plane_domains.append(domain); self.plane_atoms.append(atom)
            plane_grid.addWidget(QLabel(f"P{i+1}"), i + 1, 0)
            plane_grid.addWidget(mode, i + 1, 1)
            plane_grid.addWidget(domain, i + 1, 2)
            plane_grid.addWidget(atom, i + 1, 3)
            plane_grid.addWidget(btn, i + 1, 4)
        layout.addWidget(plane_box)

        results_box = QGroupBox("Results")
        results_layout = QVBoxLayout(results_box)
        self.results_table = QTableWidget(0, 10)
        self.results_table.setHorizontalHeaderLabels([
            "Domain", "X", "Y", "Z", "Atoms", "Mass", "Chains", "Ref Dist", "Plane Dist", "Signed Plane Dist"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.results_table)
        row = QHBoxLayout()
        for label, fn in [
            ("Show COM PDB", self._show_current_com_model),
            ("Export CSV", self._export_results_csv),
        ]:
            b = QPushButton(label); b.clicked.connect(fn); row.addWidget(b)
        row.addStretch(1)
        results_layout.addLayout(row)
        layout.addWidget(results_box, 1)

        self.tabs.addTab(tab, "COM / Geometry")

    def _build_log_panel(self, parent_layout):
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        parent_layout.addWidget(self.log)

    def _browse_executable(self):
        path, _ = QFileDialog.getOpenFileName(self.tw.ui_area, "Select cryomodel executable")
        if path:
            self.exe_edit.setText(path)

    def _browse_workdir(self):
        path = QFileDialog.getExistingDirectory(self.tw.ui_area, "Select working directory")
        if path:
            self.workdir_edit.setText(path)

    def _refresh_structure_menu(self):
        current = self.structure_combo.currentData()
        self.structure_combo.clear()
        atomic_models = []
        for model in self.session.models.list():
            if hasattr(model, 'atoms') and hasattr(model, 'residues'):
                atomic_models.append(model)
                self.structure_combo.addItem(f"#{model.id_string}  {getattr(model, 'name', '')}", model.id_string)
        if current:
            idx = self.structure_combo.findData(current)
            if idx >= 0:
                self.structure_combo.setCurrentIndex(idx)
        self._update_chain_menu()

    def _selected_structure(self):
        mid = self.structure_combo.currentData()
        for model in self.session.models.list():
            if getattr(model, 'id_string', None) == mid:
                return model
        return None

    def _update_chain_menu(self):
        self.chain_combo.clear()
        structure = self._selected_structure()
        chains = []
        if structure is not None:
            try:
                chains = sorted({r.chain_id for r in structure.residues})
            except Exception:
                chains = []
        for ch in chains:
            self.chain_combo.addItem(ch)

    def _browse_domain_file(self):
        path, _ = QFileDialog.getOpenFileName(self.tw.ui_area, "Select domain file", "", "Domain files (*.json *.txt);;All files (*)")
        if path:
            self.domain_file_edit.setText(path)

    def _load_domain_file(self):
        path = self.domain_file_edit.text().strip()
        if not path:
            raise UserError("Choose a domain JSON or TXT file")
        self.domain_model = DomainModel.from_file(path)
        self._populate_domain_table()
        self._refresh_domain_dropdowns()
        self._log(f"Loaded domain definitions from {path}")

    def _engine(self):
        return PDBComEngine(self.exe_edit.text().strip() or "cryomodel")

    def _infer_domains(self):
        structure = self._selected_structure()
        if structure is None:
            raise UserError("Select a structure first")
        engine = self._engine()
        wd = self.workdir_edit.text().strip() or None
        out_prefix = self.prefix_edit.text().strip() or "domains_auto"
        work_dir = engine.ensure_work_dir(wd)
        model_path = engine.export_model(self.session, structure, work_dir / "input_model.pdb")
        options = {
            "merge_distance": self.merge_distance.value(),
            "seed_size": self.seed_size.value(),
            "min_domain_residues": self.min_domain_res.value(),
            "prefer_gaps": self.prefer_gaps.isChecked(),
            "gap_window": self.gap_window.value(),
            "gaps_only": self.gaps_only.isChecked(),
            "sse_source": self.sse_source.currentText(),
            "sse_window": self.sse_window.value(),
            "write_pdb": self.write_pdb.isChecked(),
        }
        n_domains = self.n_domains.value() if self.n_domains.value() > 0 else None
        result = engine.run_pdbdomain(model_path, self.chain_combo.currentText() or None, n_domains, out_prefix, work_dir, options)
        self._log_command(result)
        if not result["ok"]:
            raise UserError(result["stderr"] or "pdbdomain failed")
        self.domain_model = DomainModel.from_json_file(result["json_file"])
        self._populate_domain_table()
        self._refresh_domain_dropdowns()
        self._log_result(result)

    def _populate_domain_table(self):
        rows = self.domain_model.to_rows()
        self.domain_table.setRowCount(0)
        for row in rows:
            r = self.domain_table.rowCount()
            self.domain_table.insertRow(r)
            chk = QTableWidgetItem()
            chk.setCheckState(Qt.Checked if row.enabled else Qt.Unchecked)
            self.domain_table.setItem(r, self.COL_USE, chk)
            self.domain_table.setItem(r, self.COL_DOMAIN, QTableWidgetItem(str(row.domain)))
            self.domain_table.setItem(r, self.COL_CHAIN, QTableWidgetItem(str(row.chain)))
            self.domain_table.setItem(r, self.COL_START, QTableWidgetItem(str(row.start)))
            self.domain_table.setItem(r, self.COL_END, QTableWidgetItem(str(row.end)))
            self.domain_table.setItem(r, self.COL_COLOR, QTableWidgetItem(row.color or ""))

    def _read_table_rows(self):
        rows = []
        for r in range(self.domain_table.rowCount()):
            enabled = self.domain_table.item(r, self.COL_USE).checkState() == Qt.Checked
            rows.append({
                "enabled": enabled,
                "domain": self.domain_table.item(r, self.COL_DOMAIN).text().strip(),
                "chain": self.domain_table.item(r, self.COL_CHAIN).text().strip(),
                "start": int(self.domain_table.item(r, self.COL_START).text()),
                "end": int(self.domain_table.item(r, self.COL_END).text()),
                "color": self.domain_table.item(r, self.COL_COLOR).text().strip() or None,
            })
        return rows

    def _sync_domain_model_from_table(self):
        self.domain_model = DomainModel.from_rows(self._read_table_rows())
        self._refresh_domain_dropdowns()

    def _refresh_domain_dropdowns(self):
        names = self.domain_model.domain_names()
        for combo in [self.ref_point_domain, *self.plane_domains]:
            current = combo.currentText()
            combo.clear(); combo.addItems(names)
            idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def _add_domain_row(self):
        r = self.domain_table.rowCount()
        self.domain_table.insertRow(r)
        chk = QTableWidgetItem(); chk.setCheckState(Qt.Checked)
        self.domain_table.setItem(r, self.COL_USE, chk)
        for c, text in [(self.COL_DOMAIN, f"Domain{r+1}"), (self.COL_CHAIN, self.chain_combo.currentText() or "A"), (self.COL_START, "1"), (self.COL_END, "1"), (self.COL_COLOR, "")]:
            self.domain_table.setItem(r, c, QTableWidgetItem(text))

    def _selected_row_indices(self):
        return sorted({idx.row() for idx in self.domain_table.selectedIndexes()})

    def _delete_selected_rows(self):
        for r in reversed(self._selected_row_indices()):
            self.domain_table.removeRow(r)
        self._sync_domain_model_from_table()

    def _rename_selected_domain(self):
        rows = self._selected_row_indices()
        if not rows:
            raise UserError("Select at least one row")
        old_name = self.domain_table.item(rows[0], self.COL_DOMAIN).text().strip()
        new_name, ok = QInputDialog.getText(self.tw.ui_area, "Rename Domain", "New domain name:", text=old_name)
        if not ok or not new_name.strip():
            return
        for r in rows:
            self.domain_table.item(r, self.COL_DOMAIN).setText(new_name.strip())
        self._sync_domain_model_from_table()

    def _join_selected_rows(self):
        rows = self._selected_row_indices()
        if len(rows) < 2:
            raise UserError("Select at least two rows to join")
        new_name, ok = QInputDialog.getText(self.tw.ui_area, "Join Domains", "Joined domain name:", text=self.domain_table.item(rows[0], self.COL_DOMAIN).text())
        if not ok or not new_name.strip():
            return
        for r in rows:
            self.domain_table.item(r, self.COL_DOMAIN).setText(new_name.strip())
        self._sync_domain_model_from_table()
        self._populate_domain_table()

    def _split_selected_row(self):
        rows = self._selected_row_indices()
        if len(rows) != 1:
            raise UserError("Select exactly one row to split")
        row = rows[0]
        start = int(self.domain_table.item(row, self.COL_START).text())
        end = int(self.domain_table.item(row, self.COL_END).text())
        split, ok = QInputDialog.getInt(self.tw.ui_area, "Split Row", f"Split residue ({start+1}..{end}):", value=max(start + 1, min(end, (start + end) // 2)), min=start + 1, max=end)
        if not ok:
            return
        self._sync_domain_model_from_table()
        self.domain_model.split_row(row, split)
        self._populate_domain_table()
        self._refresh_domain_dropdowns()

    def _sort_rows(self):
        rows = self._read_table_rows()
        rows.sort(key=lambda r: (r['domain'], r['chain'], int(r['start']), int(r['end'])))
        self.domain_model = DomainModel.from_rows(rows)
        self._populate_domain_table()

    def _autocolor_domains(self):
        self._sync_domain_model_from_table()
        palette = [
            "red", "orange", "yellow", "green", "cyan", "blue", "magenta", "hotpink", "tan", "slate blue"
        ]
        specs = self.domain_model.selection_specs()
        for i, (domain, parts) in enumerate(specs.items()):
            color = palette[i % len(palette)]
            for spec in parts:
                try:
                    cxrun(self.session, f"color {spec} {color}")
                except Exception:
                    pass
        self._log("Applied domain colors")

    def _clear_domain_colors(self):
        structure = self._selected_structure()
        if structure is None:
            return
        try:
            cxrun(self.session, f"color #{structure.id_string} bychain")
        except Exception:
            pass

    def _save_domains_json(self):
        self._sync_domain_model_from_table()
        path, _ = QFileDialog.getSaveFileName(self.tw.ui_area, "Save domain JSON", "domains.json", "JSON (*.json)")
        if path:
            self.domain_model.write_json(path)
            self._log(f"Saved JSON: {path}")

    def _save_domains_txt(self):
        self._sync_domain_model_from_table()
        path, _ = QFileDialog.getSaveFileName(self.tw.ui_area, "Save domain TXT", "domains.txt", "Text (*.txt)")
        if path:
            self.domain_model.write_txt(path)
            self._log(f"Saved TXT: {path}")

    def _write_temp_domains_json(self, work_dir: Path):
        self._sync_domain_model_from_table()
        path = work_dir / "domains_current.json"
        self.domain_model.write_json(path)
        return path

    def _fill_atom_lineedit(self, lineedit):
        atoms = selected_atoms(self.session)
        if len(atoms) != 1:
            raise UserError("Select exactly one atom")
        a = atoms[0]
        lineedit.setText(f"/{a.residue.chain_id}:{a.residue.number}@{a.name}")

    def _resolve_point_spec(self, mode, domain_name, atom_spec, com_rows):
        if mode == "Domain COM":
            for row in com_rows:
                if row["domain"] == domain_name:
                    return (row["x"], row["y"], row["z"])
            raise UserError(f"Could not find COM for domain {domain_name}")
        if mode in ("Atom spec", "Selected atom"):
            spec = atom_spec.strip()
            if mode == "Selected atom" and not spec:
                atoms = selected_atoms(self.session)
                if len(atoms) != 1:
                    raise UserError("Select exactly one atom")
                a = atoms[0]
                return (a.scene_coord[0], a.scene_coord[1], a.scene_coord[2])
            atoms = self.session.selection.items('atoms') if False else None
            # Fallback: parse current selection if the lineedit was filled from selected atom.
            if spec and spec.startswith('/'):
                atoms = selected_atoms(self.session)
                if len(atoms) == 1 and spec.endswith('@' + atoms[0].name):
                    a = atoms[0]
                    return (a.scene_coord[0], a.scene_coord[1], a.scene_coord[2])
            raise UserError("Atom-spec resolution is not fully implemented yet; use 'Use selected atom' and keep that atom selected")
        raise UserError("Unsupported reference mode")

    def _run_pdbcom(self):
        structure = self._selected_structure()
        if structure is None:
            raise UserError("Select a structure first")
        self._sync_domain_model_from_table()
        if not self.domain_model.domain_names():
            raise UserError("No domain definitions loaded or entered")
        engine = self._engine()
        work_dir = engine.ensure_work_dir(self.workdir_edit.text().strip() or None)
        model_path = engine.export_model(self.session, structure, work_dir / "input_model.pdb")
        domains_path = self._write_temp_domains_json(work_dir)
        out_prefix = (self.prefix_edit.text().strip() or "domains") + "_com"
        options = {
            "mass_weighted": self.mass_weighted.isChecked(),
            "atoms": self.atoms_mode.currentText(),
        }
        result = engine.run_pdbcom(model_path, domains_path, out_prefix, work_dir, options)
        self._log_command(result)
        if not result["ok"]:
            raise UserError(result["stderr"] or "pdbcom failed")
        rows = engine.parse_com_csv(result["csv_file"])
        self._com_rows = self._compute_geometry_columns(rows)
        self._com_pdb_path = result["pdb_file"]
        self._populate_results_table(self._com_rows)
        self._log_result(result)

    def _compute_geometry_columns(self, rows):
        rows = [dict(r) for r in rows]
        ref_point = None
        if self.use_ref_point.isChecked():
            ref_point = self._resolve_point_spec(
                self.ref_point_mode.currentText(),
                self.ref_point_domain.currentText(),
                self.ref_point_atom.text(),
                rows,
            )
        plane = None
        if self.use_plane.isChecked():
            pts = []
            for mode, domain, atom in zip(self.plane_modes, self.plane_domains, self.plane_atoms):
                pts.append(self._resolve_point_spec(mode.currentText(), domain.currentText(), atom.text(), rows))
            plane = _plane_from_points(*pts)
        for row in rows:
            p = (row['x'], row['y'], row['z'])
            row['ref_dist'] = _distance(p, ref_point) if ref_point else None
            if plane:
                sd = _signed_distance_to_plane(p, plane[0], plane[1])
                row['signed_plane_dist'] = sd
                row['plane_dist'] = abs(sd)
            else:
                row['signed_plane_dist'] = None
                row['plane_dist'] = None
        return rows

    def _populate_results_table(self, rows):
        self.results_table.setRowCount(0)
        for row in rows:
            r = self.results_table.rowCount()
            self.results_table.insertRow(r)
            vals = [
                row.get('domain', ''),
                f"{row.get('x', 0.0):.3f}", f"{row.get('y', 0.0):.3f}", f"{row.get('z', 0.0):.3f}",
                str(row.get('num_atoms', '')), f"{row.get('mass', 0.0):.2f}", row.get('chains', ''),
                "" if row.get('ref_dist') is None else f"{row['ref_dist']:.3f}",
                "" if row.get('plane_dist') is None else f"{row['plane_dist']:.3f}",
                "" if row.get('signed_plane_dist') is None else f"{row['signed_plane_dist']:.3f}",
            ]
            for c, v in enumerate(vals):
                self.results_table.setItem(r, c, QTableWidgetItem(v))

    def _show_current_com_model(self):
        if not self._com_pdb_path or not Path(self._com_pdb_path).exists():
            raise UserError("No COM PDB has been generated yet")
        cxrun(self.session, f'open "{self._com_pdb_path}"')

    def _export_results_csv(self):
        if not self._com_rows:
            raise UserError("No results available")
        path, _ = QFileDialog.getSaveFileName(self.tw.ui_area, "Export results CSV", "domain_com_results.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["domain", "x", "y", "z", "num_atoms", "mass", "chains", "ref_dist", "plane_dist", "signed_plane_dist"])
            writer.writeheader()
            for row in self._com_rows:
                writer.writerow(row)
        self._log(f"Exported results CSV: {path}")

    def _log(self, text):
        self.log.appendPlainText(str(text))

    def _log_command(self, result):
        self._log("$ " + " ".join(result.get("command", [])))

    def _log_result(self, result):
        if result.get("stdout"):
            self._log(result["stdout"].rstrip())
        if result.get("stderr"):
            self._log(result["stderr"].rstrip())
