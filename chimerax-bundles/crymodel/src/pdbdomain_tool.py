"""ChimeraX UI for PDB domain identification and COM analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import gemmi

from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from Qt.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QGroupBox,
)

from crymodel.domains.domain_identifier import (
    identify_domains,
    parse_sse_from_pdb_header,
    parse_sse_with_dssp,
    write_domain_spec,
    write_domain_csv,
    write_domain_pdb,
)
from crymodel.domains.pdbcom import parse_domain_spec, compute_domain_coms, write_domain_com_csv, write_domain_com_pdb


class PDBDomainTool(ToolInstance):
    """UI tool for domain identification and COM analysis."""

    SESSION_ENDURING = True

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        self._plane_points = None

        self.tool_window = MainToolWindow(self)
        self.tool_window.fill_context_menu = True
        self._build_ui()
        self.tool_window.manage(None)

    def _build_ui(self):
        widget = QWidget()
        layout = QVBoxLayout()

        # Model file
        layout.addWidget(QLabel("Model (PDB/mmCIF)"))
        model_row = QHBoxLayout()
        self.model_path = QLineEdit()
        model_browse = QPushButton("Browse")
        model_browse.clicked.connect(lambda: self._browse_file(self.model_path, "PDB/mmCIF (*.pdb *.cif *.mmcif)"))
        model_row.addWidget(self.model_path)
        model_row.addWidget(model_browse)
        layout.addLayout(model_row)

        # Domain file
        layout.addWidget(QLabel("Domain file (JSON/CSV/Text)"))
        domain_row = QHBoxLayout()
        self.domain_path = QLineEdit()
        domain_browse = QPushButton("Browse")
        domain_browse.clicked.connect(lambda: self._browse_file(self.domain_path, "Domain files (*.json *.csv *.txt)"))
        domain_row.addWidget(self.domain_path)
        domain_row.addWidget(domain_browse)
        layout.addLayout(domain_row)

        # Auto-identify group
        auto_group = QGroupBox("Auto identify domains")
        auto_layout = QVBoxLayout()
        self.auto_identify = QCheckBox("Enable auto domain identification")
        auto_layout.addWidget(self.auto_identify)

        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("Chain"))
        self.chain_id = QLineEdit()
        self.chain_id.setText("A")
        params_row.addWidget(self.chain_id)
        params_row.addWidget(QLabel("n-domains"))
        self.n_domains = QSpinBox()
        self.n_domains.setMinimum(0)
        self.n_domains.setValue(0)
        self.n_domains.setToolTip("0 = auto")
        params_row.addWidget(self.n_domains)
        auto_layout.addLayout(params_row)

        params_row2 = QHBoxLayout()
        params_row2.addWidget(QLabel("Merge Å"))
        self.merge_distance = QDoubleSpinBox()
        self.merge_distance.setRange(1.0, 100.0)
        self.merge_distance.setValue(25.0)
        params_row2.addWidget(self.merge_distance)
        params_row2.addWidget(QLabel("Seed size"))
        self.seed_size = QSpinBox()
        self.seed_size.setRange(2, 200)
        self.seed_size.setValue(20)
        params_row2.addWidget(self.seed_size)
        params_row2.addWidget(QLabel("Min domain"))
        self.min_domain_residues = QSpinBox()
        self.min_domain_residues.setRange(0, 1000)
        self.min_domain_residues.setValue(50)
        params_row2.addWidget(self.min_domain_residues)
        auto_layout.addLayout(params_row2)

        params_row3 = QHBoxLayout()
        self.prefer_gaps = QCheckBox("Prefer gaps")
        self.prefer_gaps.setChecked(True)
        self.gaps_only = QCheckBox("Gaps only")
        self.gap_window = QSpinBox()
        self.gap_window.setRange(0, 50)
        self.gap_window.setValue(10)
        params_row3.addWidget(self.prefer_gaps)
        params_row3.addWidget(self.gaps_only)
        params_row3.addWidget(QLabel("Gap window"))
        params_row3.addWidget(self.gap_window)
        auto_layout.addLayout(params_row3)

        params_row4 = QHBoxLayout()
        params_row4.addWidget(QLabel("SSE source"))
        self.sse_source = QComboBox()
        self.sse_source.addItems(["header", "dssp", "auto", "none"])
        params_row4.addWidget(self.sse_source)
        params_row4.addWidget(QLabel("SSE window"))
        self.sse_window = QSpinBox()
        self.sse_window.setRange(0, 50)
        self.sse_window.setValue(10)
        params_row4.addWidget(self.sse_window)
        auto_layout.addLayout(params_row4)

        auto_group.setLayout(auto_layout)
        layout.addWidget(auto_group)

        # Output prefix
        layout.addWidget(QLabel("Output prefix"))
        self.out_prefix = QLineEdit()
        self.out_prefix.setText("domains")
        layout.addWidget(self.out_prefix)

        # Actions
        action_row = QHBoxLayout()
        run_btn = QPushButton("Run domain + COM")
        run_btn.clicked.connect(self._run_domains)
        action_row.addWidget(run_btn)

        plane_btn = QPushButton("Use selection as plane")
        plane_btn.clicked.connect(self._use_selection_as_plane)
        action_row.addWidget(plane_btn)

        analyze_btn = QPushButton("Analyze COMs")
        analyze_btn.clicked.connect(self._analyze_coms)
        action_row.addWidget(analyze_btn)

        layout.addLayout(action_row)

        widget.setLayout(layout)
        self.tool_window.ui_area.setLayout(layout)
        self.tool_window.ui_area.layout().addWidget(widget)

    def _browse_file(self, line_edit: QLineEdit, filt: str):
        path, _ = QFileDialog.getOpenFileName(self.tool_window.ui_area, "Select file", "", filt)
        if path:
            line_edit.setText(path)

    def _log(self, message: str):
        self.session.logger.info(f"[CryoModel] {message}")

    def _run_domains(self):
        model_path = Path(self.model_path.text()).expanduser()
        if not model_path.exists():
            self._log("Model file not found.")
            return

        out_prefix = Path(self.out_prefix.text()).expanduser()
        structure = gemmi.read_structure(str(model_path))
        chain_id = self.chain_id.text().strip() or (structure[0][0].name if len(structure[0]) > 0 else "")
        if not chain_id:
            self._log("No chain found.")
            return

        if self.auto_identify.isChecked():
            n_domains = self.n_domains.value() or None
            sse_resnums = None
            source = self.sse_source.currentText()
            if source == "header" and model_path.suffix.lower() == ".pdb":
                sse_resnums = parse_sse_from_pdb_header(model_path, chain_id)
            elif source == "dssp":
                sse_resnums = parse_sse_with_dssp(model_path, chain_id)
            elif source == "auto":
                if model_path.suffix.lower() == ".pdb":
                    sse_resnums = parse_sse_from_pdb_header(model_path, chain_id)
                if not sse_resnums:
                    sse_resnums = parse_sse_with_dssp(model_path, chain_id)

            records, ranges_by_domain = identify_domains(
                structure=structure,
                chain_id=chain_id,
                seed_size=self.seed_size.value(),
                n_domains=n_domains,
                merge_distance=self.merge_distance.value(),
                min_domain_residues=self.min_domain_residues.value(),
                prefer_gaps=self.prefer_gaps.isChecked(),
                gap_window=self.gap_window.value(),
                gaps_only=self.gaps_only.isChecked(),
                sse_resnums=sse_resnums,
                sse_window=self.sse_window.value(),
            )
            write_domain_spec(out_prefix.with_suffix(".json"), chain_id, ranges_by_domain)
            write_domain_csv(out_prefix.with_suffix(".csv"), records, ranges_by_domain)
            write_domain_pdb(structure, chain_id, ranges_by_domain, out_prefix.with_suffix(".pdb"))
            domain_spec = {k: {chain_id: [f"{s}-{e}" if s != e else f"{s}" for s, e in ranges]} for k, ranges in ranges_by_domain.items()}
        else:
            domain_file = Path(self.domain_path.text()).expanduser()
            if not domain_file.exists():
                self._log("Domain file not found.")
                return
            domain_spec = _load_domain_file(domain_file)

        domain_coms = compute_domain_coms(structure, domain_spec, mass_weighted=True, atom_filter="all")
        com_prefix = out_prefix.with_name(out_prefix.name + "_com")
        write_domain_com_pdb(domain_coms, com_prefix.with_suffix(".pdb"))
        write_domain_com_csv(domain_coms, com_prefix.with_suffix(".csv"))
        self._display_coms(domain_coms, name="Domain COMs")
        self._log(f"Wrote: {com_prefix.with_suffix('.pdb')}")
        self._log(f"Wrote: {com_prefix.with_suffix('.csv')}")

    def _display_coms(self, domain_coms: Dict[str, Dict], name: str):
        from chimerax.atomic import AtomicStructure, Element
        from chimerax.core.colors import random_color

        model = AtomicStructure(self.session, name=name)
        for idx, (domain_name, data) in enumerate(domain_coms.items(), start=1):
            chain_id = chr(ord("A") + ((idx - 1) % 26))
            residue = model.new_residue("DOM", chain_id, idx)
            atom = model.new_atom("C", Element.get_element("C"))
            atom.coord = data["com"]
            atom.radius = 1.0
            residue.add_atom(atom)
            atom.color = random_color()
        self.session.models.add([model])
        try:
            from chimerax.core.commands import run
            run(self.session, f"style {model.id_string} sphere")
        except Exception:
            pass

    def _use_selection_as_plane(self):
        from chimerax.atomic import selected_atoms, selected_residues
        atoms = list(selected_atoms(self.session))
        if len(atoms) < 3:
            residues = list(selected_residues(self.session))
            atoms = []
            for residue in residues:
                if "CA" in residue:
                    atoms.append(residue["CA"][0])
        if len(atoms) < 3:
            self._log("Select at least 3 atoms or residues (CA used) to define a plane.")
            return
        points = np.stack([atoms[i].coord for i in range(3)], axis=0)
        self._plane_points = points
        self._log("Reference plane set from selection.")

    def _analyze_coms(self):
        if self._plane_points is None:
            self._log("No reference plane set. Use 'Use selection as plane' first.")
            return
        model_path = Path(self.model_path.text()).expanduser()
        if not model_path.exists():
            self._log("Model file not found.")
            return
        out_prefix = Path(self.out_prefix.text()).expanduser()
        com_path = out_prefix.with_name(out_prefix.name + "_com").with_suffix(".pdb")
        if not com_path.exists():
            self._log("COM PDB not found. Run domain + COM first.")
            return

        coms = _read_com_pdb(com_path)
        if not coms:
            self._log("No COMs to analyze.")
            return

        p1, p2, p3 = self._plane_points
        normal = np.cross(p2 - p1, p3 - p1)
        if np.linalg.norm(normal) < 1e-6:
            self._log("Invalid plane selection.")
            return
        normal = normal / np.linalg.norm(normal)

        rows = []
        domains = list(coms.keys())
        for i, d1 in enumerate(domains):
            for j, d2 in enumerate(domains):
                if j <= i:
                    continue
                v = coms[d2] - coms[d1]
                dist = float(np.linalg.norm(v))
                if dist < 1e-6:
                    angle = 0.0
                else:
                    angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(v / dist, normal)), -1.0, 1.0))))
                rows.append((d1, d2, dist, angle))

        # Distances to plane
        plane_rows = []
        for d in domains:
            v = coms[d] - p1
            signed = float(np.dot(v, normal))
            plane_rows.append((d, signed, abs(signed)))

        out_csv = out_prefix.with_name(out_prefix.name + "_com_analysis.csv")
        with out_csv.open("w") as out:
            out.write("domain_a,domain_b,distance,angle_to_plane_deg\n")
            for row in rows:
                out.write(f"{row[0]},{row[1]},{row[2]:.3f},{row[3]:.2f}\n")
            out.write("\n")
            out.write("domain,signed_distance_to_plane,abs_distance_to_plane\n")
            for row in plane_rows:
                out.write(f"{row[0]},{row[1]:.3f},{row[2]:.3f}\n")
        self._log(f"Wrote: {out_csv}")
        self._log("COM analysis complete.")


def _load_domain_file(path: Path) -> Dict[str, Dict[str, List[str]]]:
    if path.suffix.lower() == ".json":
        return parse_domain_spec(path)
    if path.suffix.lower() == ".csv":
        return _domain_spec_from_csv(path)
    return _domain_spec_from_text(path)


def _domain_spec_from_csv(path: Path) -> Dict[str, Dict[str, List[str]]]:
    lines = path.read_text().splitlines()
    header = lines[0].split(",") if lines else []
    if "domain" not in header or "chain" not in header or "resnum" not in header:
        raise ValueError("CSV must include chain,resnum,domain columns.")
    idx_chain = header.index("chain")
    idx_res = header.index("resnum")
    idx_domain = header.index("domain")
    entries: Dict[str, Dict[str, List[int]]] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        chain = parts[idx_chain]
        resnum = int(parts[idx_res])
        domain = parts[idx_domain]
        entries.setdefault(domain, {}).setdefault(chain, []).append(resnum)
    return _ranges_from_resnums(entries)


def _domain_spec_from_text(path: Path) -> Dict[str, Dict[str, List[str]]]:
    entries: Dict[str, Dict[str, List[int]]] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "," in line:
            parts = [p.strip() for p in line.split(",")]
        else:
            parts = [p.strip() for p in line.split()]
        if len(parts) < 3:
            continue
        chain, domain, range_str = parts[0], parts[1], parts[2]
        start, end = _parse_range(range_str)
        for resnum in range(start, end + 1):
            entries.setdefault(domain, {}).setdefault(chain, []).append(resnum)
    return _ranges_from_resnums(entries)


def _parse_range(range_str: str) -> Tuple[int, int]:
    if "-" in range_str:
        start, end = range_str.split("-")
        return int(start.strip()), int(end.strip())
    value = int(range_str.strip())
    return value, value


def _ranges_from_resnums(entries: Dict[str, Dict[str, List[int]]]) -> Dict[str, Dict[str, List[str]]]:
    output: Dict[str, Dict[str, List[str]]] = {}
    for domain, chain_map in entries.items():
        output[domain] = {}
        for chain, resnums in chain_map.items():
            resnums = sorted(set(resnums))
            ranges: List[Tuple[int, int]] = []
            start = prev = resnums[0]
            for r in resnums[1:]:
                if r == prev + 1:
                    prev = r
                    continue
                ranges.append((start, prev))
                start = prev = r
            ranges.append((start, prev))
            output[domain][chain] = [f"{s}-{e}" if s != e else f"{s}" for s, e in ranges]
    return output


def _read_com_pdb(path: Path) -> Dict[str, np.ndarray]:
    coms: Dict[str, np.ndarray] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("HETATM"):
            continue
        chain = line[21].strip() or "X"
        try:
            resnum = int(line[22:26].strip())
        except ValueError:
            resnum = 0
        name = f"{chain}{resnum}"
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        coms[name] = np.array([x, y, z], dtype=np.float32)
    return coms
