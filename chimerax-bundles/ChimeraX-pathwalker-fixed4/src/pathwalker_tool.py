import tempfile
import threading
import traceback
from pathlib import Path

import numpy as np
from Qt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from chimerax.atomic import selected_atoms
from chimerax.core.commands import run as cxrun
from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow

try:
    from . import pathwalker_engine as engine
    ENGINE_IMPORT_ERROR = None
except Exception as e:
    engine = None
    ENGINE_IMPORT_ERROR = e


class PathWalkerTool(ToolInstance):
    SESSION_ENDURING = False
    help = "help:user/tools/pathwalker.html"

    def __init__(self, session, tool_name="PathWalker"):
        super().__init__(session, tool_name)
        self.display_name = "PathWalker"
        self.tw = MainToolWindow(self)
        self._trigger_handlers = []
        self._trace_model_path = None
        self.fixed_edges = []
        self.current_order = []

        parent = self.tw.ui_area
        layout = QVBoxLayout(parent)

        header = QHBoxLayout()
        self.map_combo = QComboBox()
        self.model_combo = QComboBox()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_menus)
        header.addWidget(QLabel("Map"))
        header.addWidget(self.map_combo, 1)
        header.addWidget(QLabel("Model"))
        header.addWidget(self.model_combo, 1)
        header.addWidget(refresh_btn)
        layout.addLayout(header)

        grid = QGridLayout()
        r = 0

        self.nres = QSpinBox(); self.nres.setRange(2, 50000); self.nres.setValue(500)
        self.threshold = QLineEdit("")
        self.threshold.setPlaceholderText("auto")
        self.mapwt = QDoubleSpinBox(); self.mapwt.setRange(0.0, 1000.0); self.mapwt.setDecimals(2); self.mapwt.setValue(10.0)
        self.distance_cutoff = QDoubleSpinBox(); self.distance_cutoff.setRange(1.0, 100.0); self.distance_cutoff.setDecimals(1); self.distance_cutoff.setValue(15.0)
        self.sample_points = QSpinBox(); self.sample_points.setRange(3, 200); self.sample_points.setValue(11)

        grid.addWidget(QLabel("# pseudoatoms"), r, 0); grid.addWidget(self.nres, r, 1)
        grid.addWidget(QLabel("Threshold"), r, 2); grid.addWidget(self.threshold, r, 3); r += 1
        grid.addWidget(QLabel("Map weight"), r, 0); grid.addWidget(self.mapwt, r, 1)
        grid.addWidget(QLabel("Edge cutoff (Å)"), r, 2); grid.addWidget(self.distance_cutoff, r, 3); r += 1
        grid.addWidget(QLabel("Line samples"), r, 0); grid.addWidget(self.sample_points, r, 1); r += 1

        self.lkh_exec = QLineEdit("LKH")
        pick_lkh = QPushButton("LKH…")
        pick_lkh.clicked.connect(self._pick_lkh)
        grid.addWidget(QLabel("LKH executable"), r, 0); grid.addWidget(self.lkh_exec, r, 1, 1, 2); grid.addWidget(pick_lkh, r, 3); r += 1

        self.work_dir = QLineEdit(str(Path(tempfile.gettempdir()) / "pathwalker"))
        pick_wd = QPushButton("Work dir…")
        pick_wd.clicked.connect(self._pick_workdir)
        grid.addWidget(QLabel("Working directory"), r, 0); grid.addWidget(self.work_dir, r, 1, 1, 2); grid.addWidget(pick_wd, r, 3); r += 1

        self.keep_temp = QCheckBox("Keep intermediate files")
        grid.addWidget(self.keep_temp, r, 0, 1, 2); r += 1

        layout.addLayout(grid)

        row = QHBoxLayout()
        self.seed_btn = QPushButton("Pseudoatom")
        self.seed_btn.clicked.connect(self.seed_atoms)
        self.trace_btn = QPushButton("Trace")
        self.trace_btn.clicked.connect(self.trace_backbone)
        row.addWidget(self.seed_btn)
        row.addWidget(self.trace_btn)
        layout.addLayout(row)

        row = QHBoxLayout()
        self.add_btn = QPushButton("Add atom in between")
        self.add_btn.clicked.connect(self.add_atom_between)
        self.del_btn = QPushButton("Delete atoms")
        self.del_btn.clicked.connect(self.delete_atoms)
        row.addWidget(self.add_btn)
        row.addWidget(self.del_btn)
        layout.addLayout(row)

        row = QHBoxLayout()
        self.fix_btn = QPushButton("Fix bonds")
        self.fix_btn.clicked.connect(self.fix_bond)
        self.unfix_btn = QPushButton("Unfix bonds")
        self.unfix_btn.clicked.connect(self.unfix_bond)
        row.addWidget(self.fix_btn)
        row.addWidget(self.unfix_btn)
        layout.addLayout(row)

        self.helix_btn = QPushButton("Build helix")
        self.helix_btn.clicked.connect(self.build_helix)
        layout.addWidget(self.helix_btn)

        self._refresh_menus()
        self._install_model_triggers()
        self.tw.manage(None)

    def delete(self):
        for h in self._trigger_handlers:
            try:
                h.remove()
            except Exception:
                pass
        super().delete()

    def _pick_lkh(self):
        fn, _ = QFileDialog.getOpenFileName(self.tw.ui_area, "Select LKH executable", "", "All files (*)")
        if fn:
            self.lkh_exec.setText(fn)

    def _pick_workdir(self):
        dn = QFileDialog.getExistingDirectory(self.tw.ui_area, "Select working directory")
        if dn:
            self.work_dir.setText(dn)

    def _install_model_triggers(self):
        def _try(triggers, name):
            try:
                h = triggers.add_handler(name, lambda *a, **k: self._refresh_menus())
                self._trigger_handlers.append(h)
            except Exception:
                pass
        _try(self.session.triggers, 'models changed')
        try:
            for nm in getattr(self.session.models.triggers, 'names', ()):
                if 'model' in nm:
                    _try(self.session.models.triggers, nm)
        except Exception:
            pass

    def _refresh_menus(self):
        self.map_combo.clear()
        self.model_combo.clear()
        for m in self.session.models.list():
            mid = getattr(m, 'id_string', None)
            if not mid:
                continue
            name = getattr(m, 'name', mid)
            if self._looks_like_volume(m):
                self.map_combo.addItem(f"#{mid}  {name}", mid)
            if self._looks_like_atomic(m):
                self.model_combo.addItem(f"#{mid}  {name}", mid)

    def _looks_like_volume(self, model):
        return hasattr(model, 'data') and (hasattr(model, 'matrix') or hasattr(model, 'full_matrix'))

    def _looks_like_atomic(self, model):
        return hasattr(model, 'atoms') and hasattr(model, 'residues')

    def _selected_map(self):
        idx = self.map_combo.currentIndex()
        if idx < 0:
            return None
        mid = self.map_combo.itemData(idx)
        return next((m for m in self.session.models.list() if getattr(m, 'id_string', '') == mid), None)

    def _selected_model(self):
        idx = self.model_combo.currentIndex()
        if idx < 0:
            return None
        mid = self.model_combo.itemData(idx)
        return next((m for m in self.session.models.list() if getattr(m, 'id_string', '') == mid), None)

    def _user_threshold(self):
        txt = self.threshold.text().strip()
        if txt == "":
            return None
        try:
            return float(txt)
        except ValueError:
            raise ValueError(f'Invalid threshold value: "{txt}"')

    def _trace_path(self):
        wd = Path(self.work_dir.text().strip() or tempfile.gettempdir())
        wd.mkdir(parents=True, exist_ok=True)
        return wd / "pathwalker_trace.pdb"

    def _load_trace_points(self):
        path = self._trace_path()
        if not path.exists():
            return []
        pts = []
        with open(path) as f:
            for line in f:
                if line.startswith(("ATOM  ", "HETATM")):
                    try:
                        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                        pts.append([x, y, z])
                    except Exception:
                        pass
        return np.array(pts, dtype=float) if pts else np.zeros((0, 3), dtype=float)

    def _write_trace_pdb(self, points, bonds=None, fixed_edges=None, path=None):
        if path is None:
            path = self._trace_path()
        bonds = bonds or []
        fixed_edges = fixed_edges or []
        with open(path, 'w') as f:
            for i, p in enumerate(points, start=1):
                f.write(
                    f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
                    f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00 20.00           C\n"
                )
            for i, j in bonds:
                f.write(f"CONECT{i+1:5d}{j+1:5d}\n")
            f.write("END\n")
        self.fixed_edges = [tuple(sorted(x)) for x in fixed_edges]
        self.current_order = list(range(len(points)))
        self._trace_model_path = str(path)
        return path

    def _reopen_trace(self):
        path = self._trace_path()
        if not path.exists():
            return
        stem = path.stem
        for m in list(self.session.models.list()):
            try:
                if getattr(m, 'name', '') == path.name or stem in getattr(m, 'name', ''):
                    self.session.models.close([m])
            except Exception:
                pass
        cxrun(self.session, f'open "{path}"')
        self._refresh_menus()

    def _selected_trace_indices(self):
        atoms = selected_atoms(self.session)
        out = []
        for a in atoms:
            try:
                out.append(int(a.residue.number) - 1)
            except Exception:
                pass
        return sorted(set([i for i in out if i >= 0]))

    def seed_atoms(self):
        if engine is None:
            self.session.logger.error(f"pathwalker_engine could not be imported: {ENGINE_IMPORT_ERROR!r}")
            return
        vol = self._selected_map()
        if vol is None:
            self.session.logger.error("Select a map first.")
            return
        nres = int(self.nres.value())
        try:
            threshold = self._user_threshold()
            pts = engine.seed_from_volume(vol, nres, threshold=threshold)
        except Exception:
            self.session.logger.error("PathWalker pseudoatom seeding failed:\n" + traceback.format_exc())
            return
        self._write_trace_pdb(pts)
        self._reopen_trace()
        used = "auto threshold" if threshold is None else f"threshold {threshold:g}"
        self.session.logger.status(f"PathWalker: seeded {len(pts)} pseudoatoms using {used}.")

    def trace_backbone(self):
        if engine is None:
            self.session.logger.error(f"pathwalker_engine could not be imported: {ENGINE_IMPORT_ERROR!r}")
            return
        vol = self._selected_map()
        pts = self._load_trace_points()
        if len(pts) < 2:
            self.session.logger.error("Need a trace model with at least two pseudoatoms.")
            return
        params = dict(
            points=np.asarray(pts, dtype=float),
            fixed_edges=list(self.fixed_edges),
            map_model=vol,
            map_weight=float(self.mapwt.value()),
            distance_cutoff=float(self.distance_cutoff.value()),
            sample_points=int(self.sample_points.value()),
            lkh_executable=self.lkh_exec.text().strip() or "LKH",
            work_dir=self.work_dir.text().strip() or tempfile.gettempdir(),
            keep_temp=self.keep_temp.isChecked(),
        )

        self.trace_btn.setEnabled(False)
        self.trace_btn.setText("Tracing…")

        def worker():
            try:
                result = engine.trace_path(**params)
            except Exception:
                msg = traceback.format_exc()
                self.session.ui.thread_safe(lambda: self._trace_failed(msg))
                return
            self.session.ui.thread_safe(lambda r=result: self._trace_finished(r))

        threading.Thread(target=worker, daemon=True).start()

    def _trace_failed(self, msg):
        self.trace_btn.setEnabled(True)
        self.trace_btn.setText("Trace")
        self.session.logger.error("PathWalker trace failed:\n" + msg)

    def _trace_finished(self, result):
        self.trace_btn.setEnabled(True)
        self.trace_btn.setText("Trace")
        order = result["order"]
        points = result["ordered_points"]
        bonds = [(i, i + 1) for i in range(len(points) - 1)]
        fixed = [tuple(sorted((edge[0], edge[1]))) for edge in result.get("fixed_edges_reindexed", [])]
        self.current_order = list(order)
        self._write_trace_pdb(points, bonds=bonds, fixed_edges=fixed)
        self._reopen_trace()
        self.session.logger.status(f"PathWalker: traced {len(points)} nodes.")

    def add_atom_between(self):
        pts = self._load_trace_points()
        sel = self._selected_trace_indices()
        if len(sel) != 2:
            self.session.logger.warning("Select exactly two trace atoms.")
            return
        pos = pts[sel].mean(axis=0)
        insert_at = min(sel) + 1
        pts = np.insert(pts, insert_at, pos, axis=0)
        self.fixed_edges = engine.remap_fixed_edges_after_insert(self.fixed_edges, insert_at) if engine else self.fixed_edges
        bonds = [(i, i + 1) for i in range(len(pts) - 1)]
        self._write_trace_pdb(pts, bonds=bonds, fixed_edges=self.fixed_edges)
        self._reopen_trace()

    def delete_atoms(self):
        pts = self._load_trace_points()
        sel = self._selected_trace_indices()
        if not sel:
            self.session.logger.warning("Select one or more trace atoms to delete.")
            return
        keep = [i for i in range(len(pts)) if i not in set(sel)]
        if len(keep) < 2:
            self.session.logger.warning("Cannot delete all trace atoms.")
            return
        new_pts = pts[keep]
        if engine:
            self.fixed_edges = engine.remap_fixed_edges_after_delete(self.fixed_edges, sel)
        bonds = [(i, i + 1) for i in range(len(new_pts) - 1)]
        self._write_trace_pdb(new_pts, bonds=bonds, fixed_edges=self.fixed_edges)
        self._reopen_trace()

    def fix_bond(self):
        sel = self._selected_trace_indices()
        if len(sel) != 2:
            self.session.logger.warning("Select exactly two neighboring trace atoms.")
            return
        edge = tuple(sorted(sel))
        if edge not in self.fixed_edges:
            self.fixed_edges.append(edge)
        self.session.logger.status(f"PathWalker: fixed edge {edge[0] + 1}-{edge[1] + 1}.")

    def unfix_bond(self):
        sel = self._selected_trace_indices()
        if len(sel) == 2:
            edge = tuple(sorted(sel))
            self.fixed_edges = [e for e in self.fixed_edges if tuple(sorted(e)) != edge]
            self.session.logger.status(f"PathWalker: unfixed edge {edge[0] + 1}-{edge[1] + 1}.")
        else:
            self.fixed_edges = []
            self.session.logger.status("PathWalker: cleared all fixed edges.")

    def build_helix(self):
        if engine is None:
            self.session.logger.error(f"pathwalker_engine could not be imported: {ENGINE_IMPORT_ERROR!r}")
            return
        pts = self._load_trace_points()
        sel = self._selected_trace_indices()
        if len(sel) != 2:
            self.session.logger.warning("Select the two end atoms of the segment to replace with a helix.")
            return
        i0, i1 = min(sel), max(sel)
        if i1 <= i0:
            return
        try:
            new_pts, new_fixed = engine.replace_segment_with_helix(pts, i0, i1, self.fixed_edges)
        except Exception:
            self.session.logger.error("Helix build failed:\n" + traceback.format_exc())
            return
        bonds = [(i, i + 1) for i in range(len(new_pts) - 1)]
        self.fixed_edges = new_fixed
        self._write_trace_pdb(new_pts, bonds=bonds, fixed_edges=self.fixed_edges)
        self._reopen_trace()
