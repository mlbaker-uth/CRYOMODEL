"""ChimeraX widget for writing and previewing CryoModel manifests."""
from __future__ import annotations

from pathlib import Path

from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from Qt.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from .chimerax_manifest import build_manifest_entries, log_manifest_summary, write_manifest


class ManifestTool(ToolInstance):
    """Simple widget to write manifest and show what was captured."""

    SESSION_ENDURING = True

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        self.tool_window = MainToolWindow(self)
        self.tool_window.fill_context_menu = True
        self._build_ui()
        self.tool_window.manage(None)

    def _build_ui(self):
        container = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Manifest output path"))
        row = QHBoxLayout()
        self.output_path = QLineEdit()
        self.output_path.setText(str(Path.home() / "cryomodel_chimerax_manifest.json"))
        browse = QPushButton("Browse")
        browse.clicked.connect(self._choose_output_path)
        row.addWidget(self.output_path)
        row.addWidget(browse)
        layout.addLayout(row)

        write_btn = QPushButton("Write Manifest + Log Open Models")
        write_btn.clicked.connect(self._write_manifest_clicked)
        layout.addWidget(write_btn)

        container.setLayout(layout)
        self.tool_window.ui_area.setLayout(layout)
        self.tool_window.ui_area.layout().addWidget(container)

    def _choose_output_path(self):
        default_path = self.output_path.text().strip() or str(Path.home() / "cryomodel_chimerax_manifest.json")
        path, _ = QFileDialog.getSaveFileName(
            self.tool_window.ui_area,
            "Save CryoModel Manifest",
            default_path,
            "JSON files (*.json);;All files (*)",
        )
        if path:
            self.output_path.setText(path)

    def _write_manifest_clicked(self):
        raw = self.output_path.text().strip()
        out = Path(raw).expanduser() if raw else (Path.home() / "cryomodel_chimerax_manifest.json")
        entries = build_manifest_entries(self.session)
        written = write_manifest(self.session, out)
        log_manifest_summary(self.session, entries, written)
        self.session.logger.info(
            "[CryoModel Manifest] Manifest written. Use this path in workflow UI -> Load manifest."
        )
