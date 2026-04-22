"""Terminal-side hints for ChimeraX integration (manifest is written from inside ChimeraX)."""
from __future__ import annotations

import typer

app = typer.Typer(
    name="chimerax",
    help="ChimeraX bridge: manifest and workflow UI integration.",
    no_args_is_help=True,
)


@app.command("manifest")
def manifest_help() -> None:
    """Explain how to write manifest.json (must run inside ChimeraX, not in this terminal)."""
    typer.echo(
        "The manifest writer runs inside ChimeraX, not in the shell.\n\n"
        "In ChimeraX:\n"
        "  • Open the Command Line (menu: Tools, or shortcut such as F2 — depends on version).\n"
        "  • Run exactly:\n"
        "       cryomodel_manifest\n"
        "    or with a path:\n"
        "       cryomodel_manifest /path/to/manifest.json\n\n"
        "Default file: ~/cryomodel_chimerax_manifest.json\n\n"
        "If ChimeraX says the command is unknown:\n"
        "  • Install/update the CryoModel bundle: devel install <path-to-chimerax-bundles/cryomodel>\n"
        "  • Restart ChimeraX, then try again.\n\n"
        "Then use the workflow UI “Load manifest” with that path.\n"
    )

