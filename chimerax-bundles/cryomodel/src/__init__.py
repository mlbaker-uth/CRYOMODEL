# CryoModel ChimeraX bundle: commands and Tools menu integration

from typing import Optional

from chimerax.core.toolshed import BundleAPI


class _CryoModelAPI(BundleAPI):
    # api_version 1: register_command(bi, ci, logger) — see ChimeraX tut_cmd __init__.py
    api_version = 1

    @staticmethod
    def register_command(bi, ci, logger):
        """Lazy register one command per ChimeraX call; ci.name matches bundle metadata."""
        from chimerax.core.commands import CmdDesc, register, StringArg, FloatArg, BoolArg

        def _norm_cmd_name(name: str) -> str:
            # ChimeraX historically treats command spellings as multi-word commands.
            # Some tooling/metadata sources use underscores; others use spaces.
            return (name or "").strip().replace("_", " ").lower()

        n = _norm_cmd_name(getattr(ci, "name", ""))

        if n == "cryomodel findligands":

            def run_findligands(session, map_path: str, model_path: str, thresh: float = 2.5):
                try:
                    from cryomodel.io.mrc import read_map
                    from cryomodel.io.pdb import read_model
                    from cryomodel.finders.pipeline import run_pipeline
                except Exception as e:
                    session.logger.error(f"Failed to import cryomodel library: {e}")
                    return
                vol = read_map(map_path)
                model = read_model(model_path)
                assigns = run_pipeline(vol, model, thresh=thresh)
                n = len(assigns.assignments)
                session.logger.info(f"[CryoModel] Found {n} candidate sites (thresh={thresh}).")

            desc = CmdDesc(
                required=[("map_path", StringArg), ("model_path", StringArg)],
                optional=[("thresh", FloatArg)],
                synopsis="Assign unmodeled density with CryoModel",
            )
            func = run_findligands

        elif n == "cryomodel pdbdomain":

            def run_pdbdomain(session, show: bool = True):
                from .pdbdomain_tool import PDBDomainTool
                tool = PDBDomainTool(session, "CryoModel Domain Tool")
                if show:
                    tool.tool_window.shown = True

            desc = CmdDesc(
                optional=[("show", BoolArg)],
                synopsis="Open CryoModel domain identification tool",
            )
            func = run_pdbdomain

        elif n == "cryomodel manifest":

            def run_write_manifest(session, output_path: Optional[str] = None):
                from pathlib import Path

                from .chimerax_manifest import build_manifest_entries, log_manifest_summary, write_manifest

                out = (
                    Path(output_path).expanduser()
                    if output_path
                    else Path.home() / "cryomodel_chimerax_manifest.json"
                )
                entries = build_manifest_entries(session)
                written = write_manifest(session, out)
                log_manifest_summary(session, entries, written)
                session.logger.info(f"[CryoModel] Wrote ChimeraX manifest ({written}).")

            desc = CmdDesc(
                optional=[("output_path", StringArg)],
                synopsis="Write CryoModel workflow manifest JSON for open models (phase 1)",
            )
            func = run_write_manifest

        else:
            raise ValueError(f"CryoModel bundle: unknown command {ci.name!r}")

        if desc.synopsis is None:
            desc.synopsis = ci.synopsis

        # Allow both underscore and space spellings as aliases.
        # Some ChimeraX setups expose multi-word command names with spaces,
        # while other metadata uses underscores.
        primary_name = getattr(ci, "name", "").strip()
        register(primary_name, desc, func)
        alt_name = primary_name.replace("_", " ") if "_" in primary_name else primary_name.replace(" ", "_")
        if alt_name and alt_name != primary_name:
            try:
                register(alt_name, desc, func)
            except Exception:
                # Ignore alias registration failures (e.g., if the alias already exists).
                pass

    @staticmethod
    def start_tool(session, bundle_info, tool_info):
        """Called when user opens a CryoModel tool from the Tools menu."""
        if tool_info.name == "BaseHunter Interactive":
            from .basehunter_tool import BaseHunterInteractiveTool

            return BaseHunterInteractiveTool(session, tool_info.name)
        if tool_info.name == "CryoModel Manifest":
            from .manifest_tool import ManifestTool

            return ManifestTool(session, tool_info.name)
        from .pdbdomain_tool import PDBDomainTool

        return PDBDomainTool(session, tool_info.name)

    @staticmethod
    def get_class(class_name):
        """For session save/restore."""
        from . import basehunter_tool, manifest_tool, pdbdomain_tool

        if class_name == "PDBDomainTool":
            return pdbdomain_tool.PDBDomainTool
        if class_name == "ManifestTool":
            return manifest_tool.ManifestTool
        if class_name == "BaseHunterInteractiveTool":
            return basehunter_tool.BaseHunterInteractiveTool
        raise ValueError(f"Unknown class: {class_name}")


bundle_api = _CryoModelAPI()
