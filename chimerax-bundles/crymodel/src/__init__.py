# CryoModel ChimeraX bundle: commands and Tools menu integration

from chimerax.core.toolshed import BundleAPI


class _CryoModelAPI(BundleAPI):
    api_version = 1  # start_tool(session, bi, ti)

    @staticmethod
    def register_command(session, logger):
        """Register crymodel_findligands and crymodel_pdbdomain commands."""
        from chimerax.core.commands import CmdDesc, register, StringArg, FloatArg, BoolArg

        # crymodel_findligands
        def run_findligands(session, map_path: str, model_path: str, thresh: float = 2.5):
            try:
                from crymodel.io.mrc import read_map
                from crymodel.io.pdb import read_model
                from crymodel.finders.pipeline import run_pipeline
            except Exception as e:
                session.logger.error(f"Failed to import crymodel library: {e}")
                return
            vol = read_map(map_path)
            model = read_model(model_path)
            assigns = run_pipeline(vol, model, thresh=thresh)
            n = len(assigns.assignments)
            session.logger.info(f"[CryoModel] Found {n} candidate sites (thresh={thresh}).")

        register(
            "crymodel_findligands",
            CmdDesc(
                required=[("map_path", StringArg), ("model_path", StringArg)],
                optional=[("thresh", FloatArg)],
                synopsis="Assign unmodeled density with CryoModel",
            ),
            run_findligands,
        )

        # crymodel_pdbdomain (command still works; start_tool is for Tools menu)
        def run_pdbdomain(session, show: bool = True):
            from .pdbdomain_tool import PDBDomainTool
            tool = PDBDomainTool(session, "CryoModel Domain Tool")
            if show:
                tool.tool_window.shown = True

        register(
            "crymodel_pdbdomain",
            CmdDesc(
                optional=[("show", BoolArg)],
                synopsis="Open CryoModel domain identification tool",
            ),
            run_pdbdomain,
        )

    @staticmethod
    def start_tool(session, bundle_info, tool_info):
        """Called when user opens the tool from Tools > Structural Analysis > CryoModel Domain Tool."""
        from .pdbdomain_tool import PDBDomainTool
        return PDBDomainTool(session, tool_info.name)

    @staticmethod
    def get_class(class_name):
        """For session save/restore."""
        from . import pdbdomain_tool
        if class_name == "PDBDomainTool":
            return pdbdomain_tool.PDBDomainTool
        raise ValueError(f"Unknown class: {class_name}")


bundle_api = _CryoModelAPI()
