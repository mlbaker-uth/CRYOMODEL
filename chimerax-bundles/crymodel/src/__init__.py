# Minimal command: crymodel_findligands <map_path> <model_path> [thresh]

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, FloatArg
    from chimerax.core.commands import BoolArg

    def run(session, map_path: str, model_path: str, thresh: float = 2.5):
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
        # (Optional) Later: draw spheres / table UI

    desc = CmdDesc(
        required=[("map_path", StringArg), ("model_path", StringArg)],
        optional=[("thresh", FloatArg)],
        synopsis="Assign unmodeled density with CryoModel"
    )
    register("crymodel_findligands", desc, run)

    # Domain identification / COM tool
    def run_pdbdomain(session, show: bool = True):
        from .pdbdomain_tool import PDBDomainTool
        tool = PDBDomainTool(session, "CryoModel Domain Tool")
        if show:
            tool.tool_window.shown = True

    desc_pdbdomain = CmdDesc(
        optional=[("show", BoolArg)],
        synopsis="Open CryoModel domain identification tool"
    )
    register("crymodel_pdbdomain", desc_pdbdomain, run_pdbdomain)

def register(session):
    # ChimeraX entry point
    register_command(session.logger)
