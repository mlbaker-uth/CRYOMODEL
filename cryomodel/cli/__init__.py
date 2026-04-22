# Multi-command Typer app
import typer

app = typer.Typer(
    help="CryoModel: unified cryo-EM tools",
    no_args_is_help=True,
    add_completion=False,
)

# Command logging
from .command_log import log_command

# Import and register subcommands
from .findligands import findligands as _findligands
from .predictligands import predict as _predictligands
from .pathwalk import walk as _pathwalk, average as _pathwalk_average
from .pyhole import analyze as _pyhole
from .pyhole_plot import plot as _pyhole_plot
from .basehunter import compare as _basehunter
from .validate import validate as _validate
from .pdbcom import compute as _pdbcom
from .pdbdomain import identify as _pdbdomain
from .fitcompare import compare as _fitcompare
from .fitprep import check as _fitprep
from .loopcloud import generate as _loopcloud
from .pathwalker2 import discover as _pathwalker2
from .version import version as _version
from .foldhunter import search as _foldhunter
from .affilter import filter as _affilter
from .workflow import run as _workflow_run, validate as _workflow_validate
from .assistant import app as _assistant_app
from .dnabuild import app as _dnabuild_app
from .dnaaxis import app as _dnaaxis_app
from .logs import app as _logs_app
from .mapfilter import app as _mapfilter_app
from .pathmeasure import app as _pathmeasure_app
from .workflow_ui import app as _workflow_ui_app
from .manager import manager_app as _manager_app
from .chimerax_bridge import app as _chimerax_bridge_app, manifest_help as _chimerax_manifest_help
from .pdb2mrc import convert as _pdb2mrc_convert
from .pdb_mutate import app as _pdb_mutate_app
from .seqconservation import seqconservation as _seqconservation, seqconservation_diffuse as _seqconservation_diffuse
from .fasta_extract import app as _fasta_extract_app
from .zonal_refine import app as _zonal_refine_app
from .debug_report import generate as _debug_report_generate
from .symmetry import app as _symmetry_app
from .helical import app as _helical_app

# Lazy imports for ML commands (only import when actually called)
# This avoids PyTorch import issues when using non-ML commands
def _lazy_train_ml(*args, **kwargs):
    from .train_ml import train
    return train(*args, **kwargs)

def _lazy_train_ensemble(*args, **kwargs):
    from .train_ensemble import train
    return train(*args, **kwargs)

def _lazy_extract_features(*args, **kwargs):
    from .extract_features import extract
    return extract(*args, **kwargs)

def _register(name: str, func):
    app.command(name)(log_command(name)(func))


_register("findligands", _findligands)
_register("predictligands", _predictligands)
_register("pathwalker", _pathwalk)
_register("pathwalker-average", _pathwalk_average)
# Backward compatibility (prefer `pathwalker` / `pathwalker-average` in docs and UI).
_register("pathwalk", _pathwalk)
_register("pathwalk-average", _pathwalk_average)
_register("pyhole", _pyhole)
_register("pyhole-plot", _pyhole_plot)
_register("basehunter", _basehunter)
_register("validate", _validate)
_register("pdbcom", _pdbcom)
_register("pdbdomain", _pdbdomain)
_register("fitcompare", _fitcompare)
_register("fitprep", _fitprep)
_register("loopcloud", _loopcloud)
_register("pathwalker2", _pathwalker2)
_register("version", _version)
# Register ML commands with lazy loading
_register("train-ml", _lazy_train_ml)
_register("train-ensemble", _lazy_train_ensemble)
_register("extract-features", _lazy_extract_features)
_register("foldhunter", _foldhunter)
_register("affilter", _affilter)
_register("workflow", _workflow_run)
_register("workflow-validate", _workflow_validate)
app.add_typer(_assistant_app, name="assistant")
app.add_typer(_dnabuild_app, name="dnabuild")
app.add_typer(_dnaaxis_app, name="dnaaxis")
app.add_typer(_logs_app, name="log")
app.add_typer(_mapfilter_app, name="mapfilter")
app.add_typer(_pathmeasure_app, name="pathmeasure")
app.add_typer(_workflow_ui_app, name="workflow-ui")
app.add_typer(_pdb_mutate_app, name="pdb-mutate")
app.add_typer(_fasta_extract_app, name="fasta-extract")
app.add_typer(_zonal_refine_app, name="zonal-refine")
app.add_typer(_symmetry_app, name="symmetry")
app.add_typer(_helical_app, name="helical")
app.add_typer(_manager_app, name="manager")
app.add_typer(_chimerax_bridge_app, name="chimerax")
_register("chimerax-manifest", _chimerax_manifest_help)
_register("model2map", _pdb2mrc_convert)
# Backward compatibility alias (prefer `model2map` in docs/UI).
_register("pdb2mrc", _pdb2mrc_convert)
_register("seqconservation", _seqconservation)
_register("seqconservation-diffuse", _seqconservation_diffuse)
_register("debug-report", _debug_report_generate)
