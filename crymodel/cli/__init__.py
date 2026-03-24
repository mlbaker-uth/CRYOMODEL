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
