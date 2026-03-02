# Multi-command Typer app
import typer

app = typer.Typer(
    help="CryoModel: unified cryo-EM tools",
    no_args_is_help=True,
    add_completion=False,
)

# Import and register subcommands
from .findligands import findligands as _findligands
from .predictligands import predict as _predictligands
from .pathwalk import walk as _pathwalk, average as _pathwalk_average
from .pyhole import analyze as _pyhole
from .pyhole_plot import plot as _pyhole_plot
from .basehunter import compare as _basehunter
from .validate import validate as _validate
from .pdbcom import compute as _pdbcom
from .fitcompare import compare as _fitcompare
from .fitprep import check as _fitprep
from .loopcloud import generate as _loopcloud
from .pathwalker2 import discover as _pathwalker2
from .version import version as _version
from .foldhunter import search as _foldhunter
from .affilter import filter as _affilter
from .workflow import run as _workflow_run, validate as _workflow_validate
from .assistant import app as _assistant_app

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

app.command("findligands")(_findligands)
app.command("predictligands")(_predictligands)
app.command("pathwalk")(_pathwalk)
app.command("pathwalk-average")(_pathwalk_average)
app.command("pyhole")(_pyhole)
app.command("pyhole-plot")(_pyhole_plot)
app.command("basehunter")(_basehunter)
app.command("validate")(_validate)
app.command("pdbcom")(_pdbcom)
app.command("fitcompare")(_fitcompare)
app.command("fitprep")(_fitprep)
app.command("loopcloud")(_loopcloud)
app.command("pathwalker2")(_pathwalker2)
app.command("version")(_version)
# Register ML commands with lazy loading
app.command("train-ml")(_lazy_train_ml)
app.command("train-ensemble")(_lazy_train_ensemble)
app.command("extract-features")(_lazy_extract_features)
app.command("foldhunter")(_foldhunter)
app.command("affilter")(_affilter)
app.command("workflow")(_workflow_run)
app.command("workflow-validate")(_workflow_validate)
app.add_typer(_assistant_app, name="assistant")
