import typer
import importlib.metadata as im

def version():
    v = im.version("cryomodel")
    typer.echo(f"cryomodel {v}")
