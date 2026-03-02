import typer
import importlib.metadata as im

def version():
    v = im.version("crymodel")
    typer.echo(f"crymodel {v}")
