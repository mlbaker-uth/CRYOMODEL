"""CLI entrypoint for PathMeasure backend server."""
from __future__ import annotations

import typer

from .command_log import log_command

app = typer.Typer(no_args_is_help=True, help="PathMeasure web backend (2D MRC measurements)")


@app.command("serve")
@log_command("pathmeasure serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind"),
    port: int = typer.Option(8008, "--port", help="Port to bind"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Run the PathMeasure FastAPI backend."""
    try:
        import uvicorn
    except Exception as e:
        typer.echo(
            "uvicorn is required for pathmeasure serve.\n"
            "Install with: pip install uvicorn fastapi\n"
            f"Import error: {e}",
            err=True,
        )
        raise typer.Exit(1)

    uvicorn.run("crymodel.pathmeasure.api:app", host=host, port=port, reload=reload)

