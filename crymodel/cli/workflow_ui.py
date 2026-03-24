"""CLI for serving the Workflow UI backend API."""
from __future__ import annotations

import typer

from .command_log import log_command

app = typer.Typer(no_args_is_help=True, help="Workflow UI backend API")


@app.command("serve")
@log_command("workflow-ui serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind"),
    port: int = typer.Option(8010, "--port", help="Port to bind"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Serve workflow UI run/status/log API."""
    try:
        import uvicorn
    except Exception as e:
        typer.echo(
            "uvicorn is required for workflow-ui serve.\n"
            "Install with: pip install uvicorn fastapi\n"
            f"Import error: {e}",
            err=True,
        )
        raise typer.Exit(1)

    uvicorn.run("crymodel.workflow.ui_api:app", host=host, port=port, reload=reload)

