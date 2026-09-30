"""Run the API with uvicorn: ``python -m nfl_predictor.api``."""

from __future__ import annotations

import argparse

import uvicorn

from nfl_predictor.api import create_app
from nfl_predictor.api.settings import Settings


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the host, port, and reload flags."""
    parser = argparse.ArgumentParser(description="Serve the nfl-predictor web API and frontend.")
    parser.add_argument(
        "--host", default=None, help="Bind address (default: NFLP_HOST or 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Bind port (default: NFLP_PORT or 8000)"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Reload on code changes (development)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Build settings from the environment and start uvicorn."""
    args = _parse_args(argv)
    settings = Settings()
    host = args.host or settings.host
    port = args.port or settings.port
    if args.reload:
        uvicorn.run("nfl_predictor.api:create_app", factory=True, host=host, port=port, reload=True)
    else:
        uvicorn.run(create_app(settings), host=host, port=port)


if __name__ == "__main__":
    main()
