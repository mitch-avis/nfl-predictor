"""Runtime settings for the web API.

Every value can be overridden with an ``NFLP_``-prefixed environment variable, for example
``NFLP_ROOT_DIR`` or ``NFLP_PORT``. Paths that are left unset resolve relative to ``root_dir`` so a
single variable points the API at a different checkout or data tree.
"""

from __future__ import annotations

import secrets
import sys
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from nfl_predictor import constants

SIGNING_KEY_FILENAME = "secret.key"
COOKIE_NAME = "nflp_session"
CSRF_HEADER_NAME = "x-requested-with"
CSRF_HEADER_VALUE = "nflp"


class Settings(BaseSettings):
    """Configuration for the API process.

    Attributes:
        root_dir: Repository root; defaults to the checkout that owns this package.
        data_dir: Directory holding ETL outputs (defaults to ``root_dir/data``).
        models_dir: Directory holding run directories (defaults to ``root_dir/models``).
        reports_dir: Directory holding ad-hoc reports (defaults to ``root_dir/reports``).
        state_dir: Directory for the API's own state such as the SQLite database
            (defaults to ``data_dir/web``).
        db_path: SQLite database path (defaults to ``state_dir/app.db``).
        jwt_secret: Secret used to sign session tokens; generated into ``state_dir`` when unset.
        cookie_secure: Whether the session cookie carries the ``Secure`` flag.
        session_hours: Session token lifetime in hours.
        sos_data_dir: Parquet outputs of the sibling ``nfl-sos-ratings`` project.
        web_dist: Built frontend directory (defaults to ``root_dir/web/dist``).
        python_executable: Interpreter used to launch jobs (defaults to ``root_dir/.venv``).
        host: Bind address for ``python -m nfl_predictor.api``.
        port: Bind port for ``python -m nfl_predictor.api``.

    """

    model_config = SettingsConfigDict(env_prefix="NFLP_", extra="ignore")

    root_dir: Path = Field(default_factory=lambda: Path(constants.ROOT_DIR))
    data_dir: Path | None = None
    models_dir: Path | None = None
    reports_dir: Path | None = None
    state_dir: Path | None = None
    db_path: Path | None = None
    jwt_secret: str | None = None
    cookie_secure: bool = False
    session_hours: int = 24 * 7
    sos_data_dir: Path | None = None
    web_dist: Path | None = None
    python_executable: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8000

    def model_post_init(self, __context: object) -> None:
        """Fill every unset path from ``root_dir`` so callers never see ``None``."""
        root = self.root_dir.resolve()
        self.root_dir = root
        self.data_dir = (self.data_dir or root / "data").resolve()
        self.models_dir = (self.models_dir or root / "models").resolve()
        self.reports_dir = (self.reports_dir or root / "reports").resolve()
        self.state_dir = (self.state_dir or self.data_dir / "web").resolve()
        self.db_path = (self.db_path or self.state_dir / "app.db").resolve()
        self.sos_data_dir = (self.sos_data_dir or root / "nfl-sos-ratings" / "data").resolve()
        self.web_dist = (self.web_dist or root / "web" / "dist").resolve()
        # The venv interpreter is a symlink to the base Python; resolving it would launch jobs
        # outside the virtual environment, so this path is made absolute but never resolved.
        interpreter = self.python_executable or root / ".venv" / "bin" / Path(sys.executable).name
        self.python_executable = interpreter if interpreter.is_absolute() else root / interpreter

    @property
    def data_path(self) -> Path:
        """Return the resolved data directory."""
        assert self.data_dir is not None  # noqa: S101 - filled in model_post_init
        return self.data_dir

    @property
    def models_path(self) -> Path:
        """Return the resolved models directory."""
        assert self.models_dir is not None  # noqa: S101 - filled in model_post_init
        return self.models_dir

    @property
    def reports_path(self) -> Path:
        """Return the resolved reports directory."""
        assert self.reports_dir is not None  # noqa: S101 - filled in model_post_init
        return self.reports_dir

    @property
    def state_path(self) -> Path:
        """Return the resolved API state directory."""
        assert self.state_dir is not None  # noqa: S101 - filled in model_post_init
        return self.state_dir

    @property
    def database_path(self) -> Path:
        """Return the resolved SQLite database path."""
        assert self.db_path is not None  # noqa: S101 - filled in model_post_init
        return self.db_path

    @property
    def web_dist_path(self) -> Path:
        """Return the resolved built-frontend directory."""
        assert self.web_dist is not None  # noqa: S101 - filled in model_post_init
        return self.web_dist

    @property
    def sos_data_path(self) -> Path:
        """Return the resolved nfl-sos-ratings data directory."""
        assert self.sos_data_dir is not None  # noqa: S101 - filled in model_post_init
        return self.sos_data_dir

    @property
    def python_path(self) -> Path:
        """Return the interpreter used for subprocess jobs."""
        assert self.python_executable is not None  # noqa: S101 - filled in model_post_init
        return self.python_executable

    def resolve_jwt_secret(self) -> str:
        """Return the signing secret, generating and persisting one on first use.

        The generated secret is written to ``state_dir/secret.key`` with owner-only permissions so
        sessions survive restarts without the secret living in the environment.
        """
        if self.jwt_secret:
            return self.jwt_secret
        secret_path = self.state_path / SIGNING_KEY_FILENAME
        if secret_path.exists():
            self.jwt_secret = secret_path.read_text(encoding="utf-8").strip()
            return self.jwt_secret
        self.state_path.mkdir(parents=True, exist_ok=True)
        generated = secrets.token_urlsafe(48)
        secret_path.write_text(generated, encoding="utf-8")
        secret_path.chmod(0o600)
        self.jwt_secret = generated
        return generated
