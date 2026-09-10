"""Manage users from the command line.

Examples:
    .venv/bin/python -m nfl_predictor.api.auth.cli create-user mitch --role admin
    .venv/bin/python -m nfl_predictor.api.auth.cli list-users
    .venv/bin/python -m nfl_predictor.api.auth.cli set-password mitch

"""

from __future__ import annotations

import argparse
import getpass
import os

from nfl_predictor.api.auth import users as user_store
from nfl_predictor.api.db import Database
from nfl_predictor.api.settings import Settings
from nfl_predictor.utils.logger import log

PASSWORD_ENV = "NFLP_PASSWORD"  # noqa: S105 - name of the variable, not a secret


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the sub-command and its options."""
    parser = argparse.ArgumentParser(description="Manage nfl-predictor web users.")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-user", help="Create a user")
    create.add_argument("username")
    create.add_argument("--role", choices=user_store.ROLES, default="viewer")
    create.add_argument("--password", default=None, help=f"Defaults to ${PASSWORD_ENV} or a prompt")
    sub.add_parser("list-users", help="List users")
    reset = sub.add_parser("set-password", help="Change a user's password")
    reset.add_argument("username")
    reset.add_argument("--password", default=None, help=f"Defaults to ${PASSWORD_ENV} or a prompt")
    return parser.parse_args(argv)


def _resolve_password(explicit: str | None) -> str:
    """Return the password from the flag, the environment, or an interactive prompt."""
    if explicit:
        return explicit
    from_env = os.environ.get(PASSWORD_ENV)
    if from_env:
        return from_env
    return getpass.getpass("Password: ")


def main(argv: list[str] | None = None, settings: Settings | None = None) -> int:
    """Run the requested user-management command and return an exit code."""
    args = _parse_args(argv)
    settings = settings or Settings()
    db = Database(settings.database_path)
    if args.command == "create-user":
        user = user_store.create_user(
            db, args.username, _resolve_password(args.password), args.role
        )
        log.info("Created %s user %r (id=%d)", user.role, user.username, user.id)
        return 0
    if args.command == "list-users":
        for user in user_store.list_users(db):
            log.info("%-4d %-24s %s", user.id, user.username, user.role)
        return 0
    users = {user.username: user for user in user_store.list_users(db)}
    target = users.get(args.username)
    if target is None:
        log.error("No user named %r", args.username)
        return 1
    user_store.update_user(db, target.id, password=_resolve_password(args.password))
    log.info("Updated password for %r", args.username)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
