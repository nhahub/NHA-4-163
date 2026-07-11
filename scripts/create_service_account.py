"""CLI utility to create service accounts in the ``service_accounts`` table.

Usage::

    python scripts/create_service_account.py \
        --username alice \
        --role clinician \
        --postgres-dsn "postgresql://healthcare_app:pw@localhost/healthcare"

    # Or use environment variables from .env:
    python scripts/create_service_account.py --username svc-bot --role service

The password is prompted interactively (never passed as a CLI argument to
avoid shell history exposure).  The hash is stored as a bcrypt digest with
cost factor 12 — the plaintext is NEVER written to disk or logs.

Roles: admin | clinician | researcher | service
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys


def _get_postgres_dsn(args_dsn: str | None) -> str:
    if args_dsn:
        return args_dsn
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "healthcare")
    user = os.environ.get("POSTGRES_USER", "healthcare_app")
    pw = os.environ.get("POSTGRES_PASSWORD", "")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


def create_account(
    username: str,
    role: str,
    postgres_dsn: str,
    *,
    force: bool = False,
) -> None:
    """Prompt for a password and insert a service account row.

    Args:
        username: Account username (must be unique).
        role: One of ``admin``, ``clinician``, ``researcher``, ``service``.
        postgres_dsn: Sync PostgreSQL DSN.
        force: If True, update an existing account instead of failing.

    Raises:
        SystemExit: On validation failure or DB error.
    """
    valid_roles = {"admin", "clinician", "researcher", "service"}
    if role not in valid_roles:
        print(f"ERROR: role must be one of {sorted(valid_roles)}", file=sys.stderr)
        sys.exit(1)

    try:
        from passlib.context import CryptContext
    except ImportError:
        print("ERROR: passlib not installed — run: pip install passlib[bcrypt]", file=sys.stderr)
        sys.exit(1)

    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed", file=sys.stderr)
        sys.exit(1)

    password = getpass.getpass(f"Password for '{username}': ")
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("ERROR: Passwords do not match", file=sys.stderr)
        sys.exit(1)
    if len(password) < 12:
        print("ERROR: Password must be at least 12 characters", file=sys.stderr)
        sys.exit(1)

    pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)
    hashed = pwd_ctx.hash(password)
    # Zero out the plaintext variable immediately after hashing
    password = ""
    confirm = ""

    try:
        conn = psycopg2.connect(postgres_dsn)
        try:
            with conn.cursor() as cur:
                if force:
                    cur.execute(
                        """
                        INSERT INTO service_accounts (username, hashed_password, role)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (username)
                        DO UPDATE SET hashed_password = EXCLUDED.hashed_password,
                                      role = EXCLUDED.role,
                                      is_active = TRUE
                        """,
                        (username, hashed, role),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO service_accounts (username, hashed_password, role)
                        VALUES (%s, %s, %s)
                        """,
                        (username, hashed, role),
                    )
            conn.commit()
        finally:
            conn.close()
    except psycopg2.errors.UniqueViolation:
        print(
            f"ERROR: username '{username}' already exists. "
            "Use --force to update the existing account.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: Database write failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"OK: service account '{username}' (role={role}) created successfully.")


def deactivate_account(username: str, postgres_dsn: str) -> None:
    """Set ``is_active = FALSE`` for an existing service account.

    Args:
        username: Account username to deactivate.
        postgres_dsn: Sync PostgreSQL DSN.
    """
    import psycopg2

    conn = psycopg2.connect(postgres_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE service_accounts SET is_active = FALSE WHERE username = %s",
                (username,),
            )
            if cur.rowcount == 0:
                print(f"WARNING: no account found with username '{username}'", file=sys.stderr)
            else:
                print(f"OK: '{username}' deactivated.")
        conn.commit()
    finally:
        conn.close()


def list_accounts(postgres_dsn: str) -> None:
    """Print all service accounts (no passwords) to stdout.

    Args:
        postgres_dsn: Sync PostgreSQL DSN.
    """
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(postgres_dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT username, role, is_active, created_at, last_login_at "
                "FROM service_accounts ORDER BY created_at"
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        print("No service accounts found.")
        return

    fmt = "{:<30} {:<12} {:<8} {:<20} {}"
    print(fmt.format("USERNAME", "ROLE", "ACTIVE", "CREATED", "LAST LOGIN"))
    print("-" * 85)
    for r in rows:
        last = str(r["last_login_at"])[:16] if r["last_login_at"] else "never"
        print(
            fmt.format(
                r["username"],
                r["role"],
                "yes" if r["is_active"] else "no",
                str(r["created_at"])[:16],
                last,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage Healthcare API service accounts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--postgres-dsn", default=None, help="PostgreSQL DSN (falls back to env vars)"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # create
    p_create = sub.add_parser("create", help="Create a new service account")
    p_create.add_argument("--username", required=True)
    p_create.add_argument(
        "--role", required=True, choices=["admin", "clinician", "researcher", "service"]
    )
    p_create.add_argument("--force", action="store_true", help="Update if username already exists")

    # deactivate
    p_deact = sub.add_parser("deactivate", help="Deactivate an existing account")
    p_deact.add_argument("--username", required=True)

    # list
    sub.add_parser("list", help="List all service accounts")

    args = parser.parse_args()
    dsn = _get_postgres_dsn(args.postgres_dsn)

    if args.command == "create":
        create_account(args.username, args.role, dsn, force=args.force)
    elif args.command == "deactivate":
        deactivate_account(args.username, dsn)
    elif args.command == "list":
        list_accounts(dsn)


if __name__ == "__main__":
    main()
