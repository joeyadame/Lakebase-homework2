"""
One-time setup script for the Lakebase secret used by Weather Intelligence.

Run from a Databricks notebook terminal or another environment where the
Databricks SDK is authenticated:

    python setup_secrets.py
"""

import getpass
from urllib.parse import quote, urlsplit, urlunsplit

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace


def ensure_scope(client: WorkspaceClient, scope: str) -> None:
    try:
        client.secrets.create_scope(scope=scope)
    except Exception as exc:
        message = str(exc).lower()
        if "already" not in message and "exists" not in message:
            raise


def connection_url_from_prompt() -> str:
    url = getpass.getpass("Paste your Lakebase connection URL: ").strip()
    parsed = urlsplit(url)

    if not parsed.scheme.startswith("postgres"):
        raise SystemExit("Lakebase URL must start with postgresql:// or postgres://")
    if not parsed.hostname:
        raise SystemExit("Lakebase URL must include a host name.")
    if parsed.password:
        return url

    password = getpass.getpass(
        "Paste the Lakebase database password for this URL: "
    ).strip()
    if not password:
        raise SystemExit("A password is required when the URL does not include one.")

    username = parsed.username or ""
    auth = quote(username, safe="%")
    auth = f"{auth}:{quote(password, safe='')}@"
    if parsed.port:
        netloc = f"{auth}{parsed.hostname}:{parsed.port}"
    else:
        netloc = f"{auth}{parsed.hostname}"

    return urlunsplit(
        (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
    )


def main() -> None:
    client = WorkspaceClient()
    scope = "database"

    ensure_scope(client, scope)
    client.secrets.put_secret(
        scope=scope,
        key="lakebase-url",
        string_value=connection_url_from_prompt(),
    )
    client.secrets.put_acl(
        scope=scope,
        principal="users",
        permission=workspace.AclPermission.READ,
    )
    print("Stored secret database/lakebase-url. NWS weather data does not require an API key.")


if __name__ == "__main__":
    main()
