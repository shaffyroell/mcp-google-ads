from __future__ import annotations

import logging
import os
import secrets

import uvicorn
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.routing import Mount, Route

from google_ads_server import _credentials_ctx, mcp

logger = logging.getLogger(__name__)

_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/adwords",
]

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8080")
SECRET_KEY = os.environ.get("SECRET_KEY", secrets.token_hex(32))

# Allow HTTP for local dev; Railway sets BASE_URL to https so this is harmless there
os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")


def _build_flow(state: str | None = None) -> Flow:
    return Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [f"{BASE_URL}/auth/callback"],
            }
        },
        scopes=_SCOPES,
        state=state,
    )


def _creds_from_session(session: dict) -> Credentials | None:
    data = session.get("credentials")
    if not data:
        return None
    return Credentials(
        token=data["token"],
        refresh_token=data.get("refresh_token"),
        token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=data.get("client_id"),
        client_secret=data.get("client_secret"),
        scopes=data.get("scopes", _SCOPES),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def root(request: Request) -> HTMLResponse:
    creds = _creds_from_session(request.session)
    if creds:
        body = "<p>Status: <strong>Connected</strong></p><p>MCP endpoint: <code>/mcp</code></p>"
    else:
        body = (
            "<p>Status: <strong>Not connected</strong></p>"
            '<p><a href="/auth/login">Connect your Google Account</a></p>'
        )
    return HTMLResponse(
        f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Google Ads MCP</title></head>
<body>
  <h1>Google Ads MCP</h1>
  {body}
</body>
</html>"""
    )


async def auth_login(request: Request) -> RedirectResponse:
    flow = _build_flow()
    flow.redirect_uri = f"{BASE_URL}/auth/callback"
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    request.session["oauth_state"] = state
    return RedirectResponse(auth_url)


async def auth_callback(request: Request) -> HTMLResponse:
    state = request.session.get("oauth_state")
    flow = _build_flow(state=state)
    flow.redirect_uri = f"{BASE_URL}/auth/callback"

    code = request.query_params.get("code")
    if not code:
        return HTMLResponse("Authorization failed: no code returned.", status_code=400)

    try:
        flow.fetch_token(code=code)
        creds = flow.credentials
        request.session["credentials"] = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": list(creds.scopes) if creds.scopes else _SCOPES,
        }
    except Exception:
        logger.exception("OAuth callback error")
        return HTMLResponse("Authorization failed. Check server logs.", status_code=400)

    return HTMLResponse(
        """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Google Ads MCP – Connected</title></head>
<body>
  <h1>Google Ads MCP – Connected!</h1>
  <p>Authentication successful. You can close this tab.</p>
  <p>The MCP server is available at <code>/mcp</code>.</p>
</body>
</html>"""
    )


# ---------------------------------------------------------------------------
# Credentials injection middleware (ASGI)
# ---------------------------------------------------------------------------

class _CredentialsMiddleware:
    """
    Thin ASGI wrapper that reads Google credentials from the Starlette session
    and injects them into _credentials_ctx before delegating to the MCP app.

    Because asyncio copies the current context into every new task, any tasks
    spawned by the MCP SSE handler also inherit the injected credentials.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] in ("http", "websocket"):
            session = scope.get("session", {})
            creds = _creds_from_session(session)
            if creds is not None:
                _credentials_ctx.set(creds)
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> Starlette:
    mcp_asgi = _CredentialsMiddleware(mcp.sse_app())

    routes = [
        Route("/", root),
        Route("/health", health),
        Route("/auth/login", auth_login),
        Route("/auth/callback", auth_callback),
        Mount("/mcp", app=mcp_asgi),
    ]

    # Use https_only cookies in production (BASE_URL starts with https)
    https_only = BASE_URL.startswith("https://")

    middleware = [
        Middleware(
            SessionMiddleware,
            secret_key=SECRET_KEY,
            https_only=https_only,
            same_site="lax",
        ),
    ]

    return Starlette(routes=routes, middleware=middleware)


def run_web_server() -> None:
    port = int(os.environ.get("PORT", 8080))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    uvicorn.run(create_app(), host="0.0.0.0", port=port)


if __name__ == "__main__":
    run_web_server()
