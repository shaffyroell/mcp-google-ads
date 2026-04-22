"""HTTP-based MCP server with OAuth 2.0 authorization for cloud deployment.

Uses the MCP SDK's built-in auth infrastructure (mcp.server.auth.*) so the
OAuth endpoints are spec-correct and compatible with Claude.ai's MCP connector.
Google OAuth is used as the identity provider.

Setup
-----
  GOOGLE_ADS_CLIENT_ID      – OAuth 2.0 Web Application client ID
  GOOGLE_ADS_CLIENT_SECRET  – OAuth 2.0 client secret
  GOOGLE_ADS_DEVELOPER_TOKEN – Google Ads API developer token
  SECRET_KEY                – Cookie signing secret (auto-generated if unset)
  BASE_URL                  – Public URL, e.g. https://my-app.up.railway.app
  PORT                      – Listen port (default: 8080)

Add  <BASE_URL>/auth/callback  as an authorised redirect URI in Google
Cloud Console.
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import html
import json
import logging
import os
import secrets
import time
from typing import Any

import httpx
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    TokenError,
)
from mcp.server.auth.routes import create_auth_routes
from mcp.server.auth.settings import ClientRegistrationOptions
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyHttpUrl
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from google_ads_server import _credentials_ctx
from google_ads_server import mcp as fastmcp

os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

logger = logging.getLogger(__name__)

_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/adwords",
]

_SCOPE_FINGERPRINT: str = hashlib.sha256(
    " ".join(sorted(_SCOPES)).encode()
).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Environment variable '{name}' is not set.")
    return v


def _get_base_url(request: Request) -> str:
    explicit = os.environ.get("BASE_URL", "").rstrip("/")
    if explicit:
        return explicit
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.url.netloc)
    return f"{scheme}://{host}"


def _make_flow(redirect_uri: str) -> Flow:
    client_config = {
        "web": {
            "client_id": _require_env("GOOGLE_ADS_CLIENT_ID"),
            "client_secret": _require_env("GOOGLE_ADS_CLIENT_SECRET"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri],
        }
    }
    return Flow.from_client_config(client_config, scopes=_SCOPES, redirect_uri=redirect_uri)


def _creds_from_store(data: dict) -> Credentials:
    return Credentials(
        token=data["token"],
        refresh_token=data.get("refresh_token"),
        token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=data["client_id"],
        client_secret=data["client_secret"],
        scopes=data.get("scopes"),
    )


async def _fetch_email(access_token: str) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
    return (resp.json() if resp.is_success else {}).get("email", "unknown")


_PAGE = """\
<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Google Ads MCP</title>
<style>body{{font-family:system-ui,sans-serif;max-width:700px;margin:60px auto;padding:0 20px;color:#333}}
h1{{font-size:1.6rem}}pre{{background:#f5f5f5;padding:14px;border-radius:6px;overflow-x:auto;font-size:.9rem}}
a.btn{{display:inline-block;padding:10px 22px;background:#4285f4;color:#fff;border-radius:5px;text-decoration:none;font-weight:500}}
a.btn:hover{{background:#2b6fd4}}.label{{font-weight:600;margin-top:1.2rem;display:block}}</style>
</head><body>{body}</body></html>"""


def _page(body: str) -> HTMLResponse:
    return HTMLResponse(_PAGE.format(body=body))


# ---------------------------------------------------------------------------
# OAuth provider — proxies identity to Google
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Self-contained bearer tokens — survive server restarts, no DB needed.
# Format: base64url(json_payload) + "." + hmac_sha256_hex
# ---------------------------------------------------------------------------

_TOKEN_TTL = 8 * 3600  # 8 hours


def _encode_token(payload: dict, secret: str) -> str:
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    sig = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def _decode_token(token: str, secret: str) -> dict | None:
    try:
        body, sig = token.rsplit(".", 1)
        expected = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(base64.urlsafe_b64decode(body))
        if time.time() > payload.get("expires_at", 0):
            return None
        return payload
    except Exception:
        return None


class GoogleOAuthProvider(OAuthAuthorizationServerProvider):
    """MCP OAuth AS that proxies authentication to Google OAuth 2.0.

    Bearer tokens are self-contained HMAC-signed payloads so they survive
    server restarts without any external storage.
    """

    def __init__(self, secret_key: str) -> None:
        self._secret = secret_key
        self._clients: dict[str, OAuthClientInformationFull] = {}
        self._auth_codes: dict[str, AuthorizationCode] = {}
        self._auth_code_creds: dict[str, dict] = {}
        self._pending: dict[str, dict[str, Any]] = {}

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self._clients[client_info.client_id] = client_info

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        base_url = os.environ.get("BASE_URL", "").rstrip("/")
        if not base_url:
            raise RuntimeError("BASE_URL env var is required for MCP OAuth")

        google_state = secrets.token_urlsafe(16)
        callback_uri = f"{base_url}/auth/callback"
        flow = _make_flow(callback_uri)
        auth_url, _ = flow.authorization_url(
            access_type="offline",
            prompt="consent",
            state=google_state,
        )
        self._pending[google_state] = {
            "params": params,
            "client": client,
            "code_verifier": getattr(flow, "code_verifier", None),
            "callback_uri": callback_uri,
        }
        return auth_url

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        return self._auth_codes.get(authorization_code)

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        creds_data = self._auth_code_creds.pop(authorization_code.code, None)
        if creds_data is None:
            raise TokenError(error="invalid_grant", error_description="Credentials not found")

        del self._auth_codes[authorization_code.code]

        expires_at = int(time.time()) + _TOKEN_TTL
        payload = {
            "credentials": creds_data["credentials"],
            "email": creds_data["email"],
            "client_id": client.client_id,
            "scopes": list(authorization_code.scopes),
            "expires_at": expires_at,
        }
        bearer = _encode_token(payload, self._secret)
        return OAuthToken(
            access_token=bearer,
            token_type="Bearer",
            expires_in=_TOKEN_TTL,
            scope=" ".join(authorization_code.scopes),
        )

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str):
        return None

    async def exchange_refresh_token(self, client, refresh_token, scopes):
        raise TokenError(error="unsupported_grant_type")

    async def load_access_token(self, token: str) -> AccessToken | None:
        payload = _decode_token(token, self._secret)
        if payload is None:
            return None
        return AccessToken(
            token=token,
            client_id=payload["client_id"],
            scopes=payload["scopes"],
            expires_at=payload["expires_at"],
        )

    async def revoke_token(self, token: AccessToken) -> None:
        pass  # stateless tokens cannot be revoked; they expire naturally

    def decode_token_payload(self, token: str) -> dict | None:
        return _decode_token(token, self._secret)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> Starlette:
    secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
    base_url = os.environ.get("BASE_URL", "").rstrip("/")

    provider = GoogleOAuthProvider(secret_key)

    # FastMCP stores the underlying lowlevel Server on _mcp_server.
    # All @mcp.tool / @mcp.prompt handlers registered in google_ads_server.py
    # are already wired up on this server object at import time.
    session_manager = StreamableHTTPSessionManager(
        app=fastmcp._mcp_server,
        json_response=False,
        stateless=True,
    )

    issuer = base_url or "https://placeholder.invalid"
    auth_routes = create_auth_routes(
        provider=provider,
        issuer_url=AnyHttpUrl(issuer),
        client_registration_options=ClientRegistrationOptions(
            enabled=True,
            valid_scopes=_SCOPES,
            default_scopes=_SCOPES,
        ),
    )

    # ------------------------------------------------------------------
    # Override /.well-known/oauth-authorization-server to advertise
    # token_endpoint_auth_method "none" (required for Claude.ai public
    # clients; the SDK's built-in metadata omits it).
    # ------------------------------------------------------------------

    async def oauth_server_metadata(request: Request) -> JSONResponse:
        base = _get_base_url(request)
        metadata = {
            "issuer": base,
            "authorization_endpoint": f"{base}/authorize",
            "token_endpoint": f"{base}/token",
            "registration_endpoint": f"{base}/register",
            "scopes_supported": _SCOPES,
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "token_endpoint_auth_methods_supported": ["none", "client_secret_post", "client_secret_basic"],
            "code_challenge_methods_supported": ["S256"],
        }
        return JSONResponse(metadata, headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------------
    # Custom /register — must shadow the SDK's built-in handler.
    # Claude.ai registers with only ["authorization_code"] (public client
    # per MCP spec); the SDK rejects that, so we normalise grant_types.
    # ------------------------------------------------------------------

    async def register_client(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid_client_metadata"}, status_code=400)

        logger.info("CLIENT REGISTRATION: body=%s", body)
        grant_types: list = body.get("grant_types", ["authorization_code", "refresh_token"])
        if "authorization_code" not in grant_types:
            return JSONResponse(
                {"error": "invalid_client_metadata",
                 "error_description": "grant_types must include authorization_code"},
                status_code=400,
            )
        if "refresh_token" not in grant_types:
            grant_types = list(grant_types) + ["refresh_token"]

        redirect_uris_raw: list = body.get("redirect_uris", [])
        if not redirect_uris_raw:
            return JSONResponse(
                {"error": "invalid_client_metadata",
                 "error_description": "redirect_uris is required"},
                status_code=400,
            )

        auth_method: str = body.get("token_endpoint_auth_method") or "none"
        client_secret = secrets.token_hex(32) if auth_method != "none" else None

        import uuid
        client_id = str(uuid.uuid4())
        issued_at = int(time.time())

        client_info = OAuthClientInformationFull(
            client_id=client_id,
            client_secret=client_secret,
            client_id_issued_at=issued_at,
            redirect_uris=redirect_uris_raw,
            token_endpoint_auth_method=auth_method,
            grant_types=grant_types,
            response_types=body.get("response_types", ["code"]),
            scope=" ".join(_SCOPES),
        )
        await provider.register_client(client_info)
        logger.info("CLIENT REGISTERED: client_id=%s auth_method=%s", client_id, auth_method)

        resp_body: dict = {
            "client_id": client_id,
            "client_id_issued_at": issued_at,
            "redirect_uris": redirect_uris_raw,
            "grant_types": grant_types,
            "response_types": ["code"],
            "token_endpoint_auth_method": auth_method,
            "scope": " ".join(_SCOPES),
        }
        if client_secret:
            resp_body["client_secret"] = client_secret
        return JSONResponse(resp_body, status_code=201)

    # ------------------------------------------------------------------
    # Protected Resource Metadata (RFC 9728)
    # ------------------------------------------------------------------

    _wk_headers = {
        "Cache-Control": "no-store, must-revalidate",
        "ETag": f'"{_SCOPE_FINGERPRINT}"',
    }

    async def protected_resource_metadata(request: Request) -> JSONResponse:
        base = _get_base_url(request)
        return JSONResponse(
            {"resource": base, "authorization_servers": [base], "scopes_supported": _SCOPES},
            headers=_wk_headers,
        )

    # ------------------------------------------------------------------
    # Google OAuth callback — shared by MCP flow and browser login
    # ------------------------------------------------------------------

    async def auth_callback(request: Request) -> Response:
        code = request.query_params.get("code", "")
        state = request.query_params.get("state", "")
        error = request.query_params.get("error", "")

        if error:
            return _page(f"<h1>Sign-in failed</h1><p>{html.escape(error)}</p><p><a href='/'>Try again</a></p>")

        is_mcp = state in provider._pending

        if is_mcp:
            pending = provider._pending.pop(state)
            callback_uri = pending["callback_uri"]
            code_verifier = pending.get("code_verifier")
        else:
            stored = request.session.pop("oauth_state", None)
            if state != stored:
                return _page("<h1>Invalid request</h1><p>State mismatch.</p><p><a href='/'>Try again</a></p>")
            callback_uri = request.session.pop("redirect_uri", None)
            code_verifier = request.session.pop("code_verifier", None)

        try:
            flow = _make_flow(callback_uri)
            if code_verifier:
                flow.code_verifier = code_verifier
            flow.fetch_token(code=code)
        except Exception as exc:
            logger.exception("fetch_token failed")
            return _page(f"<h1>Sign-in failed</h1><pre>{html.escape(str(exc))}</pre><p><a href='/'>Try again</a></p>")

        creds = flow.credentials
        user_email = await _fetch_email(creds.token)

        creds_data = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": list(creds.scopes) if creds.scopes else _SCOPES,
        }

        if is_mcp:
            params: AuthorizationParams = pending["params"]
            client: OAuthClientInformationFull = pending["client"]
            mcp_code = secrets.token_urlsafe(32)

            auth_code_obj = AuthorizationCode(
                code=mcp_code,
                scopes=params.scopes or _SCOPES,
                expires_at=time.time() + 600,
                client_id=client.client_id,
                code_challenge=params.code_challenge,
                redirect_uri=params.redirect_uri,
                redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            )
            provider._auth_codes[mcp_code] = auth_code_obj
            provider._auth_code_creds[mcp_code] = {
                "credentials": creds_data,
                "email": user_email,
            }

            redirect_uri_str = str(params.redirect_uri)
            sep = "&" if "?" in redirect_uri_str else "?"
            return RedirectResponse(
                f"{redirect_uri_str}{sep}code={mcp_code}&state={params.state or ''}",
                status_code=302,
            )
        else:
            payload = {
                "credentials": creds_data,
                "email": user_email,
                "client_id": "browser",
                "scopes": _SCOPES,
                "expires_at": int(time.time()) + 86400,
            }
            bearer = _encode_token(payload, secret_key)
            request.session["user_email"] = user_email
            request.session["bearer_token"] = bearer
            return RedirectResponse("/", status_code=302)

    # ------------------------------------------------------------------
    # Browser login helpers
    # ------------------------------------------------------------------

    async def auth_google(request: Request) -> Response:
        base = _get_base_url(request)
        uri = f"{base}/auth/callback"
        flow = _make_flow(uri)
        auth_url, state = flow.authorization_url(
            access_type="offline", include_granted_scopes="true", prompt="consent"
        )
        request.session["oauth_state"] = state
        request.session["redirect_uri"] = uri
        if getattr(flow, "code_verifier", None):
            request.session["code_verifier"] = flow.code_verifier
        return RedirectResponse(auth_url)

    async def auth_logout(request: Request) -> Response:
        request.session.clear()
        return RedirectResponse("/")

    # ------------------------------------------------------------------
    # Landing page and health
    # ------------------------------------------------------------------

    async def health(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def index(request: Request) -> HTMLResponse:
        base = _get_base_url(request)
        email = request.session.get("user_email")
        callback_uri = f"{html.escape(base)}/auth/callback"
        body = (
            f"<h1>Google Ads MCP Server</h1>"
            + (f"<p>Signed in as <strong>{html.escape(email)}</strong></p>" if email else "")
            + f'<span class="label">1. Add this URL to Claude.ai as a remote MCP server:</span>'
            + f"<pre>{html.escape(base)}/mcp</pre>"
            + f'<span class="label">2. Required: Register this Authorized Redirect URI in <a href="https://console.cloud.google.com/apis/credentials" target="_blank">Google Cloud Console</a> (Credentials → OAuth 2.0 Client → Authorized redirect URIs):</span>'
            + f'<pre style="border:2px solid #e53935;color:#e53935">{callback_uri}</pre>'
            + '<p style="color:#555;font-size:.9rem">⚠️ Your OAuth client must be type <strong>Web application</strong> (not Desktop app). '
            + 'If you see a <em>redirect_uri_mismatch</em> error from Google, add the URI above to your OAuth client\'s authorized redirect URIs.</p>'
            + "<p>Claude.ai will open a sign-in popup automatically once the redirect URI is registered.</p>"
            + (
                "<p><a href='/auth/logout' style='color:#888;font-size:.9rem'>Sign out</a></p>"
                if email
                else f"<br><a href='/auth/google' class='btn'>Sign in with Google</a>"
            )
        )
        return _page(body)

    # ------------------------------------------------------------------
    # MCP endpoint — validates Bearer token then delegates to session manager
    # ------------------------------------------------------------------

    class _MCPAuthApp:
        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] != "http":
                return

            headers = dict(scope.get("headers", []))
            token: str | None = None

            auth = headers.get(b"authorization", b"").decode()
            if auth.startswith("Bearer "):
                token = auth[7:]

            if not token:
                from urllib.parse import parse_qs
                qs = scope.get("query_string", b"").decode()
                token = (parse_qs(qs).get("token") or [None])[0]

            payload = provider.decode_token_payload(token) if token else None

            if not payload:
                logger.warning("MCP /mcp: 401 — token=%s payload=%s", bool(token), bool(payload))
                resp = Response(
                    status_code=401,
                    headers={"WWW-Authenticate": f'Bearer realm="{_SCOPE_FINGERPRINT}"'},
                )
                await resp(scope, receive, send)
                return

            creds = _creds_from_store(payload["credentials"])
            ctx_token = _credentials_ctx.set(creds)
            try:
                await session_manager.handle_request(scope, receive, send)
            finally:
                _credentials_ctx.reset(ctx_token)

    # ------------------------------------------------------------------
    # Assemble Starlette app
    # ------------------------------------------------------------------

    async def setup(request: Request) -> HTMLResponse:
        base = _get_base_url(request)
        callback_uri = f"{base}/auth/callback"
        body = (
            "<h1>Setup Checklist</h1>"
            "<p>Complete these steps before connecting Claude.ai:</p>"
            "<ol>"
            "<li><strong>Google Cloud Console</strong> → Credentials → your OAuth 2.0 Client<br>"
            "&nbsp;&nbsp;• Client type must be <strong>Web application</strong> (not Desktop app)<br>"
            f"&nbsp;&nbsp;• Add this to <strong>Authorized redirect URIs</strong>:</li>"
            f'</ol><pre style="border:2px solid #e53935;color:#e53935;margin:4px 0 12px">{html.escape(callback_uri)}</pre>'
            "<ol start='2'>"
            "<li>Set environment variable <code>BASE_URL</code> to your server's public URL: "
            f"<code>{html.escape(base)}</code></li>"
            "<li>Add this MCP URL to Claude.ai:</li>"
            f"</ol><pre>{html.escape(base)}/mcp</pre>"
            f'<p><a href="/">← Back</a></p>'
        )
        return _page(body)

    async def debug(request: Request) -> JSONResponse:
        base_from_env = os.environ.get("BASE_URL", "").rstrip("/")
        base_from_request = _get_base_url(request)
        callback_uri_mcp = f"{base_from_env}/auth/callback" if base_from_env else "(BASE_URL not set — MCP auth will fail)"
        callback_uri_browser = f"{base_from_request}/auth/callback"
        return JSONResponse({
            "BASE_URL_env": base_from_env or "(not set)",
            "base_url_from_request_headers": base_from_request,
            "redirect_uri_sent_to_google_for_mcp_flow": callback_uri_mcp,
            "redirect_uri_sent_to_google_for_browser_flow": callback_uri_browser,
            "action_required": "Register 'redirect_uri_sent_to_google_for_mcp_flow' in Google Cloud Console → Credentials → OAuth Client → Authorized redirect URIs",
        })

    routes = [
        # These MUST come before auth_routes to shadow the SDK's built-in handlers.
        Route("/.well-known/oauth-authorization-server", oauth_server_metadata, methods=["GET"]),
        Route("/register", register_client, methods=["POST"]),
        *auth_routes,
        Route("/.well-known/oauth-protected-resource", protected_resource_metadata),
        Route("/auth/callback", auth_callback),
        Route("/auth/google", auth_google),
        Route("/auth/logout", auth_logout),
        Route("/health", health),
        Route("/setup", setup),
        Route("/debug", debug),
        Route("/", index),
        Route("/mcp", _MCPAuthApp()),
    ]

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        async with session_manager.run():
            yield

    app = Starlette(
        routes=routes,
        middleware=[
            Middleware(
                SessionMiddleware,
                secret_key=secret_key,
                max_age=86400,
                https_only=False,
            )
        ],
        lifespan=lifespan,
    )
    return app


def run_web_server() -> None:
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger.info("Google Ads MCP web server starting on port %s", port)
    logger.info("MCP endpoint: %s/mcp", os.environ.get("BASE_URL", f"http://localhost:{port}"))
    uvicorn.run(create_app(), host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    run_web_server()
