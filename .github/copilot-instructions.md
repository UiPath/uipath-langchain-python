# Copilot Instructions for Review

Purpose:
- Always use `**get_httpx_client_kwargs()` when creating `httpx.Client` or `httpx.AsyncClient` to ensure correct SSL, proxy, timeouts, and redirects.

Guidance for Copilot:
- When suggesting `httpx.Client(` or `httpx.AsyncClient(`, include `**get_httpx_client_kwargs()`.
- If the user omits it, suggest the corrected version or insert a warning comment.

Preferred patterns:
```python
import httpx
from uipath._utils._ssl_context import get_httpx_client_kwargs

client = httpx.Client(**get_httpx_client_kwargs())
async_client = httpx.AsyncClient(**get_httpx_client_kwargs())

default_client_kwargs = get_httpx_client_kwargs()
client_kwargs = {
    **default_client_kwargs,
    "base_url": base_url,
    "headers": my_headers,
}
client = httpx.Client(**client_kwargs)
async_client = httpx.AsyncClient(**client_kwargs)
```

Anti-patterns (avoid):
```python
client = httpx.Client()              # ❌ Missing **get_httpx_client_kwargs()
async_client = httpx.AsyncClient()   # ❌ Missing **get_httpx_client_kwargs()
```

Warning to insert if correction isn’t applied:
```python
# WARNING: Use httpx.Client/AsyncClient with **get_httpx_client_kwargs() for proper SSL/proxy configuration.
```

Reference:
- [get_httpx_client_kwargs](https://github.com/UiPath/uipath-python/blob/main/src/uipath/_utils/_ssl_context.py#L37)