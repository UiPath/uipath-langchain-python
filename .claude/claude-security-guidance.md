## Security Rules

- Always validate and sanitize user input at system boundaries
- Use parameterized queries for all database access — never concatenate user input into queries
- Never store secrets, API keys, or credentials in source code — use environment variables or a secrets manager
- Use strong cryptographic algorithms (SHA-256+, AES-256, Ed25519) — avoid MD5, SHA-1, DES
- Apply the principle of least privilege to all access control decisions
- Escape all output rendered in HTML to prevent cross-site scripting (XSS)
- Validate file paths to prevent directory traversal attacks
- Use HTTPS for all external communications
- Log security-relevant events but never log sensitive data (passwords, tokens, PII)
- Keep dependencies up to date and audit them regularly for known vulnerabilities
- Calculate prices and enforce limits server-side — never trust client-submitted values
- Use integer cents for money — never floating point for currency
- Mask PII before sending content to LLMs — remove tokens, credentials, emails
- Authenticate and authorize every endpoint — never defer auth to a later step

## Vulnerability Checklists

### Injection (CRITICAL)
- SQL queries with string concatenation or template literals — use parameterized queries
- Raw query methods (.raw(), .query(), $queryRaw, extra(), $where) with interpolated user input
- ORM bypasses: Django extra(), MongoDB $gt/$ne operator injection from user objects
- Second-order SQL injection — data stored safely but used unsafely in later queries
- exec(), spawn(), system(), popen() with user-controlled arguments
- subprocess.Popen(cmd, shell=True) with user input
- User-controlled URLs in server-side HTTP requests (SSRF) — validate against allowlist, block internal ranges
- SSRF bypass techniques: IPv6 encoding, DNS rebinding, vertical tab (%09), whitelisted domain prefix
- render_template_string(), ejs.render(), Template() with user input (SSTI)
- User input in LDAP filters, XPath expressions, or HTTP headers
- File upload type validation relying on extension or Content-Type header without reading magic bytes from file content
- Regex with nested quantifiers or ambiguous alternation causing ReDoS — (a|a)+, (r*)+, (__|.)+ — use linear-time engines or rewrite

### Authentication & Authorization (CRITICAL)
- Endpoints without authentication middleware — check for commented-out [Authorize] or AllowAnonymous on sensitive endpoints
- Routes accessing resources by user-supplied ID without ownership checks (BOLA/IDOR)
- Admin routes accessible to regular users — missing role checks server-side
- Sequential/predictable IDs enabling enumeration — use UUIDs
- JWT accepting alg:none or RS256→HS256 switching
- JWT without exp claim or aud/iss not validated — always set ValidateAudience=true, ValidateLifetime=true
- JWT validated via ReadJwtToken() not ValidateToken() — read-only parsing trusts forged claims
- Hardcoded JWT signing secrets
- Session not regenerated after login or invalidated on logout/password change
- Password reset tokens that are predictable or don't expire
- Development/test mode auth bypass — RequireAssertion(_ => true), IsDevelopment() checks reachable in production
- Auth/authz gated behind feature flags that can be toggled off
- S2S scope validation using OR instead of AND — require ALL needed scopes
- Bulk endpoints looping over IDs without per-item authorization
- @uipath.com email check used for authorization — bypassable via custom SSO
- Body field (tenantId, orgId, resourceId) used for permission decision instead of JWT/header
- POST/mutation endpoints with cookie auth missing anti-CSRF token validation
- Login/reset endpoints returning different error messages for valid vs invalid accounts — enables user enumeration

### XSS & Frontend (HIGH)
- innerHTML, outerHTML, document.write() with user-controlled data
- dangerouslySetInnerHTML in React, v-html in Vue, [innerHTML] in Angular, bypassSecurityTrustHtml()
- Template unescaped output: <%- %> (EJS), {{{ }}} (Handlebars), |safe (Jinja2)
- Template literal HTML with user input (no framework escaping)
- eval(), setTimeout(string), new Function(string) with user input
- javascript: URI in href/redirect attributes
- window.location = userInput, res.redirect(req.query.next) without domain validation (open redirect)
- LLM/MCP server responses rendered as HTML without DOMPurify
- SVG payloads in chat/upload interfaces (SVG can contain script tags)
- lodash.merge({}, userInput) or deep merge with user objects — filter __proto__, constructor, prototype (prototype pollution)

### Path Traversal (HIGH)
- Query parameters or headers (tenantId, orgName, solutionId) used to construct file paths without validation
- File upload accepting arbitrary extensions without allowlist
- File update endpoints allowing overwrite of critical system files
- URL-encoded traversal not blocked: %2e%2e, %09, %0a — vertical tab bypass (.%09./)
- Missing path canonicalization before file/URL construction — use path.resolve + containment check
- MCP tool URL construction with string concatenation — validate IDs as UUIDs

### Cross-Tenant Isolation (CRITICAL)
- Tenant ID from request body instead of authenticated session/JWT
- Database queries without tenant/org ID filter on shared tables
- Resource IDs from one org accepted in another org's API
- X-UiPath-Internal-TenantId accepted without validating against JWT org (prt_id)
- Missing folder authorization for Orchestrator folder-scoped resources
- Tenant ID not verified as belonging to authenticated org via OMS
- EF Core IgnoreQueryFilters() calls in production code paths
- Raw SQL queries missing explicit tenant/org WHERE clauses (not relying on ORM global filters)
- Stored procedures and custom SQL joins missing tenant scoping
- Bulk operations filtering at request level only, not per-item by tenant/org

### Cryptography & Secrets (CRITICAL)
- Database connection strings with embedded passwords
- MD5/SHA1 for password hashing — use bcrypt/scrypt/argon2
- Math.random()/random.random() for security tokens — use crypto.randomBytes/secrets
- .env files with real credentials committed
- appsettings.Development.json with production secrets — add to .gitignore and .dockerignore
- .npmrc with _authToken= containing PATs, NuGet.config with ClearTextPassword
- SSH/TLS private keys committed to repo
- Secrets in Docker intermediate build layers — use multi-stage builds with BuildKit secrets
- AES in ECB mode, DES, RC4, or static/predictable IVs/nonces — use AES-GCM
- Missing salt in password hashing
- Environment variable fallbacks with real credentials as defaults — os.getenv('KEY', 'actualpassword')
- Specific token prefixes to grep for: AKIA (AWS), xoxb-/xoxp- (Slack), SG. (SendGrid), ghp_/github_pat_ (GitHub), sk-ant- (Anthropic), ATATT3x (Atlassian)

### Deserialization (CRITICAL)
- **Python**: pickle.load/loads, marshal, shelve with untrusted input; yaml.load() without SafeLoader; jsonpickle.decode() on untrusted input
- **Java**: ObjectInputStream.readObject() on untrusted streams; Jackson DefaultTyping.NON_FINAL or @JsonTypeInfo(Id.CLASS); Fastjson autoType enabled; SnakeYAML without SafeConstructor; XMLDecoder with untrusted XML
- **.NET**: BinaryFormatter, SoapFormatter, NetDataContractSerializer, ObjectStateFormatter; TypeNameHandling=All/Auto/Objects in Newtonsoft.Json; JavaScriptSerializer with SimpleTypeResolver; ViewState with EnableViewStateMac=false
- **JavaScript**: node-serialize, serialize-javascript (eval-based) with untrusted input; js-yaml.load() with DEFAULT_SCHEMA allowing !!js/function; class-transformer plainToClass() without type validation
- Serialized data in cookies, URL parameters, hidden form fields, message queues
- Deserialization before authentication/authorization

### CORS (HIGH)
- Access-Control-Allow-Origin reflecting request Origin without allowlist validation — use exact match
- Origin validation using regex without ^ and $ anchors, or substring/prefix match
- null origin in allowlist — sandboxed iframes and data: URIs send null origin
- Credentials: true with wildcard or reflected origin
- Access-Control-Allow-Methods/Headers not restricted (using *)
- CORS policy not applied to error responses (4xx, 5xx) — leaks data via error timing
- Missing SameSite cookie attribute as defense-in-depth

### CI/CD Pipeline (CRITICAL)
- pull_request_target with checkout of PR code (runs untrusted code with write perms)
- GitHub event fields (${{ github.event.issue.title }}) in run: steps without env: quoting — command injection
- Production service connections accessible from PR pipelines
- Azure DevOps pipelines with pr: trigger AND production service connections
- Missing environment: with protection rules for production deployments
- PATs (ghp_, github_pat_) hardcoded in pipeline YAML
- Permissions: contents: write or pull-requests: write without justification
- --no-verify, --force in push commands

### Header Trust Abuse (CRITICAL)
- Header fallback chains where user-controllable headers take priority over infrastructure-set headers
- X-Internal-UiPath-AccountId/OrganisationId accepted without JWT prt_id validation
- S2S endpoints protected only by scopes in the user scope list — user tokens can call them
- x-envoy-peer-metadata not stripped on egress
- EnvoyFilter with empty workloadSelector: {} — applies globally instead of to specific workloads
- Feature flags controlling which identity headers are read — toggling changes security posture
- jwtRules header output names not matching application code reads

### OAuth & OIDC (CRITICAL)
- redirect_uri validated by prefix/substring match — use exact registered URI match
- PKCE not enforced for public clients, or using plain instead of S256
- Missing or static state parameter — must be cryptographically random, session-bound, single-use
- Authorization codes not single-use or expiring too slowly (>10 minutes)
- Refresh tokens not rotated on use — reuse not detected
- Tokens not bound to audience (aud claim not validated by resource server)
- Implicit flow (response_type=token) still enabled — use authorization code + PKCE
- Client secrets embedded in SPAs, mobile apps, or frontend code
- Client credentials grant not restricted to specific scopes
- Dynamic client registration open to unauthenticated users

### API Security (HIGH)
- Mass assignment — create/update operations without explicit field allowlists (DTOs or select)
- No pagination or unbounded ?limit=999999 — enforce max page size
- API responses returning password hashes, tokens, admin flags, or internal fields — use DTOs
- Error responses exposing stack traces, SQL errors, internal paths
- Missing Content-Type validation on request bodies
- GraphQL introspection enabled in production
- GraphQL queries without depth or complexity limits

### Business Logic (HIGH)
- Non-atomic read-modify-write on balances, inventory, licenses, quotas — use DB transactions with appropriate isolation
- License/quota check-then-assign without locks (TOCTOU) — enforce limits at DB level with constraints
- Client-side price sent to server and trusted — calculate server-side
- Negative quantities or zero-amount transactions accepted
- Discount codes reusable or stackable without limit
- Checkout or approval steps callable out of order — validate state transitions
- Integer overflow in financial calculations
- Missing idempotency keys on payment or provisioning operations
- Concurrent requests bypassing limits — race condition on org/resource creation

### Data Exposure (HIGH)
- console.log(user), logger.info(request.body) logging full objects with PII — redact sensitive fields
- Passwords, tokens, API keys, credit card numbers written to logs
- Stack traces or database errors returned to clients — return generic error with correlation ID
- API responses returning all fields instead of select/DTO — over-fetching
- Tokens in URL query parameters — logged by proxies, browser history, referrer headers
- Sensitive data in localStorage/sessionStorage — accessible to XSS
- Source maps served in production

### AI & Prompt Injection (HIGH)
- User/DOM content concatenated into LLM prompts without separation markers — use clear delimiters
- PII/credentials/tokens in content sent to LLMs without masking
- configure_server or baseUrl settings changeable via user/LLM input — allowlist valid hosts
- MCP tool URL construction with string concatenation — path traversal via instanceId
- Unbounded LLM token consumption — no budget limits on recursive exploration (financial DoS)
- Agent tool definitions with permissions broader than needed — use least-privilege
- Missing UUID validation on LLM-provided IDs used in API paths
- LLM output rendered as HTML without DOMPurify sanitization
- Missing output schema validation on LLM responses

### WebSocket Security (HIGH)
- Origin header not validated during WebSocket handshake — use exact allowlist, reject missing origins
- Authentication not verified before WS upgrade — verify during HTTP upgrade, not after first message
- Incoming WS messages not validated against schema — same injection risks as HTTP
- No per-message authorization — user permissions may change during long-lived connection
- No rate limiting on incoming frames — per-connection throttle required
- WS connection not closed on session expiry/logout
- ws:// used in production instead of wss:// (TLS)

### Infrastructure Misconfigurations (HIGH)
- Containers running as root — use runAsNonRoot: true with specific UID
- privileged: true or dangerous capabilities (SYS_ADMIN, NET_RAW, SYS_PTRACE)
- Docker socket (/var/run/docker.sock) or host filesystem mounted
- Missing resource limits on containers
- Istio PeerAuthentication mode: PERMISSIVE — use STRICT
- Missing NetworkPolicy — default allows all pod-to-pod communication
- ClusterRole with wildcard verbs/resources
- automountServiceAccountToken: true when not needed
- Terraform with publicly_accessible=true, acl="public-read", Action="*", Resource="*"
- Images with :latest tag — pin digests
- Services with type: LoadBalancer or NodePort that should be ClusterIP for internal services
- Internal endpoints (/admin, /debug, /metrics, /actuator) reachable via ingress without auth
- DNS CNAME records pointing to unclaimed cloud resources — subdomain takeover

### Supply Chain (HIGH)
- Known vulnerable dependency versions — check against advisory databases
- Unpinned versions or missing lockfile
- Lockfile out of sync with manifest
- postinstall scripts downloading external binaries
- Private package names claimable on public registries (dependency confusion)
- Dependencies 2+ major versions behind
- Git URL dependencies (mutable, not pinned to hash)

### Rate Limiting (MEDIUM)
- No rate limiting on login/authentication endpoints — brute force
- No rate limiting on password reset endpoints
- No rate limiting on expensive operations (LLM calls, file processing, report generation)
- No pagination limits on list endpoints
- No file size limits on uploads
- No batch size limits on bulk operations

### Client-Side Enforcement (MEDIUM)
- Organization/resource creation limits enforced only in UI — backend accepts unlimited
- Trial/feature activation state checked only in frontend
- URL scheme validation (http/https) only in frontend — backend accepts javascript: URIs
- Input length/format validation only in frontend
- Rate limits or quotas implemented only at UI layer
- Feature flags checked only client-side

### XML/XXE (MEDIUM)
- XML parsing without DtdProcessing.Prohibit or equivalent — enables file disclosure and SSRF via external entities
- Missing XmlResolver = null in .NET — allows resolution of external entities
- Python XML parsing with default parser (xml.etree) instead of defusedxml
- XML entity expansion limits not configured — nested entities cause billion-laughs DoS
- SOAP endpoints or SAML assertion parsing accepting DTD in input
- XML accepted without rejecting <!DOCTYPE declarations at input validation

### gRPC & Protobuf (HIGH)
- gRPC services deployed without global auth interceptor — any client with network access calls any RPC
- Auth interceptor validates identity but no per-RPC authorization — authenticated != authorized
- gRPC server reflection enabled in production — attacker enumerates all services and methods
- Client-supplied user_id/tenant_id in protobuf message trusted instead of extracted from auth context
- Missing message size limits (grpc.max_receive_message_length) — OOM from oversized requests
- Streaming RPCs not revalidating auth periodically — long-lived streams outlive token expiry
- mTLS not enforced for service-to-service gRPC — plaintext or server-only TLS insufficient
- gRPC-Web proxy with permissive CORS — same origin risks as REST

### Iframe & PostMessage (HIGH)
- sandbox="allow-scripts allow-same-origin" on iframe — defeats the sandbox entirely
- Auth tokens (JWT, session) accessible to iframe/worker contexts — use proxy API instead
- addEventListener('message', ...) without validating event.origin against strict allowlist
- User-authored iframe content can initiate API calls in viewing user's context — privilege escalation
- Auto-login mechanisms injecting tokens into embedded app JavaScript context
- allow-popups or allow-top-navigation on sandboxed iframes hosting untrusted content

### Token & Cache Revocation (HIGH)
- S2S token cache TTL too long (>1 hour) — revoked credentials stay valid until cache expires
- JWKS signing key cache without time-based auto-refresh — compromised key trusted until restart
- Permission/role cache TTL >15 minutes — revoked access persists
- Roles baked into JWT as sole authority with no per-request revalidation — changes only on re-login
- Cache keys missing tenant/org ID — cross-tenant cache pollution
- Single global cache key for S2S tokens — one stale token affects all concurrent calls
- Third-party credential caches with no TTL — indefinite retention until service restart
