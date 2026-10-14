# Security policy

## Supported versions

Security fixes are applied on a best-effort basis to the default branch of
this repository. There is no long-term support promise for older tags.

## Reporting a vulnerability

Please use **GitHub private vulnerability reporting** for this repository:

1. Open the repository on GitHub.
2. Open **Security** (or **Security and quality**) and choose
   **Report a vulnerability**.
3. Include a description, impact, and minimal steps to reproduce. Use dummy
   credentials and synthetic conversation content, not production secrets.

Do **not** open a public issue for unfixed vulnerabilities.

Private reporting must be enabled by the repository owner on GitHub. If the
reporting button is unavailable, ask a maintainer to provide a private channel
without disclosing the vulnerability publicly.

See [GitHub's private reporting instructions](https://docs.github.com/en/code-security/how-tos/report-and-fix-vulnerabilities/report-privately)
for the current interface and reporting process.

## Scope notes for reporters

muse-gateway is a local loopback proxy. It handles model requests and tool
history, holds credentials in memory, and may refresh Codex/Claude credentials
on disk. The default Claude transport also starts the official Claude Code
CLI. It is not a multi-user server and has no client authentication.

Secret leakage, unsafe binds, credential file handling, cross-origin redirects,
unsafe host overrides, and sensitive logging/capture are especially relevant.
`/health` exposes process/source information to clients that can reach the
listener; this is another reason not to expose the port publicly.

Third-party outages and client bugs unrelated to this gateway are not gateway
vulnerabilities. Please do not send provider account-abuse reports here.

## Safe defaults (for users)

- Keep the default loopback bind (`127.0.0.1`).
- `--allow-non-loopback` is not access control. Supply your own authentication,
  network restrictions, and transport protection if remote access is needed.
- Do not use `--allow-oauth-host-override` unless you intend to trust the
  replacement host with credentials. Codex/Claude's native host allowlists do
  not cover arbitrary Meta, Devin, local, or generic endpoints.
- Treat Codex/Claude credentials, Devin CLI credentials, OMP's `agent.db`, API
  keys, and any external gateway's credentials like passwords. Provider login
  files can be updated during OAuth refresh; Devin logins are not refreshed
  by this integration.
- Meta, local, and generic routes can forward incoming authorization when no
  route key is configured. Only configure upstream URLs you trust; use a
  dedicated local API key rather than forwarding a Muse subscription token to
  a third-party server. Custom Meta/local URLs are not protected by the native
  Codex/Claude OAuth host allowlists.
- Prefer TLS for non-loopback upstreams. Plain HTTP sends prompts and credentials
  unencrypted, even when the endpoint is on a private network.
- Review diagnostics before sharing: provider errors and request metadata may
  contain sensitive information. This project does not guarantee universal
  redaction of arbitrary upstream error messages.
- Ordinary rotating logs exclude request/response bodies and authorization
  headers. Verbose mode adds HTTP access metadata, not full wire capture.
  Request metadata can still be sensitive in shared environments. Keep `.env`,
  logs, and captures outside public repositories and shared folders.
- `--capture-dir` / `MUSE_GATEWAY_CAPTURE_DIR` is a separate, opt-in debugging
  mode that saves conversation bodies and wire evidence. Captures are private
  local files with authentication credentials scrubbed; prompts, file contents,
  tool output, and model responses can still contain sensitive information.
  Owner-only permissions, a per-request budget, and credential scrubbing do
  not make a capture safe to publish. Leave capture disabled normally, choose
  a private directory outside the repository, and review any files you share.

See [README.md](README.md#private-failure-captures) for capture details and
[CONTRIBUTING.md](CONTRIBUTING.md#keeping-tests-useful) for synthetic test-fixture
guidance. A private report does not need a complete personal session transcript.
