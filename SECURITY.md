# Security policy

## Supported versions

Security fixes are applied on a best-effort basis to the default branch of this repository. There is no long-term support promise for older tags.

## Reporting a vulnerability

Please report security issues through **GitHub private security advisories** on this repository:

1. Open the repository on GitHub.
2. Use **Security → Advisories → Report a vulnerability** (or the equivalent “New draft security advisory” flow for maintainers).
3. Include a clear description, impact, and steps to reproduce. Do not include production secrets.

Do **not** open a public issue for unfixed vulnerabilities.

Private reporting must be enabled by the repository owner on GitHub. If the
reporting button is unavailable, ask a maintainer to provide a private channel
without disclosing the vulnerability publicly.

## Scope notes for reporters

muse-gateway is a local loopback proxy that may hold OAuth tokens or API keys in process memory and may refresh credentials on disk. Reports related to secret leakage, unsafe default binds, credential file handling, SSRF-style host overrides, or logging of sensitive data are especially relevant.

Out of scope examples: third-party provider outages, Muse CLI bugs unrelated to this proxy, and account abuse against OpenAI/Anthropic/Meta.

## Safe defaults (for users)

- Keep the default loopback bind (`127.0.0.1`).
- Do not pass `--allow-non-loopback` or `--allow-oauth-host-override` unless you understand the risk.
- Do not enable verbose logging in shared environments if your threat model includes local log access (bodies and keys are not logged by design, but request metadata may still be sensitive).
- Treat `~/.codex/auth.json`, Claude credential files, and API keys like passwords.
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
