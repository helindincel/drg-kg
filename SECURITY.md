# Security Policy

## Supported versions

DRG is in **alpha** (`0.1.x`). Only the latest minor version receives
security fixes. There are no LTS or backport branches.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | ✅                 |
| < 0.1   | ❌                 |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security problems.**

Instead, use **GitHub's private security advisory** flow:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability**.
3. Provide a minimal reproduction, the DRG version, and the impact.

You should expect:

- An acknowledgement within **5 business days**.
- A status update within **14 days**.
- Coordinated disclosure once a fix is released.

## In scope

- Remote/arbitrary code execution paths in `drg.api.server` (FastAPI).
- Authentication / authorization issues in the API server or MCP layer.
- Prompt-injection or data-exfiltration patterns specifically enabled by
  DRG's extraction code (not generic LLM provider issues).
- Secret leakage from configuration loaders / `.env` handling.
- Dependency vulnerabilities flagged via `pip-audit` / Dependabot.

## Out of scope

- Vulnerabilities in third-party LLM providers (OpenAI, Gemini, etc.) or
  their SDKs — report those upstream.
- Resource exhaustion via deliberately large input documents (DRG does
  not enforce input-size limits by default; callers are expected to).
- Issues that require attacker-controlled environment variables or
  filesystem access.

## Disclosure practices

We follow **coordinated disclosure**: we work with reporters on a fix and
release notes before public details are published. Credit is given in the
release notes unless the reporter requests anonymity.

## Handling secrets in this repository

- DRG itself never logs secrets; if you find a code path that does,
  please report it.
- `.gitignore` blocks `.env*` files; `pre-commit` includes
  `detect-private-key`. If you spot a leaked credential in git history,
  report it privately and we will rotate + force-push immediately.
