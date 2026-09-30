# Publishing this site

How this documentation is built and deployed. This page doubles as the runbook for anyone
updating it.

## Approach

The site is **plain Markdown in `docs/` on the `main` branch**, rendered by GitHub Pages'
built-in Jekyll with GitHub-flavored Markdown (`markdown: GFM`).

Chosen because it is the simplest reliable option:

- **No build tooling and no CI.** GitHub Pages converts Markdown automatically on every push to
  `main` — no workflow file, no secrets, no Python/Node toolchain.
- **Markdown is the project's documentation language** — content stays reviewable in the repo
  and readable on GitHub.com before it is ever published.
- **`/docs` on `main` keeps the docs versioned with the code** they describe.

## Enabling GitHub Pages (one-time, human action)

1. In the repo: **Settings → Pages**.
2. Under **Build and deployment**, set **Source** to **Deploy from a branch**.
3. **Branch:** `main` · **Folder:** `/docs`.
4. Click **Save**.

GitHub then builds and publishes on every push to `main`. The site appears at:

> `https://devrkd.github.io/morpheus/`

The first build takes about a minute; deployment status is shown on the Pages settings page and
as a check on each commit. Free-tier Pages requires a public repository — this repo is public.

## Configuration

`docs/_config.yml` is the only config file:

```yaml
title: Morpheus
description: A Bedrock-style unified inference service — one API, one model id, any provider
markdown: GFM
```

- `markdown: GFM` makes the renderer match GitHub's own UI (tables, lists, fenced code render
  identically in the repo browser and on the published site).
- No theme is declared, so the default Primer theme applies. To change the look later, add a
  line such as `theme: minima` or `theme: jekyll-theme-cayman` — one-line change, no other
  setup.

## Content conventions

- **Pages are plain Markdown** with no front matter: `index.md`, `features.md`,
  `architecture.md`, `getting-started.md`, `operations.md`, `publishing.md`.
- **Relative links end in `.md`** — e.g. `[Capabilities](features.md)`. These resolve when
  browsing the repo on GitHub.com, and Jekyll rewrites them to `.html` on the published site.
- **Only portable Markdown** is used: headings, tables, lists, fenced code blocks, and ASCII
  diagrams. The default Pages pipeline does not render Mermaid, and some kramdown-only syntax
  is unavailable under GFM.
- **One level of headings per page**, no front matter, no assets — everything renders with the
  default theme out of the box.

## Updating content

1. Edit the relevant `.md` file in `docs/`.
2. Commit and push to `main` (directly or via PR).
3. Wait ~1 minute for the Pages build; verify at `https://devrkd.github.io/morpheus/`.

When the API surface or behaviour changes, update the corresponding page (`features.md` for the
API, `operations.md` for configuration and deployment, `architecture.md` for internals) in the
same PR as the change.

## Optional upgrades (not currently used)

- **A Jekyll theme** (`minima`, `cayman`, or a remote theme) for a different look — a one-line
  `_config.yml` change.
- **MkDocs + Material** for sidebar navigation and full-text search — requires a GitHub Actions
  workflow (`peaceiris/actions-gh-pages` or similar) instead of the branch deploy.
- **A custom domain** — set it under Settings → Pages and add the `CNAME` DNS record; GitHub
  provisions TLS automatically.
