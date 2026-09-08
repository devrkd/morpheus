"""``python -m model_harness`` / ``model-harness`` entrypoint.

Two subcommands: ``serve`` (the default) and ``mint-key``, which exists so
that creating an inbound credential never requires hand-computing a hash or
pasting a secret into a file the server reads.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .auth.principals import hash_key, mint_key
from .config import get_settings


def _serve() -> int:
    import uvicorn

    from .api.app import ConfigurationError, create_app

    settings = get_settings()
    try:
        create_app(settings)  # fail fast, before binding the port
    except ConfigurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Single worker: the default session store lives in this process, so a
    # second worker would not see sessions created by the first.
    uvicorn.run(
        "model_harness.api.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        workers=1,
    )
    return 0


def _mint_key(principal_id: str, models: list[str] | None, max_tokens: int | None) -> int:
    raw = mint_key()
    entry: dict[str, object] = {"id": principal_id, "key_sha256": hash_key(raw)}
    if models:
        entry["allowed_models"] = models
    if max_tokens is not None:
        entry["max_tokens_per_turn"] = max_tokens

    print("Give this key to the caller. It is shown once and cannot be recovered:\n")
    print(f"  {raw}\n")
    print("Add this entry to the file named by HARNESS_PRINCIPALS_FILE:\n")
    print(json.dumps(entry, indent=2))
    print("\nThe key itself is never stored — only its SHA-256. Losing it means minting a new one.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="model-harness")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("serve", help="Run the HTTP service (default)")

    mint = sub.add_parser("mint-key", help="Generate an inbound API key and its entry")
    mint.add_argument("--id", required=True, help="Principal id, e.g. a team or service name")
    mint.add_argument(
        "--models",
        help="Comma-separated model allowlist. Omit to permit every model.",
    )
    mint.add_argument(
        "--max-tokens-per-turn",
        type=int,
        help="Cap this principal's max_tokens per turn.",
    )

    args = parser.parse_args()

    if args.command == "mint-key":
        models = [m.strip() for m in args.models.split(",")] if args.models else None
        raise SystemExit(_mint_key(args.id, models, args.max_tokens_per_turn))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )
    raise SystemExit(_serve())


if __name__ == "__main__":
    main()
