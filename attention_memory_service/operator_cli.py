"""Human-operator lifecycle controls for the independent Memory Service."""

from __future__ import annotations

import argparse
import json
import os

from .memory_service_client import MemoryServiceClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-url", default="http://127.0.0.1:8768")
    parser.add_argument("--actor", required=True, help="human operator identifier")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("disable", "rollback", "set-expiry"):
        command = commands.add_parser(name)
        command.add_argument("memory_id")
        command.add_argument("--reason", required=True)
        if name == "set-expiry":
            command.add_argument("--expires-at", type=float, required=True, help="Unix timestamp in seconds")
    args = parser.parse_args()
    client = MemoryServiceClient(
        args.service_url,
        api_key=os.environ.get("ATTENTION_MEMORY_API_KEY", ""),
        operator_key=os.environ.get("ATTENTION_MEMORY_OPERATOR_KEY"),
    )
    if args.command == "disable":
        memory = client.disable(args.memory_id, actor=args.actor, reason=args.reason)
    elif args.command == "rollback":
        memory = client.rollback(args.memory_id, actor=args.actor, reason=args.reason)
    else:
        memory = client.set_expiry(
            args.memory_id, expires_at=args.expires_at,
            actor=args.actor, reason=args.reason,
        )
    print(json.dumps(memory.artifact(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
