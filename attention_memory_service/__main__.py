"""Start the authenticated Memory Service HTTP daemon."""

from .memory_service_api import main

if __name__ == "__main__":
    raise SystemExit(main())
