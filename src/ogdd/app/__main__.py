"""Allow OGDD to start with ``python -m ogdd.app``."""

from .application import main


raise SystemExit(main())
