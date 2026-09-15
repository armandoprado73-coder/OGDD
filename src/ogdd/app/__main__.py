"""Allow OGDD to start with ``python -m ogdd.app``."""

from ogdd.app.application import main


raise SystemExit(main())
