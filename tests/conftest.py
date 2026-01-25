"""Pytest configuration.

The minimal TP6 codebase is not packaged as an installable distribution. Ensure
that the repo root is on `sys.path` so `import prime_c19` works regardless of
how pytest was invoked.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
