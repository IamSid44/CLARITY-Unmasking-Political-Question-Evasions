"""Repository paths, defined once. `CLARITY_RUNS` overrides where runs are read and written."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

ROOT: Final[Path] = Path(__file__).resolve().parents[2]
RUNS: Final[Path] = Path(os.environ.get("CLARITY_RUNS", ROOT / "runs"))
LOGS: Final[Path] = ROOT / "logs"
DATA: Final[Path] = ROOT / "data"
DATA_CACHE: Final[Path] = DATA / "cache"
SPLITS: Final[Path] = DATA / "splits"
TEST_CSV: Final[Path] = DATA / "clarity_task_evaluation_dataset.csv"
DOCS: Final[Path] = ROOT / "docs"
RAW: Final[Path] = DOCS / "raw"
FIGURES: Final[Path] = DOCS / "figures"
REPORT_FIGURES: Final[Path] = ROOT / "Report" / "latex" / "figures"
