"""Import bridge for executing an individual case file as a script."""

import pathlib
import sys


TEST_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from dsl_e2e.common import E2ECase, case_main  # noqa: E402


__all__ = ["E2ECase", "case_main"]
