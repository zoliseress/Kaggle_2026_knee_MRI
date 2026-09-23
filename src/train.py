"""Convenience entry point: `python src/train.py --mode synthetic`.

Equivalent to `python -m knee_mri.cli train ...` when run from the `src` directory.
The main guard matters on Windows, where DataLoader workers re-import this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from knee_mri.cli import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main(["train", *sys.argv[1:]]))
