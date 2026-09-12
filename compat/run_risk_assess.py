#!/usr/bin/env python3
"""
Launcher that installs the legacy PyG compatibility shim, then runs the
original, unmodified tools/risk_assess.py.

roadscene2vec is never touched: `git status --short` in that repo should
stay empty before and after using this launcher.
"""
import sys
from pathlib import Path

GBSED_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GBSED_ROOT))              # for `compat` package + gbsed_semantic
sys.path.insert(0, str(GBSED_ROOT / "tools"))    # to `import risk_assess` as a module
sys.path.insert(0, "/home/opp_env/default_workspace/roadscene2vec")

from compat.pyg_legacy import install as install_pyg_legacy_compat
install_pyg_legacy_compat()  # must happen before risk_assess (and mrgcn) is imported

import risk_assess  # noqa: E402 -- byte-for-byte original file

if __name__ == "__main__":
    risk_assess.main()