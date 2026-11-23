"""Where things live. Put this folder in the team repository next to pc-mle/ (as model3-fix/),
or point LSD_REPO at a checkout of https://github.com/Simonjrt3/Learning-Speed-Dating."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("LSD_REPO", HERE.parent)).resolve()
DATA = REPO / "data"
SCRIPT_DIR = REPO / "pc-mle"          # the committed pc_mle_analysis.py is imported unchanged
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
