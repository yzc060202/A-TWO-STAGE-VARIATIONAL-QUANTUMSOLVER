"""Single-process wrapper for one cross-PDE Stage-I -> Stage-II run."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_parallel_crosspde_twostage import parse_args, project_root, run_pde, status


def main() -> int:
    ns = parse_args()
    if len(ns.pdes) != 1:
        raise SystemExit("run_single_crosspde_twostage.py requires exactly one --pdes value")
    phase = Path(ns.phase) if ns.phase else project_root() / "results" / "single_crosspde_twostage"
    if not phase.is_absolute():
        phase = project_root() / phase
    phase.mkdir(parents=True, exist_ok=True)
    args = vars(ns)
    args["phase"] = str(phase)
    status(phase / "parallel_status.json", state="running", pdes=list(ns.pdes), workers=1, runner="single_process")
    row = run_pde(ns.pdes[0], str(phase), args)
    status(phase / "parallel_status.json", state=row.get("state", "failed"), pdes=list(ns.pdes), workers=1, runner="single_process")
    print(row)
    return 0 if row.get("state") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
