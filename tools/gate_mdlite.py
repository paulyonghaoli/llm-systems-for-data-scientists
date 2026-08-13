"""Gate 29: run the mdLite renderer tests under node.

The tests themselves are `tools/test_mdlite.js`, because the thing under test
is JavaScript and a Python re-implementation would only test the
re-implementation. This wrapper exists so the gate has the same interface as
every other one, and so a missing `node` is reported rather than raising
`FileNotFoundError` out of `tools/verify.py`.

node is already required to deploy this site (`npx wrangler deploy`), so it is
not a new dependency — but the suite should still say plainly when a gate did
not run rather than passing quietly.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    node = shutil.which("node")
    if node is None:
        print("node not found on PATH — the mdLite renderer tests DID NOT RUN.\n"
              "Install node, or run `node tools/test_mdlite.js` by hand.",
              file=sys.stderr)
        # Not a pass. A gate that cannot run has not verified anything, and
        # exiting 0 here would make an absent toolchain look like a green tick.
        return 1

    proc = subprocess.run([node, str(ROOT / "tools" / "test_mdlite.js")],
                          cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
