#run_pipeline.py
#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llama20.modules import (
    module0,
    module_a,
    module_b,
    module_c,
    module_d,
    module_e,
)
from llama20 import loop


def main() -> int:
    print("==> Module 0: model setup")
    module0.run_module0()

    print("==> Module A: dataset build")
    module_a.run_module_a()

    print("==> Module B: activation probing")
    module_b.run_module_b()

    print("==> Module C: signature mining")
    module_c.run_module_c()

    print("==> Module D: capsule forging")
    module_d.run_module_d()

    print("==> Module E: hyper-sentinel runtime")
    module_e.run_module_e_tight()

    print("==> Self-healing loop (Module 7 -> Module 8)")
    history = loop.run_forget_loop()
    if not history:
        print("Self-healing loop produced no history.")
        return 1
    last = history[-1]
    if last.get("reason") == "time_limit":
        print("Self-healing loop stopped due to time limit.")
        return 2

    print("Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
