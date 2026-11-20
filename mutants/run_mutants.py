"""Run every mutant against the verifier and report per-mutant detection rates.

Detection rate = fraction of (shape, seed) trials where the oracle rejects the mutant.
Bit-exact tasks should sit at 1.00. An escape on a derived-tolerance task means the planted
fault slips under the format-derived bound — the oracle-adequacy signal that KernelBench-M
(arXiv:2609.22220) formalized, and the reason this benchmark mutation-tests its own oracles.

Usage: make mutants   (from repo root)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from verifier import check  # noqa: E402
from mutants.variants import MUTANTS  # noqa: E402


def main():
    caught_total = trials_total = 0
    escapes = []
    for (group, task), mutants in sorted(MUTANTS.items()):
        spec, impl = check.load_task(group, task)
        shapes = spec["shapes"]["visible"] + spec["shapes"]["hidden"]
        for name, fn in mutants:
            caught = trials = 0
            for shape in shapes:
                for seed in (0, 1, 2):
                    trials += 1
                    inputs = impl.gen_inputs(shape, seed)
                    ref_out = impl.ref(**inputs)
                    try:
                        cand_out = fn(**inputs)
                        ok, _ = check.check_outputs(spec, impl, inputs, ref_out, cand_out)
                        detected = not ok
                    except Exception:  # noqa: BLE001 — a mutant that crashes cannot ship
                        detected = True
                    caught += detected
                    trials_total += 1
            caught_total += caught
            flag = "" if caught == trials else "  <-- oracle escape"
            if caught < trials:
                escapes.append((group, task, name))
            print(f"{group}/{task:34s} {name:28s} {caught:3d}/{trials}{flag}")
    print(f"\noverall detection: {caught_total}/{trials_total}")
    if escapes:
        for e in escapes:
            print("escape:", "/".join(e))
    sys.exit(1 if escapes else 0)


if __name__ == "__main__":
    main()
