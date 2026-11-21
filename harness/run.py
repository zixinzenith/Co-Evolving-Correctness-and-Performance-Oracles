"""Run the generation loop over tasks x conditions and record one JSONL entry per attempt
series to runs/.

Phase 0 driver — the fake models exercise the full pipeline (prompt building, execution,
verifier feedback, hidden-shape evaluation) without any API spend. Phase 1 swaps the model
callable for the real API clients and keeps everything else fixed.

Usage:
    python harness/run.py --task quant_pack/int4_quantize_asym --condition unittest_feedback
    python harness/run.py --group kv_cache --conditions all --model fake_buggy
"""

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent import fake_model, loop  # noqa: E402
from prompts.build import CONDITIONS  # noqa: E402
from verifier import check  # noqa: E402
from verifier.anticheat import DEFAULT_FORBIDDEN, scan_source  # noqa: E402

MODELS = {
    "fake": fake_model.good,
    "fake_buggy": fake_model.buggy_then_good,
    "fake_broken": fake_model.always_buggy,
}


def eval_hidden(src, spec, impl):
    """Structural anti-cheat: evaluate the final source on shapes it never saw."""
    violations = scan_source(src, DEFAULT_FORBIDDEN)
    if violations:
        return False, [f"forbidden import/call: {v}" for v in violations]
    try:
        kernel = loop._compile_kernel(src)
    except Exception as e:  # noqa: BLE001
        return False, [f"{type(e).__name__}: {e}"]
    failures = []
    for shape in spec["shapes"]["hidden"]:
        for seed in (0, 1, 2):
            inputs = impl.gen_inputs(shape, seed)
            try:
                cand = kernel(**inputs)
            except Exception as e:  # noqa: BLE001
                failures.append(f"hidden {shape} seed{seed}: {type(e).__name__}: {e}")
                continue
            ref = impl.ref(**inputs)
            ok, f = check.check_outputs(spec, impl, inputs, ref, cand)
            failures.extend(f"hidden {shape} seed{seed}: {x}" for x in f)
    return not failures, failures


def run_one(task_spec, condition, model_name):
    group, name = task_spec
    spec, impl = check.load_task(group, name)
    t0 = time.time()
    attempts, final = loop.run_task(MODELS[model_name], spec, impl, condition)
    hidden_ok, hidden_failures = eval_hidden(final.source, spec, impl)
    return {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "task": f"{group}/{name}",
        "group": group,
        "condition": condition,
        "model": model_name,
        "iterations": len(attempts),
        "passed_visible": final.ok,
        "passed_hidden": hidden_ok,
        "first_failures": (attempts[0].failures or hidden_failures)[:3],
        "seconds": round(time.time() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", action="append", default=[],
                    help="group/name, repeatable")
    ap.add_argument("--group", action="append", default=[],
                    help="run every task in a group dir, repeatable")
    ap.add_argument("--condition", default="baseline", choices=CONDITIONS)
    ap.add_argument("--model", default="fake", choices=sorted(MODELS))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "records.jsonl"))
    args = ap.parse_args()

    tasks = [(t.split("/")[0], t.split("/")[1]) for t in args.task]
    for g in args.group:
        gdir = os.path.join(ROOT, "tasks", g)
        tasks.extend((g, d) for d in sorted(os.listdir(gdir))
                     if os.path.isdir(os.path.join(gdir, d)))
    if not tasks:
        ap.error("no tasks given (use --task or --group)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_pass = 0
    with open(args.out, "a") as out:
        for group, name in tasks:
            rec = run_one((group, name), args.condition, args.model)
            n_pass += rec["passed_visible"] and rec["passed_hidden"]
            print(f"{rec['task']:44s} {rec['condition']:18s} "
                  f"visible={'PASS' if rec['passed_visible'] else 'FAIL':4s} "
                  f"hidden={'PASS' if rec['passed_hidden'] else 'FAIL':4s} "
                  f"iters={rec['iterations']}")
            out.write(json.dumps(rec) + "\n")
    print(f"\n{n_pass}/{len(tasks)} fully passed -> {args.out}")


if __name__ == "__main__":
    main()
