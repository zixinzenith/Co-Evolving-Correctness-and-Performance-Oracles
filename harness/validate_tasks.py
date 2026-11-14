"""Validate every task spec: schema completeness, ref runs, output shapes/dtypes,
determinism, and integer outputs for bit_exact tasks.

Usage: python harness/validate_tasks.py  (from repo root). Writes tasks/manifest.json.
Exits non-zero on any failure.
"""

import importlib
import json
import os
import re
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REQUIRED_FIELDS = ["name", "group", "description", "inputs", "outputs", "verification",
                   "shapes", "baseline"]
SHAPE_RE = re.compile(r"^([A-Za-z]\w*|\d+)(?:/([A-Za-z]\w*|\d+))?$")  # K, K/8, K/g, 1, M


def shape_of(spec_expr, shape):
    """Evaluate a shape expression like 'K/8' or 'K/g' against the concrete shape dict."""
    total = 1
    for part in spec_expr.replace(" ", "").split(","):
        if part == "":
            continue  # trailing comma, e.g. "(N,)"
        m = SHAPE_RE.match(part)
        if not m:
            raise ValueError(f"bad shape expr {part!r}")

        def resolve(tok):
            if tok.isdigit():
                return int(tok)
            if tok not in shape:
                raise ValueError(f"unknown shape symbol {tok}")
            return shape[tok]

        base, div = resolve(m.group(1)), resolve(m.group(2)) if m.group(2) else 1
        if base % div:
            raise ValueError(f"{base} not divisible by {div}")
        total *= base // div
    return total


def check_task(group_dir, task_dir):
    errors = []
    spec_path = os.path.join(task_dir, "spec.yaml")
    if not os.path.exists(spec_path):
        return [f"{task_dir}: no spec.yaml"], None
    with open(spec_path) as f:
        spec = yaml.safe_load(f)
    for field in REQUIRED_FIELDS:
        if field not in spec:
            errors.append(f"{spec_path}: missing field {field!r}")

    mod = importlib.import_module(
        f"tasks.{os.path.basename(group_dir)}.{os.path.basename(task_dir)}.impl"
    )
    names_in = [s["name"] for s in parse_io(spec["inputs"])]
    names_out = [s["name"] for s in parse_io(spec["outputs"])]

    for kind in ("visible", "hidden"):
        for shape in spec["shapes"][kind]:
            for seed in (0, 1, 2):
                try:
                    inp = mod.gen_inputs(shape, seed)
                    if set(inp) != set(names_in):
                        errors.append(f"{shape} seed{seed}: gen keys {set(inp)} != spec {set(names_in)}")
                        continue
                    out1 = mod.ref(**inp)
                    out2 = mod.ref(**{k: v.copy() for k, v in inp.items()})
                    if set(out1) != set(names_out):
                        errors.append(f"{shape} seed{seed}: ref keys {set(out1)} != spec {set(names_out)}")
                        continue
                    for o in parse_io(spec["outputs"]):
                        arr = out1[o["name"]]
                        want = shape_of(o["shape"], shape)
                        if arr.size != want:
                            errors.append(f"{shape} seed{seed}: {o['name']} size {arr.size} != {want}")
                        if o["dtype"] == "int32":
                            assert_like = np.int32
                        elif o["dtype"] == "int64":
                            assert_like = np.int64
                        elif o["dtype"] == "int8":
                            assert_like = np.int8
                        elif o["dtype"] == "uint8":
                            assert_like = np.uint8
                        else:
                            assert_like = np.float32
                        if not np.issubdtype(arr.dtype, np.dtype(assert_like).type):
                            errors.append(f"{shape} seed{seed}: {o['name']} dtype {arr.dtype} not {o['dtype']}")
                    # determinism: identical bytes across runs
                    for k in out1:
                        a, b = np.ascontiguousarray(out1[k]), np.ascontiguousarray(out2[k])
                        if a.tobytes() != b.tobytes():
                            errors.append(f"{shape} seed{seed}: {k} nondeterministic")
                    if spec["verification"]["mode"] == "bit_exact":
                        for k, v in out1.items():
                            if not (np.issubdtype(v.dtype, np.integer) or v.dtype == np.float32):
                                errors.append(f"{shape} seed{seed}: bit_exact task returned non-exact dtype {k}")
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{shape} seed{seed}: {type(e).__name__}: {e}")
    entry = {
        "name": spec["name"],
        "group": spec["group"],
        "verification": spec["verification"]["mode"],
        "n_shapes": {"visible": len(spec["shapes"]["visible"]), "hidden": len(spec["shapes"]["hidden"])},
        "expert_baseline": spec["baseline"].get("expert"),
    }
    return errors, entry


def parse_io(s):
    """Parse the compact 'name dtype (shape); ...' io strings, or structured lists."""
    if isinstance(s, list):
        return s
    out = []
    for part in s.split(";"):
        part = part.strip()
        m = re.match(r"(\w+) (\w+) \(([^)]+)\)", part)
        out.append({"name": m.group(1), "dtype": m.group(2), "shape": m.group(3)})
    return out


def main():
    manifest, failed = [], 0
    for group in sorted(os.listdir(os.path.join(ROOT, "tasks"))):
        gdir = os.path.join(ROOT, "tasks", group)
        if not os.path.isdir(gdir) or group in ("common", "__pycache__"):
            continue
        for task in sorted(os.listdir(gdir)):
            tdir = os.path.join(gdir, task)
            if not os.path.isdir(tdir):
                continue
            errors, entry = check_task(gdir, tdir)
            if entry:
                manifest.append(entry)
            if errors:
                failed += len(errors)
                for e in errors:
                    print("FAIL", e)
            else:
                print("ok  ", task)
    with open(os.path.join(ROOT, "tasks", "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\n{len(manifest)} tasks, {failed} errors")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
