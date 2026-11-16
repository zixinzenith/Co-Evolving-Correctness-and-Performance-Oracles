"""Anti-cheating checks for generated kernels.

Phase 0 scope (CPU, no GPU timing yet):

- Forbidden-import scan (AST): the candidate may not import the oracle stack
  (`tasks.common`, `verifier`, `harness`) or call the reference module directly. Any
  additional per-task banned APIs (e.g. a "no torch.matmul" condition) can be passed in.
- Output-aliasing check: outputs must not be (views of) the input arrays — catches
  "implementations" that return their inputs unchanged.
- Hidden-shape evaluation is the structural anti-cheat: every candidate is evaluated on
  hidden shapes it never saw in the prompt (see harness/run.py), so hard-coded outputs
  cannot survive.

PyTorch-fallback detection (detecting that a "kernel" just calls torch.matmul on a fully
dequantized weight matrix) is a Phase 1 measurement concern: on GPU we compare against the
expert baselines' runtime rather than banning eager fallbacks, so it lands with the
performance harness.
"""

import ast


DEFAULT_FORBIDDEN = ("tasks.common", "verifier", "harness", "importlib", "ctypes",
                     "subprocess", "os.system", "eval(", "exec(")


def scan_source(src, forbidden=DEFAULT_FORBIDDEN):
    """Return the list of forbidden tokens found in candidate source (via AST + raw scan)."""
    violations = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    for token in forbidden:
        tok = token.rstrip("(")
        if tok in imported or token in src:
            violations.append(token)
    return sorted(set(violations))
