"""End-to-end tests of the generation loop with the scripted fake model."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from agent import fake_model, loop  # noqa: E402
from verifier import check  # noqa: E402


def _task():
    return check.load_task("quant_pack", "int4_quantize_asym")


def test_good_model_passes_on_first_iteration():
    spec, impl = _task()
    history, final = loop.run_task(fake_model.good, spec, impl, "baseline")
    assert final.ok
    assert len(history) == 1


def test_feedback_loop_recovers_after_planted_bug():
    spec, impl = _task()
    history, final = loop.run_task(fake_model.buggy_then_good, spec, impl, "unittest_feedback")
    assert final.ok
    assert len(history) == 2
    assert not history[0].ok  # first attempt carried the half-up rounding bug
    assert any("halfway" in "\n".join(history[0].failures) or "mismatch" in f
               for f in history[0].failures)


def test_always_buggy_exhausts_iteration_budget():
    spec, impl = _task()
    history, final = loop.run_task(fake_model.always_buggy, spec, impl, "baseline")
    assert not final.ok
    assert len(history) == loop.MAX_ITERS


def test_oracle_import_is_rejected():
    spec, impl = _task()

    def cheater(prompt, history):
        return ("import numpy as np\nfrom tasks.common import ops\n"
                "def kernel(**inp):\n    return ops.quantize_int4_asym("
                "inp['w'], inp['scales'], inp['zeros'])\n")

    history, final = loop.run_task(cheater, spec, impl, "baseline")
    assert not final.ok
    assert any("forbidden" in f for f in final.failures)
