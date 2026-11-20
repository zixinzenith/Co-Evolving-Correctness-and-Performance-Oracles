# Condition 3 — + bit-exact unit-test feedback

The baseline prompt plus, from iteration 2 on, the verifier's failures from the previous
attempt rendered as concise lines: which output, first differing flat index, expected vs
got vs bound. Bit-exact tasks effectively become unit-test-driven development with an
oracle that does not blink; derived-tolerance tasks report the violating element and the
format-derived bound.

The intervention being tested: does a *precise* oracle (vs the field's loose allclose)
change the failure mode at all?
