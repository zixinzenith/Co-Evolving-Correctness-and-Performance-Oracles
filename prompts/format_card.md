# Condition 2 — + format spec card

The baseline prompt plus the full text of the format spec the task was built from
(`spec.format_ref` → `formats/*.md`): the exact packing layouts, rounding modes, zero-point
conventions and saturation behavior, with sha256-pinned sources.

This is the cheapest intervention and the most targeted: every bug axis in the spec's
`adversarial_classes` is documented in the card. The question (RQ3) is whether reading the
contract moves models past the "numerical contract" failures KernelBenchX reported.
