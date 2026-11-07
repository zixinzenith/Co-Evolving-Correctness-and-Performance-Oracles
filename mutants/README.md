# mutants/

Hand-written faulty variants of reference kernels (~6-10 per task group): wrong nibble order, wrong scale broadcast, wrong rounding mode, missing clamp/overflow, etc.

Plus a script that runs the verifier against every mutant and reports the detection rate, as an oracle-adequacy measure (cf. arXiv:2609.22220).
