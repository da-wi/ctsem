# Validation of the ordered beta measurement model (manifesttype = 2)

- `ordbeta_filter.py` – filter mathematics: adaptive Gauss-Hermite update vs. dense-grid reference
  (single step) and vs. a 40k-particle filter (full likelihood).
- `test_regression.R <lib> <out.rds>` – gaussian and binary models; run with original and extended
  package, results must be identical.
- `test_ordbeta_recovery.R <lib> <seed> <out.rds>` – group ESM simulation (80 persons x 42 beeps,
  ~50% exact zeros), parameter recovery, plus naive gaussian model for comparison.
- `test_ordbeta_outputs.R <lib> <rec1.rds>` – Kalman predictions on the 0-1 scale and data generation.
- `test_mixed.R <lib>` – one latent process with an ordered beta and a gaussian indicator.

Results (Nov 2026, 13 replications): no detectable bias (all |bias| < 1 Monte Carlo SE);
Hessian SE of drift 0.095 vs empirical SD 0.12; gaussian model drift SD 0.20 vs 0.12 (ordered beta).
Gaussian and binary results bit-identical to ctsem 3.11.1.
