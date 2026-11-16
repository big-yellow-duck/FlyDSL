# E4M3 versus E5M2 folded MXFP4 weights

Measured on R9700 / gfx1201 GPU 0, 16 November 2026, after the local cache-policy removal. This compares the FP8 format used to represent decoded MXFP4 weight fragments; activations remain E4M3 and output remains BF16.

**The distinction is primarily exponent coverage.** Both formats can exactly represent the E2M1 significands scaled by powers of two when their range permits it. E4M3's extra mantissa precision adds no weight precision for these inputs. E5M2 handles a wider scale span. Keeping a losslessness check and grouped fallback remains necessary; this bounded performance sample informed the later choice of E5M2 as the only folded format.

## Paired performance

Speedup is E4M3 latency / E5M2 latency; above one favors E5M2. The aggregate is the geometric mean of each case's median three-pass ratio.

| E8M0 exponent span | Shapes | E4M3 normal-only decode | Hot speedup | Read-evicted speedup |
| --- | ---: | --- | ---: | ---: |
| 7 | 12 | Yes | 0.9942× | 0.9960× |
| 12 | 4 | No | 1.0258× | 1.0021× |
| 14 | 4 | No | 1.0306× | 1.0012× |

For the ordinary span-7 weights, E5M2 is about 0.6% slower hot and 0.4% slower read-evicted in this sample. Individual hot median ratios range from about 0.980× to 1.009×. This is a small measured difference, not proof of universal equivalence. For span 12/14, E5M2's normal-only ALU path avoids the E4M3 subnormal corrections; the two ALU register cases show roughly 4.4–6.6% hot gains. The LUT cases remain close. This is consistent with the source paths, but no ISA or hardware-counter profiling was performed to establish causality.

All 60 paired case/pass evaluations produced **bitwise-identical full BF16 outputs** across E4M3 and E5M2. Both formats passed sampled independent FP32 references and changed-input graph replay. Maximum sampled reference relative RMS was 1.0942563e-06.

## Range boundaries

Probes use every FP4 code, maximum E8M0 exponent byte 131, and the stated difference between maximum and minimum exponent bytes within an output channel. All accepted modes passed exact full-output reference comparisons. Folded probes also checked exact FP8 representability independently of kernel decoding.

| Exponent span | Forced E4M3 | Forced E5M2 | Auto choice at measurement |
| --- | --- | --- | --- |
| 7 | Exact | Exact | e4m3 |
| 11 | Exact | Exact | e4m3 |
| 12 | Exact | Exact | e4m3 |
| 14 | Exact | Exact | e4m3 |
| 15 | Rejected | Exact | e5m2 |
| 26 | Rejected | Exact | e5m2 |
| 27 | Rejected | Exact | e5m2 |
| 28 | Rejected | Exact | e5m2 |
| 29 | Rejected | Rejected | grouped |

The 14/28 span limits describe these ordinary exponent ranges. The implementation also clamps the reference exponent and rejects reserved exponent values, so actual eligibility remains the losslessness predicate in `prepare_weights`, not a span-only shortcut. Beyond E5M2's folding range, grouped K32 scaling is still required for arbitrary weights.

## Method and scope

Twenty cases: twelve distinct shapes at exponent span 7, plus four selected shapes each at spans 12 and 14. Shapes cover register/hybrid, ALU/LUT, decode/prefill, N tails, and the previous M8 outlier. Three passes used the same deterministic synthetic data and identical configs for both formats, randomized case/method order, and 16 paired GPU-event rounds per cache condition. Hot measurements use eight calls for M<128 and two otherwise. Read-evicted measurements read 256 MiB and activations outside timing, then time one call. Preparation, compilation and correctness checks are outside timing. All pairs run on the same GPU. No clock or power settings were changed; this experiment does not include synchronized hardware-counter profiling.

This is a targeted format comparison, not a repeat of the 1,842-shape sweep, exhaustive scale testing, or a full-model quality evaluation. The format-selection code was held fixed during this comparison. PR #1374 subsequently adopted E5M2 as the single folded format plus grouped fallback in `495bb86`. The full 1,842-shape sweep has not been repeated on that revision.

## Reproduction

This comparison requires the earlier source that supported both folded formats. The [measured kernel source](measured-kernel-source/) is preserved here and every file matches the SHA-256 recorded in [provenance](provenance.json). Use a separate workspace with FlyDSL checkout `3ed2ca5` at `FlyDSL-pr1374-review/`, its matching built runtime, and this report directory at `reports/flydsl-weight-format-20261116/`. Restore the measured host cleanup before running:

```bash
cp reports/flydsl-weight-format-20261116/measured-kernel-source/*.py \
  FlyDSL-pr1374-review/kernels/gemm/rdna4_mxfp4/
```

Then run from that workspace root:

```bash
ROCM_PATH=$(.venv/bin/rocm-sdk path --root)
export ROCM_PATH
export PYTHONPATH="$PWD/FlyDSL-pr1374-review/python:$PWD/FlyDSL-pr1374-review:$ROCM_PATH/share/amd_smi"
export FLYDSL_GPU_ARCH=gfx1201 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python reports/flydsl-weight-format-20261116/compare_formats.py
.venv/bin/python reports/flydsl-weight-format-20261116/summarize_formats.py
```

Re-running overwrites this experiment's output files. [Source/environment hashes](provenance.json), [raw samples](results.json), [paired CSV](paired.csv), [boundary checks](boundaries.json), and [summary](summary.json) are retained locally.
