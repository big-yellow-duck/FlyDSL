# Proposed RDNA4 MXFP4 native-baseline benchmark sweep

**Status: proposal, not executed.** Target: Radeon AI PRO R9700 (`gfx1201`). Kernel source baseline: `3ed2ca5f68ecc2fd972cb23d4da366364e91d261`. Freeze the exact source and installed compiler before running. This plan does not modify runtime selection or add kernel configuration parameters.

## Scope and comparison

For every shape, `A[M,K] × B[K,N] → C[M,N]`. CLI `--nk` is **N×K**, not K×N. Each M list is crossed with every listed N/K pair; separate N and K lists form a full Cartesian product.

- Primary comparison: the existing MXFP4 analytical selection versus native `torch._scaled_mm`, using the same E4M3 activation values and mathematically identical weights. The baseline weights are losslessly decoded to E4M3 before timing. Both produce BF16; activation scales are one.
- MXFP4 retains compressed weights and decodes inside GEMM. Native FP8 stores one byte per weight and excludes decoding/preparation cost. This compares kernel execution, not equal storage footprints, end-to-end serving, or hardware FP4 instructions.
- Current MXFP4 supports positive arbitrary M/N with K divisible by 32. Installed PyTorch `_scaled_mm` checks K and N divisible by 16. Therefore the direct-comparison domain is arbitrary positive M, **N divisible by 16 and K divisible by 32**. Check actual backend acceptance before the full run; record rejection explicitly.
- N tails are measured separately without `--native-fp8`. Do not silently pad them or compare them against a differently sized native operation. A padded-native adapter would be a separately labeled future experiment, not supported by the current script.
- The script fixes E4M3 activations, folded random weights, BF16 output and unit activation scales. E5M2/grouped modes, FP16 output and non-unit scales remain covered by numerical tests; this sweep does not claim to measure them.

## Sweep size

| Cohort | Shapes before cross-cohort deduplication | Coverage |
|---|---:|---|
| A-core | 280 | Decode through large prefill; square, tall and wide matrices. |
| B-m-boundaries | 240 | Adjacent row counts around tile/reuse boundaries, including non-power-of-two M. |
| C-n-boundaries | 240 | Aligned N neighbors around 16/32/64/128/256/512/1024/2048/4096/8192. |
| D-k-factors | 350 | K32 groups, odd K-group factors, long reductions, and neighbors of powers of two. |
| E-cache-footprint | 48 | Compressed B just below/above approximately 8, 16, 32 and 64 MiB. |
| F-reserved | 128 | 128 frozen random shapes, excluded from the structured cohorts; no tuning on these results. |
| G-n-tails | 600 | Unaligned and tiny N; MXFP4-only diagnostic timings, not native speedup comparisons. |

After deduplication: **1,242 native-comparison shapes + 600 MXFP4-only N-tail shapes = 1,842 distinct shapes**. All run with `cache_policy=auto`.
An additional **60 native-comparison shapes** run under both `hot` and `rotating` selection policies. Total: **1,962 shape/policy evaluations per pass**, or **5,886 across three independent passes**. Every evaluation measures both hot and read-evicted timing regimes; a selection policy is distinct from a timing regime.
Expected per pass, if all succeed: **2,724 paired-method JSON rows + 600 MXFP4-only rows = 3,324 rows**. Each row contains 16 event samples per timing regime. Report rejected/failed cases against the planned denominator.

## Exact structured shape sets

### A-core

Decode through large prefill; square, tall and wide matrices.

```text
M = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
N×K = 128x128, 256x256, 512x512, 1024x1024, 2048x2048, 4096x4096, 8192x8192, 16384x16384, 16x16384, 64x16384, 256x16384, 1024x16384, 16384x256, 16384x1024, 4096x1024, 1024x4096, 2048x6144, 6144x2048, 3072x10240, 10240x3072
```

### B-m-boundaries

Adjacent row counts around tile/reuse boundaries, including non-power-of-two M.

```text
M = 15, 17, 31, 33, 47, 63, 65, 95, 96, 97, 127, 129, 191, 192, 193, 255, 257, 383, 384, 385, 511, 513, 1023, 1025, 2047, 2049, 4095, 4097, 8191, 8193
N×K = 256x8192, 1024x8192, 2048x8192, 4096x4096, 5120x17408, 17408x5120, 4096x1056, 4096x8320
```

### C-n-boundaries

Aligned N neighbors around 16/32/64/128/256/512/1024/2048/4096/8192.

```text
M = 1, 8, 73, 192, 1024
N = 16, 32, 48, 112, 128, 144, 240, 256, 272, 496, 512, 528, 1008, 1024, 1040, 2032, 2048, 2064, 4080, 4096, 4112, 8176, 8192, 8208
K = 4096, 8192
```

### D-k-factors

K32 groups, odd K-group factors, long reductions, and neighbors of powers of two.

```text
M = 1, 8, 73, 192, 1024
N = 256, 4096
K = 32, 64, 96, 128, 160, 192, 224, 256, 288, 480, 512, 544, 672, 960, 1024, 1056, 1120, 1568, 2080, 2720, 4064, 4096, 4128, 5856, 8160, 8192, 8224, 8256, 8320, 16352, 16384, 16416, 17408, 18592, 19040
```

### E-cache-footprint

Compressed B just below/above approximately 8, 16, 32 and 64 MiB.

```text
M = 1, 8, 128, 1024
N = 8192
K = 1888, 1920, 1952, 3808, 3840, 3872, 7648, 7680, 7712, 15360, 15392, 15424
```

### G-n-tails

Unaligned and tiny N; MXFP4-only diagnostic timings, not native speedup comparisons.

```text
M = 1, 8, 73, 192, 1024
N = 1, 7, 15, 17, 31, 33, 63, 65, 127, 129, 255, 257, 511, 513, 1023, 1025, 2047, 2049, 4095, 4097
K = 32, 160, 1120, 4096, 8256, 19040
```

For cohort E, compressed weight storage is `ceil(N/16)*16 * (17*(K/32) + 4)` bytes. For N=8192, the listed K triplets straddle approximately 8/16/32/64 MiB. The predecoded baseline has a larger footprint and may cross cache boundaries at different shapes; this is an intentional part of the storage tradeoff.

### F-reserved: exact frozen arbitrary shapes

Generated once with Python `random.Random(137420261116)`: eight unique N/K pairs for each of 16 M values; N=16×uniform integer [1,1024], K=32×uniform integer [1,595]. Reject overlap with A–E and earlier reserved triples. The explicit list below is authoritative. These are reserved for this proposed run; they are not guaranteed absent from every historical development dataset.

<details>
<summary>Expand all 128 reserved (M, N, K) triples</summary>

```text
(1, 14592, 3168)  (1, 13680, 352)  (1, 14080, 16704)  (1, 6608, 6112)
(1, 4032, 15264)  (1, 672, 800)  (1, 10112, 6208)  (1, 3344, 5856)
(3, 2640, 17792)  (3, 14768, 17280)  (3, 9232, 3168)  (3, 464, 7424)
(3, 14496, 12640)  (3, 8192, 17728)  (3, 4256, 4576)  (3, 8160, 1312)
(7, 13904, 17536)  (7, 1968, 2752)  (7, 3216, 7040)  (7, 13584, 256)
(7, 14624, 12832)  (7, 15952, 2208)  (7, 2016, 800)  (7, 1664, 15392)
(13, 15552, 256)  (13, 14656, 16640)  (13, 2336, 3392)  (13, 11680, 8864)
(13, 12560, 7136)  (13, 13728, 14624)  (13, 13584, 480)  (13, 8112, 7072)
(29, 7312, 3008)  (29, 2368, 7296)  (29, 2176, 7968)  (29, 1808, 14624)
(29, 992, 2944)  (29, 15856, 3360)  (29, 10448, 5888)  (29, 11520, 10208)
(47, 8576, 14912)  (47, 8688, 16512)  (47, 15408, 18400)  (47, 5776, 192)
(47, 10640, 6784)  (47, 1936, 288)  (47, 6400, 2304)  (47, 14144, 16128)
(73, 15392, 7584)  (73, 15408, 18208)  (73, 16352, 7808)  (73, 2976, 11968)
(73, 10000, 13952)  (73, 14288, 6432)  (73, 5456, 12992)  (73, 5280, 15584)
(113, 13936, 13952)  (113, 6688, 18624)  (113, 6048, 14880)  (113, 8208, 9792)
(113, 9392, 7168)  (113, 14960, 9760)  (113, 1744, 15520)  (113, 32, 14144)
(191, 7968, 9184)  (191, 15280, 2816)  (191, 2480, 17600)  (191, 64, 13248)
(191, 12240, 4032)  (191, 4336, 16288)  (191, 5744, 4288)  (191, 5184, 9792)
(257, 14704, 8128)  (257, 10960, 928)  (257, 5920, 10976)  (257, 14320, 11904)
(257, 10944, 9760)  (257, 5120, 6784)  (257, 4608, 16832)  (257, 10256, 14496)
(509, 15200, 2272)  (509, 15248, 15712)  (509, 10512, 14464)  (509, 3072, 17568)
(509, 8112, 8960)  (509, 10640, 18560)  (509, 9696, 4864)  (509, 9200, 18944)
(1009, 14320, 4640)  (1009, 3296, 1568)  (1009, 7424, 8320)  (1009, 1744, 12096)
(1009, 1984, 3968)  (1009, 7536, 1920)  (1009, 9776, 5056)  (1009, 4688, 1056)
(2053, 4672, 12736)  (2053, 4096, 12032)  (2053, 1440, 5792)  (2053, 320, 16160)
(2053, 4624, 13376)  (2053, 9408, 3232)  (2053, 3344, 16928)  (2053, 14848, 3552)
(4093, 4160, 13920)  (4093, 5904, 5056)  (4093, 14720, 6752)  (4093, 720, 5888)
(4093, 11920, 11936)  (4093, 11696, 6752)  (4093, 3536, 13152)  (4093, 7520, 9920)
(6143, 15824, 1600)  (6143, 11392, 6880)  (6143, 4768, 2624)  (6143, 10304, 4960)
(6143, 12736, 6624)  (6143, 11888, 18272)  (6143, 13536, 9760)  (6143, 12752, 13792)
(8191, 10080, 2496)  (8191, 5312, 3328)  (8191, 9552, 9152)  (8191, 4816, 4864)
(8191, 13792, 2912)  (8191, 11408, 5952)  (8191, 6928, 10400)  (8191, 6016, 10368)
```

</details>

### Selection-policy sensitivity

Use all 48 E-cache-footprint shapes plus these 12 triples; repeat them with `--cache-policy hot` and `--cache-policy rotating`, in addition to their primary `auto` measurements:

```text
(1, 4096, 4096)  (8, 4096, 4096)  (15, 5120, 17408)  (17, 5120, 17408)
(95, 2048, 8192)  (97, 2048, 8192)  (128, 4096, 4096)  (191, 2048, 8192)
(193, 2048, 8192)  (257, 4096, 8320)  (1024, 4096, 4096)  (1025, 4096, 8320)
```

## Run protocol

1. Freeze the shape manifest, source commit, compiler/binary identity, Torch/ROCm versions, GPU name, visible-device mapping and reported processor/cache/LDS capacities. On R9700, verify AMD SMI exposes the 64 MiB last-level data cache; an 8 MiB L2 fallback changes selector behavior and must be a separately labeled run.
2. Preflight `--native-fp8` at `(1,16,32)`, `(17,48,160)`, `(73,528,1120)` and `(192,2048,8192)`. Test MXFP4-only `(1,1,32)` and `(73,513,1120)`. If a native shape is rejected, record unsupported; if correctness fails, investigate before extending the run. These checks are not timings for the final summary.
3. Run one measurement process on one otherwise idle GPU. Use three serial full passes, rotating job and M order with recorded seeds 1374, 1375 and 1376. Do not mix another benchmark or serving load onto that GPU. Keep clock/power settings fixed and record thermal conditions; do not silently change system settings.
4. Group one N/K pair and one selection policy per process, supplying its deduplicated M list. This limits compiler/cache accumulation and avoids a fresh process for every M. Save separate JSON and log files per job and pass. Track cohort membership separately so overlapping shapes are timed once.
5. Use the existing 16 randomized paired event rounds, captured graph replay, and hot/read-evicted protocols. Compilation, preparation and the 256 MiB conditioning buffer run outside measured events. The rotating regime means a cache-conditioning proxy, not actual model-layer cycling; there are no DRAM-counter claims.
6. Every candidate must pass the script's independent sampled FP32 reference (up to 32×64 entries, relative RMS ≤0.005) and changed-input graph replay check. Also run the full-output numerical kernel tests once for the frozen revision. Do not describe the benchmark's sampled comparison as full-output verification.
7. Keep OOMs, compile failures, native unsupported shapes and numerical failures in the results. A partially written JSON does not mean the job completed. Save exit status and compare observed rows with the expected manifest before aggregation.
8. Begin with A–E, then run F without retuning based on its results. Report G separately. If measurements motivate selector changes, freeze a new revision and rerun; do not mix measurements from different implementations.

## Commands and execution manifest

Run from the FlyDSL checkout with the matching compiler installed or built. Set `PYTHONPATH` to the checkout root; for a source runtime also include `python/`. Configure `ROCM_PATH`, `FLYDSL_GPU_ARCH=gfx1201` and AMD SMI as required by the environment. The following are examples; the shape sets above define the full sweep.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python tests/perf/bench_rdna4_mxfp4.py --native-fp8 --gpu 0 \
  --m 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 \
  --nk 4096x4096 --cache-policy auto --output results/pass1/core-4096x4096.json
python tests/perf/bench_rdna4_mxfp4.py --native-fp8 --gpu 0 \
  --m 1 8 128 1024 --nk 8192x15392 --cache-policy rotating \
  --output results/pass1/cache-8192x15392-rotating.json
python tests/perf/bench_rdna4_mxfp4.py --gpu 0 \
  --m 1 8 73 192 1024 --nk 513x1120 --cache-policy auto \
  --output results/pass1/tail-513x1120.json
python -m pytest -q tests/kernels/test_rdna4_mxfp4.py
```

Each planned job should record `{pass, native_baseline, policy, N, K, M_list, cohorts, command, output_path, exit_code}`. Native jobs use `--native-fp8`; tail jobs omit it. Run the script unchanged: no new runtime flags or selector tests are needed.

The largest direct-comparison triple is not a memory-capacity prediction: native-baseline preparation temporarily materializes int64 codes and FP32 weights in addition to the compressed and FP8 buffers. Profile peak allocation during the first largest-shape job. If it exceeds the available budget, record the failure and execute a declared reduced-memory cohort; do not silently shrink shapes. Each input/output buffer stays below the kernel's 4 GiB addressing limit in this proposal.

## Reporting and decision rules

- Publish the shape manifest, source/runtime identities, all raw `samples_us`, logs, exit-status ledger, and a per-shape CSV. Preserve every pass rather than overwriting results.
- For each matched shape/policy/pass/regime compute **speedup = native-FP8 median latency / MXFP4 median latency**. A value above one favors MXFP4. Combine the three pass ratios using their median; also report all three values and their spread.
- Report hot and read-evicted results separately. Use an equal-weight per-shape geometric mean for overall speedup, plus per-cohort and M-band summaries (M≤16, 17–128, 129–1024, M>1024). Deduplicate the overall denominator; overlapping cohorts may each show the same shape with that overlap disclosed.
- Publish worst cases, win/loss counts, absolute microseconds, and latency ratios; do not show only aggregate speedup. Mark a material win/loss only if all three pass ratios exceed 1.05 / fall below 0.95 respectively. Treat the rest as small or inconsistent differences, not proof of equivalence.
- Tail cohort G reports MXFP4 latency/correctness only. Native unsupported shapes have no speedup ratio. OOMs and failures remain visible. Do not add them to a success-only performance aggregate without reporting excluded counts.
- Logical GB/s and TFLOP/s are derived metrics, not measured DRAM traffic. Report compressed-vs-expanded weight sizes alongside latency.
- Do not estimate completion time from the eight-shape pilot: first measure the preflight/core throughput and compilation cost, then publish an ETA for the remaining jobs.
