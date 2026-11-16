# vLLM MXFP4 dispatch and benchmark methods

## Dispatch finding

The cached `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` config (revision `93245f7c00f1a8964b66a97c6e02e2fd09080be4`, config SHA-256 `4a139d3e01df039e17b4c8b8362f7944674d76051a8cd96f213c6bc332d69f55`) declares static MXFP4 weights and dynamic, group-32 MXFP4 activations. This is W4A4. The text model uses dense linears; its config contains no expert-count field. Vision layers are excluded from quantization.

Historical and current dispatch need separate labels:

- **Historical default before FlyDSL:** the FlyDSL backend was added by commit `9662fa0a5c3f33c211a5702654f00fd7fb6b6ce2`; its parent, `335dd77b09161c5d63139b4e7ae51bb9928305b1`, has the ROCm candidate order RDNA3 W4A8, AITER MXFP4, then emulation. RDNA3 rejected gfx1201 and AITER required native MX support unavailable on gfx1201, so the checkpoint’s unchanged MXFP4 activation scheme selected `EmulationMxfp4LinearKernel` (W4A4), assuming Quark was installed. This is established from source history; the historical server binary was not executed.
- **Current default:** commit `4816c7b43f52a09efd09409e6a023b5dd31579f5` added an RDNA4-specific activation substitution. On gfx1200/gfx1201, auto selection maps the checkpoint’s MXFP4-dynamic activation to per-token FP8 for the RDNA4 candidate, then selects `Rdna4Mxfp4LinearKernel` when its module is importable. That is W4A8. The preflight called the current `init_mxfp4_linear_kernel(kMxfp4Dynamic)` and observed this dispatch; the W4A4 arm was explicitly selected through the per-quant `emulation` backend to reproduce the historical arithmetic.

The arithmetic is different even though both arms read the same packed MXFP4 weights:

- W4A4 emulation quantizes and dequantizes BF16 activations to MXFP4, decodes MXFP4 weights using E8M0 group scales, and calls `F.linear`. With `VLLM_MXFP4_EMULATION_DEQUANT_AT_LOAD=0`, weight decode happens on each `apply_weights` call and stays inside the timed GPU work.
- W4A8 FlyDSL prepares/reorders the packed weights once, dynamically quantizes each activation row to FP8 E4M3 with an FP32 scale, decodes packed E2M1 weights to FP8 fragments, accumulates the GEMM in FP32, applies scales, and casts to BF16/FP16.

Therefore compare each backend against its own quantization reference. The W4A4-to-W4A8 output delta is an activation-quantization difference, not solely a GEMM implementation error. Preflight RMS metrics and detailed source references are in `preflight-gpu1.json` and `evidence.json`.

## Harness and execution protocol

`bench_mxfp4_pair.py` builds separate layer objects from identical packed weights/scales and calls vLLM’s real selector and `apply_weights`. FlyDSL post-load preparation occurs once outside timing; the default W4A4 weight dequantization and both activation quantizers remain in every timed call. Each output is checked against its own reference, using an independent OCP E2M1 LUT decode for sampled weights. Cross-backend error is recorded separately. Captured hot and read-evicted graphs use 16 event-timed rounds; read-evicted replay reads a 256 MiB buffer and activation tensor first. The final loop queues conditioning and measured replay on the same stream, then synchronizes after the measured replay. CUDA-event timings exclude Python dispatch and host allocation overhead.

The benchmark adapter sets `flydsl.kernels` to the checked-out `FlyDSL-pr1374-review/kernels` package because this vLLM checkout imports that namespace while FlyDSL source exposes `kernels.gemm`. `FlyDSL-pr1374-review/python` must precede the checkout root in `PYTHONPATH`, and the matching ROCm binary path, `CXX`, and `share/amd_smi` path are required. Exact preflight environment values are in `preflight-provenance.json`. This source adapter does not prove stock vLLM packaging installs FlyDSL at that namespace.

The sweep uses synthetic packed weights and activations for all shapes; no full checkpoint tensor or model forward is loaded. The preflight also used synthetic weights at the model’s `N=17408,K=5120` projection shape. It passed six paired shapes and two FlyDSL-only cases. The preflight timing values were captured before the final same-stream synchronization cleanup and are protocol-debug output, not final performance evidence; use sweep outputs from the frozen harness for performance conclusions.

## Shape coverage and emulation limits

The frozen plan contains 1,842 unique `(M,N,K)` shapes. Based on Quark source and runtime errors, a paired W4A4 run is valid only when both:

- Activation QDQ: `M*K % 64 == 0`.
- Weight dequantization: `N*K % 512 == 0`.

Of the plan, 1,158 shapes satisfy both conditions and are expected paired cases. Across three passes, that is 3,474 paired shape evaluations. The other 684 remain explicit `emulation_unsupported_flydsl_ok` rows with a Quark rejection string and FlyDSL-only timings; across three passes that is 2,052 one-arm evaluations. The total plan is 5,526 shape-evaluation rows. Do not assign the one-arm rows a W4A4/W4A8 speedup. Do not pad or otherwise alter the logical dimensions. The 34 generated batch manifests partition the 1,842 shapes without overlap: 808 are assigned to GPU0 and 1,034 to GPU1, each across three passes.

These Quark limits are stricter than the kernels’ shared basic layout checks (`N>0`, `K>0`, `K%32==0`). The unsupported classification comes from activation or weight-decode divisibility, not from a claim that FlyDSL cannot implement those shapes. Current FlyDSL performs row padding/tail handling for its own kernel.

## Runner and aggregator review

`run_vllm_sweep.py` sets the same source Python paths, ROCm SDK root, AMD SMI path, compiler, and load-time-dequantization setting used in the successful preflight. It also sets `PYTORCH_ROCM_ARCH=gfx1201` and `MAX_JOBS=4`; the preflight inherited neither setting. GPU visibility masks were unset in the preflight, and the runner removes them to keep logical device mapping consistent. It checks source revisions and cleanliness, freezes the harness hash for the run, archives replaced output on retry, keeps per-batch logs/status and incremental JSON, validates exact shape coverage, and fails on unsupported/unexpected row statuses or incomplete successful row counts.

`run_vllm_sweep.py` and `summarize_vllm.py` now independently validate that every `correctness` metric is finite, enforce the per-backend own-reference RMS limit, and check timing samples plus changed-input graph RMS. The summarizer records failed batch statuses and row errors, reports missing shapes per pass, separates paired from FlyDSL-only cases, and summarizes only shapes with three passes. It exposes expected and completed counts but does not itself assert the final totals equal 1,158 paired and 684 FlyDSL-only shapes per pass; the final completion check should verify 3,474 paired and 2,052 one-arm shape evaluations, no missing shapes, and all 102 batch jobs complete.
