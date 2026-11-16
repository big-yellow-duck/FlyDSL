#!/usr/bin/env python3
"""Paired vLLM MXFP4 W4A4-emulation vs FlyDSL W4A8 layer benchmark.

This runner calls the checked-out vLLM ``apply_weights`` implementations.
It aliases the FlyDSL checkout's existing top-level ``kernels`` package as
``flydsl.kernels`` because vLLM's integration imports that name while the
FlyDSL checkout currently exposes ``kernels.gemm``. The package itself and its
submodules remain the checkout's source modules.

Example after a coordinated GPU run is approved:
    python bench_mxfp4_pair.py --m 1 8 128 --nk 17408x5120 --gpu 0 \
        --weights /path/to/model.safetensors \
        --weight-key model.language_model.layers.0.mlp.gate_proj.weight \
        --output paired.json

Without ``--weights``, deterministic synthetic packed weights are used. This
mode supports arbitrary shape sweeps, but is not a real-checkpoint weight
benchmark. Activations are deterministic BF16 Gaussian samples in either mode.
No model-level forward pass is run.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[3]
VLLM_ROOT = ROOT / "vllm-flydsl-mxfp4-rdna4"
FLYDSL_ROOT = ROOT / "FlyDSL-pr1374-review"


def install_source_alias() -> dict[str, str]:
    """Expose source ``kernels`` at the namespace expected by this vLLM tree."""
    if str(VLLM_ROOT) not in sys.path:
        sys.path.insert(0, str(VLLM_ROOT))
    if str(FLYDSL_ROOT) not in sys.path:
        sys.path.insert(0, str(FLYDSL_ROOT))

    import flydsl

    source_kernels = importlib.import_module("kernels")
    existing = sys.modules.get("flydsl.kernels")
    if existing is not None and Path(existing.__file__).resolve() != Path(
        source_kernels.__file__
    ).resolve():
        raise RuntimeError(
            "flydsl.kernels is already loaded from a different package: "
            f"{getattr(existing, '__file__', None)}"
        )
    sys.modules["flydsl.kernels"] = source_kernels
    flydsl.kernels = source_kernels
    return {
        "alias": "flydsl.kernels -> FlyDSL-pr1374-review/kernels",
        "flydsl_module": str(Path(flydsl.__file__).resolve()),
        "source_kernels_module": str(Path(source_kernels.__file__).resolve()),
    }


def parse_nk(values: list[str]) -> list[tuple[int, int]]:
    result = []
    for value in values:
        try:
            n_text, k_text = value.lower().split("x", 1)
            n, k = int(n_text), int(k_text)
        except (ValueError, TypeError) as exc:
            raise argparse.ArgumentTypeError(f"invalid N x K shape: {value}") from exc
        if n <= 0 or k <= 0 or k % 32:
            raise argparse.ArgumentTypeError(
                f"FlyDSL/emulation require N>0 and K>0 divisible by 32: {value}"
            )
        result.append((n, k))
    return result


def load_weights(
    args: argparse.Namespace, device: torch.device, n: int, k: int
) -> tuple[torch.Tensor, torch.Tensor, str]:
    if args.weights:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("--weights requires the safetensors package") from exc
        weight_key = args.weight_key
        if not weight_key:
            raise ValueError("--weight-key is required with --weights")
        with safe_open(str(args.weights), framework="pt", device="cpu") as handle:
            weight = handle.get_tensor(weight_key)
            scale = handle.get_tensor(weight_key + "_scale")
        if weight.dtype != torch.uint8 or scale.dtype != torch.uint8:
            raise ValueError("checkpoint MXFP4 weight and scales must both be uint8")
        if tuple(weight.shape) != (n, k // 2) or tuple(scale.shape) != (n, k // 32):
            raise ValueError(
                f"checkpoint tensor shapes are {tuple(weight.shape)} and "
                f"{tuple(scale.shape)}, expected {(n, k // 2)} and {(n, k // 32)}"
            )
        return weight.to(device), scale.to(device), f"{args.weights}:{weight_key}"

    generator = torch.Generator(device=device).manual_seed(args.seed)
    packed = torch.randint(
        0, 256, (n, k // 2), dtype=torch.uint8, device=device, generator=generator
    )
    scales = torch.randint(
        124,
        132,
        (n, k // 32),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    return packed, scales, "synthetic:seeded uint8 MXFP4 weights and E8M0 scales"


def build_layer(
    backend: str,
    packed: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[Any, torch.nn.Module]:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.kernels.linear import init_mxfp4_linear_kernel
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp4Dynamic,
    )

    vllm_config = VllmConfig() if backend == "auto" else VllmConfig(
        kernel_config={"linear_backend_per_quant": {"mxfp4": backend}}
    )
    with set_current_vllm_config(vllm_config), torch.device(packed.device):
        kernel = init_mxfp4_linear_kernel(kMxfp4Dynamic)
        layer = torch.nn.Module()
        layer.params_dtype = torch.bfloat16
        layer.register_parameter(
            "weight", torch.nn.Parameter(torch.empty_like(packed), requires_grad=False)
        )
        layer.register_parameter(
            "weight_scale",
            torch.nn.Parameter(torch.empty_like(scales), requires_grad=False),
        )
        default_weight_loader(layer.weight, packed.clone())
        default_weight_loader(layer.weight_scale, scales.clone())
        kernel.process_weights_after_loading(layer)
    return kernel, layer


def relative_rms(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = (actual.float() - expected.float()).square().mean()
    denominator = expected.float().square().mean().clamp_min(1e-30)
    return float(torch.sqrt(numerator / denominator))


def decode_reference_weights(
    packed: torch.Tensor, scales: torch.Tensor, sample_cols: torch.Tensor
) -> torch.Tensor:
    packed_rows = packed[sample_cols]
    scale_rows = scales[sample_cols]
    codes = torch.stack((packed_rows & 15, packed_rows >> 4), -1).flatten(-2).long()
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=packed.device
    )
    decoded = magnitudes[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)
    decoded *= torch.exp2(scale_rows.float() - 127).repeat_interleave(32, dim=1)
    return decoded.to(torch.bfloat16)


def reference_w4a8(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    actual: torch.Tensor,
    sample_rows: torch.Tensor,
    sample_cols: torch.Tensor,
) -> dict[str, float]:
    from vllm import _custom_ops as ops

    decoded_weight = decode_reference_weights(packed, scales, sample_cols)
    fp8_x, inv_scale = ops.scaled_fp8_quant(
        x[sample_rows].contiguous(), use_per_token_if_dynamic=True
    )
    expected = (
        fp8_x.float() @ decoded_weight.float().T
        * inv_scale.reshape(-1, 1).float()
    ).to(torch.bfloat16)
    observed = actual[sample_rows][:, sample_cols]
    return {
        "w4a8_flydsl_own_ref_relative_rms": relative_rms(observed, expected)
    }


def references(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    out_w4a4: torch.Tensor,
    out_w4a8: torch.Tensor,
    sample_rows: torch.Tensor,
    sample_cols: torch.Tensor,
) -> dict[str, float]:
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        quant_dequant_mxfp4,
    )

    reference_dtype = torch.bfloat16
    # Independent OCP E2M1 decode; the Quark helper has a minimum/stride
    # constraint that can reject narrow sampled output slices.
    decoded_weight = decode_reference_weights(packed, scales, sample_cols)

    # The implementation's Q/DQ kernel requires the whole activation tensor
    # element count to be a multiple of 64. Quantize the real full input first,
    # then sample outputs for the independent reference calculation.
    qdq_x = quant_dequant_mxfp4(x.contiguous())[sample_rows]
    ref_w4a4 = (qdq_x.float() @ decoded_weight.float().T).to(reference_dtype)

    fp8_x, inv_scale = ops.scaled_fp8_quant(
        x[sample_rows].contiguous(), use_per_token_if_dynamic=True
    )
    ref_w4a8 = (
        fp8_x.float() @ decoded_weight.float().T
        * inv_scale.reshape(-1, 1).float()
    ).to(reference_dtype)

    actual4 = out_w4a4[sample_rows][:, sample_cols]
    actual8 = out_w4a8[sample_rows][:, sample_cols]
    cross = actual4.float() - actual8.float()
    return {
        "w4a4_emulation_own_ref_relative_rms": relative_rms(actual4, ref_w4a4),
        "w4a8_flydsl_own_ref_relative_rms": relative_rms(actual8, ref_w4a8),
        "cross_backend_relative_rms": relative_rms(actual4, actual8),
        "cross_backend_max_abs": float(cross.abs().max()),
    }


def capture(
    fn: Callable[[], torch.Tensor], calls: int, stream: torch.cuda.Stream
) -> tuple[
    torch.cuda.CUDAGraph,
    torch.cuda.Event,
    torch.cuda.Event,
    int,
    list[torch.Tensor],
]:
    graph = torch.cuda.CUDAGraph()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    outputs: list[torch.Tensor] = []
    with torch.cuda.graph(graph, stream=stream):
        start.record()
        for _ in range(calls):
            outputs.append(fn())
        end.record()
    return graph, start, end, calls, outputs


def time_pair(
    functions: dict[str, Callable[[], torch.Tensor]],
    stream: torch.cuda.Stream,
    rounds: int,
    flush_bytes: int,
    x: torch.Tensor,
) -> dict[str, dict[str, Any]]:
    flush = torch.empty(flush_bytes, dtype=torch.uint8, device="cuda")
    condition = torch.cuda.CUDAGraph()
    with torch.cuda.graph(condition, stream=stream):
        flush.sum(dtype=torch.int64)
        x.view(torch.uint8).sum(dtype=torch.int64)

    graphs: dict[
        tuple[str, str],
        tuple[torch.cuda.CUDAGraph, torch.cuda.Event, torch.cuda.Event, int, list[torch.Tensor]],
    ] = {}
    for name, fn in functions.items():
        hot_calls = 8 if x.shape[0] < 128 else 2
        for regime, calls in (("hot", hot_calls), ("rotating", 1)):
            graphs[name, regime] = capture(fn, calls, stream)
    torch.cuda.synchronize()

    for graph, _, _, _, _ in graphs.values():
        graph.replay()
    torch.cuda.synchronize()

    original_x = x.clone()
    x.neg_()
    negative_references = {name: fn() for name, fn in functions.items()}
    torch.cuda.synchronize()
    if not all(torch.isfinite(output).all() for output in negative_references.values()):
        raise AssertionError("non-finite output for changed-input reference")
    changed_input_errors: dict[str, float] = {}
    for (name, _regime), (graph, _start, _end, _calls, outputs) in graphs.items():
        graph.replay()
        torch.cuda.synchronize()
        actual = outputs[-1]
        if not torch.isfinite(actual).all():
            raise AssertionError(f"non-finite changed-input graph output for {name}")
        changed_input_errors[name] = max(
            changed_input_errors.get(name, 0.0),
            relative_rms(actual, negative_references[name]),
        )
    x.copy_(original_x)
    torch.cuda.synchronize()

    samples: dict[str, dict[str, list[float]]] = {
        name: {"hot": [], "rotating": []} for name in functions
    }
    order_rng = random.Random(20261116)
    for _ in range(rounds):
        order = [(name, regime) for name in functions for regime in ("hot", "rotating")]
        order_rng.shuffle(order)
        for name, regime in order:
            graph, start, end, calls, _ = graphs[name, regime]
            if regime == "rotating":
                condition.replay()
            else:
                graph.replay()
            graph.replay()
            torch.cuda.synchronize()
            samples[name][regime].append(start.elapsed_time(end) * 1000 / calls)

    return {
        "regimes": {
            name: {
                regime: {
                    "samples_us": values,
                    "median_us": statistics.median(values),
                }
                for regime, values in by_regime.items()
            }
            for name, by_regime in samples.items()
        },
        "graph_changed_input_relative_rms": changed_input_errors,
        "cold_protocol": "read_evicted_weights_warm_activations",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", nargs="+", type=int, default=[])
    parser.add_argument("--nk", nargs="+", default=[], metavar="N×K")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--weight-key")
    parser.add_argument(
        "--shape-manifest",
        type=Path,
        help="Run the exact union of native_shapes and tail_shapes triples in a manifest",
    )
    parser.add_argument("--pass-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--flush-mib", type=int, default=256)
    parser.add_argument(
        "--max-own-ref-rms", type=float, default=0.005,
        help="fail correctness gate if either path exceeds this sampled relative RMS",
    )
    args = parser.parse_args()
    if args.shape_manifest:
        manifest = json.loads(args.shape_manifest.read_text())
        triples = set()
        for name in ("native_shapes", "tail_shapes"):
            for m, n, k in manifest.get(name, []):
                triples.add((int(m), int(n), int(k)))
        if not triples:
            parser.error("shape manifest has no native_shapes/tail_shapes entries")
        if args.m or args.nk:
            parser.error("--shape-manifest cannot be combined with --m/--nk")
        shape_groups: dict[tuple[int, int], list[int]] = {}
        for m, n, k in sorted(triples, key=lambda shape: (shape[1], shape[2], shape[0])):
            if m <= 0 or n <= 0 or k <= 0 or k % 32:
                parser.error(f"manifest contains unsupported dimensions {(m, n, k)}")
            shape_groups.setdefault((n, k), []).append(m)
    else:
        if not args.m or not args.nk:
            parser.error("provide --m and --nk, or --shape-manifest")
        shape_groups = {
            (n, k): sorted(set(args.m)) for n, k in parse_nk(args.nk)
        }
    if args.weights and len(shape_groups) != 1:
        parser.error("--weights supports one checkpoint N/K shape per process")
    if os.getenv("VLLM_MXFP4_EMULATION_DEQUANT_AT_LOAD", "false").lower() in (
        "1", "true", "yes"
    ):
        parser.error(
            "primary baseline requires VLLM_MXFP4_EMULATION_DEQUANT_AT_LOAD=0; "
            "default W4A4 timing dequantizes weights inside each apply_weights call"
        )
    if any(m <= 0 for m in args.m) or args.rounds <= 0 or args.flush_mib < 1:
        parser.error("M, rounds, and flush MiB must be positive")
    alias_info = install_source_alias()
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(1)
    if not torch.version.hip:
        raise RuntimeError("This paired benchmark requires ROCm")
    device = torch.device(f"cuda:{args.gpu}")

    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
    )

    rows: list[dict[str, Any]] = []
    dispatch_checks: list[dict[str, Any]] = []
    from flydsl.kernels.gemm.rdna4_mxfp4 import (
        choose_config,
        get_device_capabilities,
    )

    device_capabilities = get_device_capabilities(args.gpu)
    if device_capabilities.get("cache_size_bytes") != 64 * 1024**2:
        raise RuntimeError(
            "expected 64 MiB L2/cache capacity for this sweep; got "
            f"{device_capabilities.get('cache_size_bytes')} bytes"
        )
    failed_cases = 0

    def write_report() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "runner": str(Path(__file__).resolve()),
                    "vllm_revision": "3c138b588506cf2196e5f183d3fe893d973ec2e3",
                    "flydsl_revision": "3ed2ca5f68ecc2fd972cb23d4da366364e91d261",
                    "source_alias": alias_info,
                    "execution_environment": {
                        "python_executable": sys.executable,
                        "rocm_path": os.getenv("ROCM_PATH"),
                        "compiler": os.getenv("CXX"),
                        "path": os.getenv("PATH"),
                        "pythonpath": os.getenv("PYTHONPATH"),
                        "flydsl_gpu_arch": os.getenv("FLYDSL_GPU_ARCH"),
                        "emulation_dequant_at_load": os.getenv(
                            "VLLM_MXFP4_EMULATION_DEQUANT_AT_LOAD", "false"
                        ),
                    },
                    "gpu": args.gpu,
                    "gpu_name": torch.cuda.get_device_name(args.gpu),
                    "architecture": torch.cuda.get_device_properties(args.gpu).gcnArchName,
                    "torch_version": torch.__version__,
                    "rounds": args.rounds,
                    "pass_id": args.pass_id,
                    "flush_mib": args.flush_mib,
                    "device_capabilities": device_capabilities,
                    "runtime_dispatch_checks": dispatch_checks,
                    "timed_region": (
                        "CUDA event interval around captured apply_weights kernels; includes "
                        "per-call W4A4 weight decode and activation Q/DQ, excludes CPU "
                        "dispatch/allocator overhead and one-time weight setup"
                    ),
                    "activation_provenance": "seeded BF16 Gaussian input; same tensor per paired arm",
                    "baseline_weight_decode": "per apply_weights call (VLLM_MXFP4_EMULATION_DEQUANT_AT_LOAD must be false)",
                    "results": rows,
                },
                indent=2,
            )
            + "\n"
        )

    for n, k in shape_groups:
        packed, scales, weights_source = load_weights(args, device, n, k)
        try:
            k4, layer4 = build_layer("emulation", packed, scales)
            k8, layer8 = build_layer("auto", packed, scales)
            if type(k4).__name__ != "EmulationMxfp4LinearKernel":
                raise AssertionError(f"forced W4A4 selector chose {type(k4).__name__}")
            if type(k8).__name__ != "Rdna4Mxfp4LinearKernel":
                raise AssertionError(f"auto gfx1201 selector chose {type(k8).__name__}")
            if k8.config.activation_quant_key != kFp8DynamicTokenSym:
                raise AssertionError("auto RDNA4 selector did not apply the FP8 override")
            backend_info = {
                "w4a4_emulation": {
                    "class": type(k4).__name__,
                    "activation_quant_key": str(k4.config.activation_quant_key),
                    "selected_by": "init_mxfp4_linear_kernel(kMxfp4Dynamic) with explicit per-quant emulation backend",
                },
                "w4a8_flydsl": {
                    "class": type(k8).__name__,
                    "activation_quant_key": str(k8.config.activation_quant_key),
                    "selected_by": "init_mxfp4_linear_kernel(kMxfp4Dynamic) with auto gfx1201 dispatch and RDNA4 FP8 substitution",
                },
            }
            dispatch_checks.append({"shape_nk": [n, k], **backend_info})
        except Exception as exc:
            for failed_m in shape_groups[n, k]:
                failed_cases += 1
                rows.append(
                    {
                        "shape_mnk": [failed_m, n, k],
                        "pass_id": args.pass_id,
                        "status": "setup_failed_or_unsupported",
                        "error": f"{type(exc).__name__}: {exc}",
                        "weights_source": weights_source,
                    }
                )
                write_report()
            del packed, scales
            torch.cuda.empty_cache()
            continue

        for m in shape_groups[n, k]:
            x = out4 = out8 = fn4 = fn8 = None
            try:
                generator = torch.Generator(device=device).manual_seed(args.seed + m + n + k)
                x = torch.randn((m, k), dtype=torch.bfloat16, device=device, generator=generator)
                fn4 = lambda: k4.apply_weights(layer4, x)
                fn8 = lambda: k8.apply_weights(layer8, x)
                torch.backends.cuda.matmul.allow_tf32 = False
                try:
                    out4 = fn4()
                except Exception as emulation_error:
                    rejection = str(emulation_error)
                    if not (
                        "multiple of 64" in rejection
                        or "multiple of 512" in rejection
                        or "multiple of OUTPUTS_PER_THREAD * 64 = 512" in rejection
                    ):
                        raise
                    out8 = fn8()
                    torch.cuda.synchronize()
                    if not torch.isfinite(out8).all():
                        raise AssertionError("non-finite FlyDSL output")
                    sample_rows = torch.linspace(
                        0, m - 1, min(m, 32), device=device
                    ).long().unique()
                    sample_cols = torch.linspace(
                        0, n - 1, min(n, 64), device=device
                    ).long().unique()
                    errors = reference_w4a8(
                        x, packed, scales, out8, sample_rows, sample_cols
                    )
                    if not all(math.isfinite(value) for value in errors.values()):
                        raise AssertionError(f"non-finite W4A8 reference metric: {errors}")
                    if max(errors.values()) > args.max_own_ref_rms:
                        raise AssertionError(f"W4A8 own-reference gate failed: {errors}")
                    timing_stream = torch.cuda.Stream(device=args.gpu)
                    with torch.cuda.stream(timing_stream):
                        timings = time_pair(
                            {"w4a8_flydsl": fn8},
                            timing_stream,
                            args.rounds,
                            args.flush_mib * 1024 * 1024,
                            x,
                        )
                    changed = timings["graph_changed_input_relative_rms"]
                    if not all(math.isfinite(value) for value in changed.values()):
                        raise AssertionError(f"non-finite W4A8 changed-input metric: {changed}")
                    if max(changed.values()) > args.max_own_ref_rms:
                        raise AssertionError(f"W4A8 changed-input gate failed: {changed}")
                    flydsl_config = choose_config(
                        m, n, k, layer8._flydsl_mxfp4_mode,
                        cache_policy="rotating", **device_capabilities,
                    )
                    rows.append(
                        {
                            "shape_mnk": [m, n, k],
                            "pass_id": args.pass_id,
                            "weights_source": weights_source,
                            "backend": backend_info,
                            "emulation_rejection": rejection,
                            "correctness": errors,
                            "flydsl_runtime_selection": {
                                "cache_policy": "rotating (vLLM integration hardcodes this)",
                                "config": flydsl_config,
                            },
                            "timings": timings,
                            "status": "emulation_unsupported_flydsl_ok",
                        }
                    )
                    write_report()
                    continue
                out8 = fn8()
                torch.cuda.synchronize()
                if not torch.isfinite(out4).all() or not torch.isfinite(out8).all():
                    raise AssertionError("non-finite eager output")
                sample_rows = torch.linspace(0, m - 1, min(m, 32), device=device).long().unique()
                sample_cols = torch.linspace(0, n - 1, min(n, 64), device=device).long().unique()
                errors = references(x, packed, scales, out4, out8, sample_rows, sample_cols)
                own_errors = max(
                    errors["w4a4_emulation_own_ref_relative_rms"],
                    errors["w4a8_flydsl_own_ref_relative_rms"],
                )
                if own_errors > args.max_own_ref_rms:
                    raise AssertionError(f"own-reference correctness gate failed: {errors}")
                if not all(math.isfinite(value) for value in errors.values()):
                    raise AssertionError(f"non-finite numerical error metric: {errors}")

                rotating_config = choose_config(
                    m,
                    n,
                    k,
                    layer8._flydsl_mxfp4_mode,
                    cache_policy="rotating",
                    **device_capabilities,
                )
                timing_stream = torch.cuda.Stream(device=args.gpu)
                with torch.cuda.stream(timing_stream):
                    timings = time_pair(
                        {"w4a4_emulation": fn4, "w4a8_flydsl": fn8},
                        timing_stream,
                        args.rounds,
                        args.flush_mib * 1024 * 1024,
                        x,
                    )
                changed_input_errors = timings["graph_changed_input_relative_rms"]
                if not all(math.isfinite(value) for value in changed_input_errors.values()):
                    raise AssertionError(
                        f"non-finite changed-input graph metric: {changed_input_errors}"
                    )
                if max(changed_input_errors.values()) > args.max_own_ref_rms:
                    raise AssertionError(
                        f"changed-input graph correctness gate failed: {changed_input_errors}"
                    )
                rows.append(
                    {
                        "shape_mnk": [m, n, k],
                        "pass_id": args.pass_id,
                        "weights_source": weights_source,
                        "seed": args.seed + m + n + k,
                        "input_dtype": "bfloat16",
                        "output_dtype": "bfloat16",
                        "backend": {
                            "w4a4_emulation": type(k4).__name__,
                            "w4a8_flydsl": type(k8).__name__,
                        },
                        "activation_schemes": {
                            "w4a4_emulation": "dynamic_mxfp4 checkpoint scheme",
                            "w4a8_flydsl": "dynamic per-token fp8_e4m3",
                        },
                        "flydsl_runtime_selection": {
                            "cache_policy": "rotating (vLLM integration hardcodes this)",
                            "config": rotating_config,
                        },
                        "correctness": errors,
                        "timings": timings,
                        "status": "ok",
                    }
                )
            except Exception as exc:
                failed_cases += 1
                rows.append(
                    {
                        "shape_mnk": [m, n, k],
                        "pass_id": args.pass_id,
                        "weights_source": weights_source,
                        "backend": backend_info,
                        "status": "correctness_failed_or_runtime_unsupported",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            finally:
                del x, out4, out8, fn4, fn8
            write_report()

        del k4, k8, layer4, layer8, packed, scales
        torch.cuda.empty_cache()
    if failed_cases:
        raise SystemExit(f"paired benchmark recorded {failed_cases} failed/unsupported cases")


if __name__ == "__main__":
    main()
