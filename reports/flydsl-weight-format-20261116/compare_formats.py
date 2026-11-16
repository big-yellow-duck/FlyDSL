"""Local paired E4M3/E5M2 experiment; does not modify kernel selection."""
import hashlib
import importlib.util
import json
import random
import statistics
from functools import partial
from pathlib import Path

import torch

from kernels.gemm.rdna4_mxfp4 import choose_config, gemm, get_device_capabilities, prepare_weights

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
REPO = ROOT / "FlyDSL-pr1374-review"
spec = importlib.util.spec_from_file_location("native_bench", REPO / "tests/perf/bench_rdna4_mxfp4.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

SHAPES = [
    (1, 4096, 4096), (8, 528, 8192), (8, 4096, 4096),
    (32, 256, 256), (73, 2064, 8192), (128, 4096, 4096),
    (512, 4096, 4096), (2048, 4096, 4096), (1, 17408, 5120),
    (128, 17408, 5120), (73, 513, 1120), (257, 128, 512),
]
BOUNDARY_SHAPES = [(1, 4096, 4096), (8, 4096, 4096), (32, 256, 256), (73, 2064, 8192)]
CASES = [(shape, 7) for shape in SHAPES] + [(shape, span) for span in (12, 14) for shape in BOUNDARY_SHAPES]


def save(name, data):
    path = OUT / name
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def main():
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    hardware = get_device_capabilities(0)
    stream = torch.cuda.Stream(device=0)
    flush = torch.ones(256 * 1024**2, dtype=torch.uint8, device="cuda")
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
    provenance = dict(
        gpu=torch.cuda.get_device_name(0), hardware=hardware, torch=torch.__version__,
        shapes=SHAPES, boundary_shapes=BOUNDARY_SHAPES, cases=CASES, passes=3, rounds=16,
        source_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (REPO / "kernels/gemm/rdna4_mxfp4").glob("*.py")},
        note="Same inputs, selected config, GPU and paired protocol; only preparation format changes. Synthetic BF16-output GEMM, not model evaluation.",
    )
    save("provenance.json", provenance)

    # Boundary probes include every FP4 code and independently check exact folding.
    boundaries = []
    for span in (7, 11, 12, 14, 15, 26, 27, 28, 29):
        codes = torch.arange(16, device="cuda", dtype=torch.uint8)[:, None]
        packed = (codes | (codes << 4)).expand(16, 32).contiguous()
        scales = torch.full((16, 2), 131, dtype=torch.uint8, device="cuda")
        scales[:, 0] -= span
        x = torch.ones((16, 64), device="cuda", dtype=torch.float8_e4m3fn)
        xs = torch.ones(16, device="cuda")
        reference_weights = lut[codes.long()].expand(16, 64) * torch.exp2(scales.float() - 127).repeat_interleave(32, 1)
        expected = (x.float() @ reference_weights.T).to(torch.bfloat16)
        row = dict(span=span, auto=prepare_weights(packed, scales).mode, modes={})
        for mode, dtype in (("e4m3", torch.float8_e4m3fn), ("e5m2", torch.float8_e5m2), ("grouped", None)):
            try:
                w = prepare_weights(packed, scales, mode=mode)
            except ValueError as exc:
                assert mode != "grouped"
                row["modes"][mode] = dict(rejected=str(exc))
                continue
            if dtype is not None:
                base = w.base_exponents.reshape(-1)[:16].float()
                folded = reference_weights / torch.exp2(base[:, None] - 127)
                assert torch.equal(folded.to(dtype).float(), folded), (span, mode)
            y = gemm(x, w, xs)
            torch.testing.assert_close(y, expected, rtol=0, atol=0)
            row["modes"][mode] = dict(normal_only=w.normal_only, exact_output=True, exact_fold=dtype is not None)
        boundaries.append(row)
    save("boundaries.json", boundaries)
    print("Boundary probes passed", flush=True)

    rows = []
    for pass_id in (1, 2, 3):
        order = list(CASES)
        random.Random(700 + pass_id).shuffle(order)
        for shape, span in order:
            m, n, k = shape
            generator = torch.Generator(device="cuda").manual_seed(20261116)
            packed = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8, generator=generator)
            scales = torch.randint(131-span, 132, (n, k // 32), device="cuda", dtype=torch.uint8, generator=generator)
            scales[:, 0] = 131 - span
            scales[:, -1] = 131
            x = torch.randn(m, k, device="cuda", generator=generator).to(torch.float8_e4m3fn)
            xs = torch.ones(m, device="cuda")
            rr = torch.linspace(0, m-1, min(m, 32), device="cuda").long().unique()
            cc = torch.linspace(0, n-1, min(n, 64), device="cuda").long().unique()
            codes = torch.stack((packed[cc] & 15, packed[cc] >> 4), -1).flatten(-2).long()
            decoded = lut[codes] * torch.exp2(scales[cc].float()-127).repeat_interleave(32, 1)
            expected = (x.float()[rr] @ decoded.T).to(torch.bfloat16).float()
            config = choose_config(m, n, k, "e4m3", **hardware)
            assert config == choose_config(m, n, k, "e5m2", **hardware)
            condition = torch.cuda.CUDAGraph()
            with torch.cuda.graph(condition, stream=stream):
                flush.sum(dtype=torch.int32)
                x.view(torch.uint8).sum(dtype=torch.int32)
                xs.sum()
            graphs, outputs, prepared, errors = {}, {}, {}, {}
            for mode in ("e4m3", "e5m2"):
                w = prepare_weights(packed, scales, mode=mode)
                out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
                fn = partial(gemm, x, w, xs, out=out, config=config)
                fn()
                torch.cuda.synchronize()
                bench.check(out[rr[:, None], cc[None, :]], expected)
                actual = out[rr[:, None], cc[None, :]].float()
                errors[mode] = float(((actual-expected).square().mean()/expected.square().mean().clamp_min(1e-30)).sqrt())
                for regime, calls in (("hot", 8 if m < 128 else 2), ("read_evicted", 1)):
                    graphs[mode, regime] = bench.capture(fn, calls, stream)
                prepared[mode], outputs[mode] = w, out
            identical = torch.equal(outputs["e4m3"], outputs["e5m2"])
            delta = outputs["e4m3"].float() - outputs["e5m2"].float()
            cross_rms = float((delta.square().mean()/outputs["e4m3"].float().square().mean().clamp_min(1e-30)).sqrt())
            assert cross_rms <= .005
            x.view(torch.uint8).bitwise_xor_(128)
            for (mode, regime), (graph, events, calls) in graphs.items():
                outputs[mode].fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                bench.check(outputs[mode][rr[:, None], cc[None, :]], -expected)
            x.view(torch.uint8).bitwise_xor_(128)
            samples = {mode: {regime: [] for regime in ("hot", "read_evicted")} for mode in prepared}
            rng = random.Random(f"{pass_id}-{shape}-{span}")
            for _ in range(16):
                trials = list(graphs)
                rng.shuffle(trials)
                for mode, regime in trials:
                    graph, events, calls = graphs[mode, regime]
                    (condition if regime == "read_evicted" else graph).replay()
                    graph.replay()
                    torch.cuda.synchronize()
                    samples[mode][regime].append(events[0].elapsed_time(events[1])*1000/calls)
            medians = {mode: {regime: statistics.median(v) for regime, v in data.items()} for mode, data in samples.items()}
            row = dict(shape=shape, span=span, pass_id=pass_id, config=config,
                       normal_only={mode: w.normal_only for mode, w in prepared.items()},
                       own_reference_rms=errors, bitwise_identical=identical, cross_format_rms=cross_rms,
                       samples_us=samples, medians_us=medians,
                       e5m2_speedup={regime: medians["e4m3"][regime]/medians["e5m2"][regime] for regime in ("hot", "read_evicted")})
            rows.append(row)
            save("results.json", rows)
            print(json.dumps({key: row[key] for key in ("shape", "span", "pass_id", "bitwise_identical", "e5m2_speedup")}), flush=True)
            del graphs, outputs, prepared, condition, fn, graph, out, w, delta, decoded, x, xs
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
