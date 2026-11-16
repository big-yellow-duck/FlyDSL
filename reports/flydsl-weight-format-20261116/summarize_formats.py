"""Summarize the local paired format experiment without dropping cases."""
import collections
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "results.json").read_text())
groups = collections.defaultdict(list)
for row in data:
    groups[tuple(row["shape"]), row["span"]].append(row)
assert len(data) == 60 and len(groups) == 20
rows = []
for (shape, span), passes in sorted(groups.items()):
    assert len(passes) == 3 and {r["pass_id"] for r in passes} == {1, 2, 3}
    for regime in ("hot", "read_evicted"):
        ratios = [r["e5m2_speedup"][regime] for r in passes]
        rows.append(dict(
            m=shape[0], n=shape[1], k=shape[2], scale_span=span, regime=regime,
            engine=passes[0]["config"]["engine"], decoder=passes[0]["config"]["table_mode"],
            e4m3_us=statistics.median(r["medians_us"]["e4m3"][regime] for r in passes),
            e5m2_us=statistics.median(r["medians_us"]["e5m2"][regime] for r in passes),
            e5m2_speedup=statistics.median(ratios), min_pass_speedup=min(ratios), max_pass_speedup=max(ratios),
        ))
with (ROOT / "paired.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

by_span = {}
for span in (7, 12, 14):
    by_span[span] = {
        regime: math.exp(statistics.mean(math.log(r["e5m2_speedup"]) for r in rows if r["scale_span"] == span and r["regime"] == regime))
        for regime in ("hot", "read_evicted")
    }
summary = dict(cases=20, passes=3, evaluations=60,
               all_outputs_bitwise_identical=all(r["bitwise_identical"] for r in data),
               max_own_reference_rms=max(max(r["own_reference_rms"].values()) for r in data),
               by_span_geomean_e5m2_speedup=by_span)
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = [
    "# E4M3 versus E5M2 folded MXFP4 weights",
    "",
    "Measured on R9700 / gfx1201 GPU 0, 16 November 2026, after the local cache-policy removal. This compares the FP8 format used to represent decoded MXFP4 weight fragments; activations remain E4M3 and output remains BF16.",
    "",
    "**The distinction is primarily exponent coverage.** Both formats can exactly represent the E2M1 significands scaled by powers of two when their range permits it. E4M3's extra mantissa precision adds no weight precision for these inputs. E5M2 handles a wider scale span. Keeping a losslessness check and grouped fallback remains necessary; maintaining two folded formats looks optional based on this bounded performance sample.",
    "",
    "## Paired performance",
    "",
    "Speedup is E4M3 latency / E5M2 latency; above one favors E5M2. The aggregate is the geometric mean of each case's median three-pass ratio.",
    "",
    "| E8M0 exponent span | Shapes | E4M3 normal-only decode | Hot speedup | Read-evicted speedup |",
    "| --- | ---: | --- | ---: | ---: |",
]
for span, count in ((7, 12), (12, 4), (14, 4)):
    v = by_span[span]
    lines.append(f"| {span} | {count} | {'Yes' if span == 7 else 'No'} | {v['hot']:.4f}× | {v['read_evicted']:.4f}× |")
lines += [
    "",
    "For the ordinary span-7 weights, E5M2 is about 0.6% slower hot and 0.4% slower read-evicted in this sample. Individual hot median ratios range from about 0.980× to 1.009×. This is a small measured difference, not proof of universal equivalence. For span 12/14, E5M2's normal-only ALU path avoids the E4M3 subnormal corrections; the two ALU register cases show roughly 4.4–6.6% hot gains. The LUT cases remain close. This is consistent with the source paths, but no ISA or hardware-counter profiling was performed to establish causality.",
    "",
    "All 60 paired case/pass evaluations produced **bitwise-identical full BF16 outputs** across E4M3 and E5M2. Both formats passed sampled independent FP32 references and changed-input graph replay. Maximum sampled reference relative RMS was " + f"{summary['max_own_reference_rms']:.8g}." ,
    "",
    "## Range boundaries",
    "",
    "Probes use every FP4 code, maximum E8M0 exponent byte 131, and the stated difference between maximum and minimum exponent bytes within an output channel. All accepted modes passed exact full-output reference comparisons. Folded probes also checked exact FP8 representability independently of kernel decoding.",
    "",
    "| Exponent span | Forced E4M3 | Forced E5M2 | Current auto choice |",
    "| --- | --- | --- | --- |",
]
for row in json.loads((ROOT / "boundaries.json").read_text()):
    status = lambda mode: "Rejected" if "rejected" in row["modes"][mode] else "Exact"
    lines.append(f"| {row['span']} | {status('e4m3')} | {status('e5m2')} | {row['auto']} |")
lines += [
    "",
    "The 14/28 span limits describe these ordinary exponent ranges. The implementation also clamps the reference exponent and rejects reserved exponent values, so actual eligibility remains the losslessness predicate in `prepare_weights`, not a span-only shortcut. Beyond E5M2's folding range, grouped K32 scaling is still required for arbitrary weights.",
    "",
    "## Method and scope",
    "",
    "Twenty cases: twelve distinct shapes at exponent span 7, plus four selected shapes each at spans 12 and 14. Shapes cover register/hybrid, ALU/LUT, decode/prefill, N tails, and the previous M8 outlier. Three passes used the same deterministic synthetic data and identical configs for both formats, randomized case/method order, and 16 paired GPU-event rounds per cache condition. Hot measurements use eight calls for M<128 and two otherwise. Read-evicted measurements read 256 MiB and activations outside timing, then time one call. Preparation, compilation and correctness checks are outside timing. All pairs run on the same GPU. No clock or power settings were changed; this experiment does not include synchronized hardware-counter profiling.",
    "",
    "This is a targeted format comparison, not a repeat of the 1,842-shape sweep, exhaustive scale testing, or a full-model quality evaluation. The production format-selection code was not changed. A reasonable simplification to consider is E5M2 as the single folded format plus grouped fallback, followed by broader validation before replacing the existing default.",
    "",
    "## Reproduction",
    "",
    "Run from the recorded workspace root with the existing environment and local source checkout:",
    "",
    "```bash",
    "ROCM_PATH=$(.venv/bin/rocm-sdk path --root)",
    "export ROCM_PATH",
    'export PYTHONPATH="$PWD/FlyDSL-pr1374-review/python:$PWD/FlyDSL-pr1374-review:$ROCM_PATH/share/amd_smi"',
    "export FLYDSL_GPU_ARCH=gfx1201 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
    ".venv/bin/python reports/flydsl-weight-format-20261116/compare_formats.py",
    ".venv/bin/python reports/flydsl-weight-format-20261116/summarize_formats.py",
    "```",
    "",
    "Re-running overwrites this experiment's output files. [Source/environment hashes](provenance.json), [raw samples](results.json), [paired CSV](paired.csv), [boundary checks](boundaries.json), and [summary](summary.json) are retained locally.",
]
(ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, indent=2))
