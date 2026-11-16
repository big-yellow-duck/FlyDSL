"""Aggregate actual vLLM layer pairs and preserve missing/failed cases."""
import collections
import csv
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
plan = json.loads((ROOT.parent / "flydsl-native-sweep-plan-20261116/shape-manifest.json").read_text())
expected = {tuple(s) for s in plan["native_shapes"] + plan["tail_shapes"]}
cohorts = {name: {tuple(s) for s in shapes} for name, shapes in plan["cohorts"].items()}
paired = collections.defaultdict(dict)
flydsl_only = collections.defaultdict(dict)
failures = []
statuses = []
observed = collections.defaultdict(set)
hashes = set()
for gpu in (0, 1):
    for path in sorted((ROOT / f"gpu{gpu}" / "vllm").glob("*.status.json")):
        status = json.loads(path.read_text())
        statuses.append(status)
        if status["status"] == "running":
            continue
        if status["status"] == "failed":
            failures.append(dict(file=str(path), kind="failed_batch", exit_code=status.get("exit_code"), validation_error=status.get("validation_error")))
        output = path.with_name(path.name.replace(".status.json", ".json"))
        if not output.exists():
            failures.append(dict(file=str(path), kind="missing_output", exit_code=status.get("exit_code")))
            continue
        data = json.loads(output.read_text())
        hashes.add(status["harness_sha256"])
        assert data["gpu"] == gpu
        for row in data["results"]:
            shape = tuple(row["shape_mnk"])
            pass_id = row["pass_id"]
            assert shape in expected and pass_id in (1, 2, 3)
            assert shape not in observed[pass_id], (shape, pass_id, "duplicate")
            observed[pass_id].add(shape)
            if row["status"] not in ("ok", "emulation_unsupported_flydsl_ok"):
                failures.append(dict(shape=shape, pass_id=pass_id, gpu=gpu, file=str(output), error=row.get("error")))
                continue
            only = row["status"] == "emulation_unsupported_flydsl_ok"
            assert all(math.isfinite(x) for x in row["correctness"].values()), "nonfinite correctness metric"
            assert row["correctness"]["w4a8_flydsl_own_ref_relative_rms"] <= .005
            if not only:
                assert row["correctness"]["w4a4_emulation_own_ref_relative_rms"] <= .005
            if only:
                m, n, k = shape
                assert m*k % 64 or n*k % 512
            for backend in (("w4a8_flydsl",) if only else ("w4a4_emulation", "w4a8_flydsl")):
                assert row["timings"]["graph_changed_input_relative_rms"][backend] <= .005
                for regime in ("hot", "rotating"):
                    samples = row["timings"]["regimes"][backend][regime]["samples_us"]
                    assert len(samples) == 16 and all(math.isfinite(x) and x > 0 for x in samples)
            (flydsl_only if only else paired)[shape, gpu][pass_id] = row


def band(m):
    return "M<=16" if m <= 16 else "17<=M<=128" if m <= 128 else "129<=M<=1024" if m <= 1024 else "M>1024"


rows = []
for (shape, gpu), passes in sorted(paired.items()):
    if set(passes) != {1, 2, 3}:
        continue
    configs = {json.dumps(r["flydsl_runtime_selection"]["config"], sort_keys=True) for r in passes.values()}
    assert len(configs) == 1, (shape, "config changed across passes")
    for regime in ("hot", "rotating"):
        times4 = [passes[p]["timings"]["regimes"]["w4a4_emulation"][regime]["median_us"] for p in (1, 2, 3)]
        times8 = [passes[p]["timings"]["regimes"]["w4a8_flydsl"][regime]["median_us"] for p in (1, 2, 3)]
        ratios = [a / b for a, b in zip(times4, times8)]
        rows.append(dict(
            m=shape[0], n=shape[1], k=shape[2], gpu=gpu, regime=regime, m_band=band(shape[0]),
            cohorts=";".join(name for name, shapes in cohorts.items() if shape in shapes),
            n_aligned_16=shape[1] % 16 == 0,
            w4a4_emulation_us=statistics.median(times4), w4a8_flydsl_us=statistics.median(times8),
            speedup=statistics.median(ratios), min_pass_speedup=min(ratios), max_pass_speedup=max(ratios),
            classification="consistent_win_gt5pct" if min(ratios)>1.05 else "consistent_loss_gt5pct" if max(ratios)<.95 else "small_or_inconsistent",
            **{f"pass{p}_speedup": ratios[p-1] for p in (1, 2, 3)},
            max_w4a4_own_ref_rms=max(r["correctness"]["w4a4_emulation_own_ref_relative_rms"] for r in passes.values()),
            max_w4a8_own_ref_rms=max(r["correctness"]["w4a8_flydsl_own_ref_relative_rms"] for r in passes.values()),
            median_cross_backend_rms=statistics.median(r["correctness"]["cross_backend_relative_rms"] for r in passes.values()),
            max_cross_backend_abs=max(r["correctness"]["cross_backend_max_abs"] for r in passes.values()),
        ))


def describe(sub):
    return dict(count=len(sub), geomean_speedup=math.exp(statistics.mean(math.log(r["speedup"]) for r in sub)) if sub else None,
                median_speedup=statistics.median(r["speedup"] for r in sub) if sub else None,
                min_speedup=min((r["speedup"] for r in sub), default=None), max_speedup=max((r["speedup"] for r in sub), default=None),
                consistent_wins=sum(r["classification"] == "consistent_win_gt5pct" for r in sub),
                consistent_losses=sum(r["classification"] == "consistent_loss_gt5pct" for r in sub),
                small_or_inconsistent=sum(r["classification"] == "small_or_inconsistent" for r in sub))


only_rows = []
for (shape, gpu), passes in sorted(flydsl_only.items()):
    if set(passes) != {1, 2, 3}:
        continue
    for regime in ("hot", "rotating"):
        times = [passes[p]["timings"]["regimes"]["w4a8_flydsl"][regime]["median_us"] for p in (1, 2, 3)]
        only_rows.append(dict(m=shape[0], n=shape[1], k=shape[2], gpu=gpu, regime=regime,
                             w4a8_flydsl_us=statistics.median(times), min_pass_us=min(times), max_pass_us=max(times),
                             activation_alignment_unsupported=bool(shape[0]*shape[2] % 64),
                             weight_alignment_unsupported=bool(shape[1]*shape[2] % 512),
                             emulation_rejection=passes[1]["emulation_rejection"],
                             max_w4a8_own_ref_rms=max(r["correctness"]["w4a8_flydsl_own_ref_relative_rms"] for r in passes.values())))
if only_rows:
    with (ROOT / "vllm-flydsl-only.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(only_rows[0]))
        writer.writeheader()
        writer.writerows(only_rows)

summary = dict(planned_shapes=len(expected), planned_evaluations=len(expected)*3,
               completed_three_pass_shapes=len(rows)//2, status_counts=dict(collections.Counter(s["status"] for s in statuses)),
               completed_flydsl_only_shapes=len(only_rows)//2, expected_paired_shapes=1158, expected_emulation_unsupported_shapes=684,
               failures=failures, missing_per_pass={str(p): [list(s) for s in sorted(expected-observed[p])] for p in (1,2,3)},
               harness_sha256=sorted(hashes), overall={}, by_m_band={}, by_cohort={}, worst={})
for regime in ("hot", "rotating"):
    subset = [r for r in rows if r["regime"] == regime]
    summary["overall"][regime] = describe(subset)
    summary["by_m_band"][regime] = {b: describe([r for r in subset if r["m_band"]==b]) for b in ("M<=16", "17<=M<=128", "129<=M<=1024", "M>1024")}
    summary["by_cohort"][regime] = {c: describe([r for r in subset if c in r["cohorts"].split(";")]) for c in cohorts}
    summary["worst"][regime] = sorted(subset, key=lambda r: r["speedup"])[:15]
if rows:
    with (ROOT / "vllm-paired.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    unique = [r for r in rows if r["regime"] == "hot"]
    summary["numerical"] = dict(max_w4a4_own_ref_rms=max(r["max_w4a4_own_ref_rms"] for r in unique),
                                max_w4a8_own_ref_rms=max(r["max_w4a8_own_ref_rms"] for r in unique),
                                median_cross_backend_rms=statistics.median(r["median_cross_backend_rms"] for r in unique),
                                max_cross_backend_rms=max(r["median_cross_backend_rms"] for r in unique))
(ROOT / "vllm-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k:summary[k] for k in ("planned_evaluations", "completed_three_pass_shapes", "status_counts", "overall")}, indent=2))
