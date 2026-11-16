"""Run the frozen 1,842 shapes against both real vLLM linear backends."""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FLY = ROOT / "FlyDSL-pr1374-review"
VLLM = ROOT / "vllm-flydsl-mxfp4-rdna4"
PYTHON = ROOT / ".venv/bin/python"
PLAN = ROOT / "reports/flydsl-native-sweep-plan-20261116/shape-manifest.json"
SCRIPT = OUT / "vllm-audit/bench_mxfp4_pair.py"


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def batches():
    plan = json.loads(PLAN.read_text())
    groups = collections.defaultdict(list)
    for m, n, k in plan["native_shapes"] + plan["tail_shapes"]:
        groups[n, k].append(m)
    assigned = [[], []]
    loads = [0.0, 0.0]
    jobs = []
    for (n, k), ms in groups.items():
        # Balance fixed setup/conditioning, repeated decode, and matrix work.
        cost = len(ms) * 3 + n * k * len(ms) / 1e7 + sum(m*n*k for m in ms) / 1e9
        jobs.append(dict(n=n, k=k, ms=sorted(ms), cost=cost))
    for job in sorted(jobs, key=lambda j: (-j["cost"], j["n"], j["k"])):
        gpu = min((0, 1), key=lambda i: (loads[i], i))
        assigned[gpu].append(job)
        loads[gpu] += job["cost"]
    result = []
    for gpu, work in enumerate(assigned):
        random.Random(1374).shuffle(work)
        for offset in range(0, len(work), 12):
            chunk = work[offset:offset+12]
            shapes = [[m, j["n"], j["k"]] for j in chunk for m in j["ms"]]
            result.append(dict(id=f"gpu{gpu}-batch{offset//12:02d}", gpu=gpu, shapes=shapes))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--gpu", type=int, choices=(0, 1))
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--limit-batches", type=int)
    p.add_argument("--timeout", type=int, default=3600)
    args = p.parse_args()
    jobs = batches()
    if args.prepare:
        save(OUT / "vllm-jobs.json", dict(jobs=jobs, plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest()))
        for job in jobs:
            save(OUT / "vllm-manifests" / (job["id"] + ".json"), dict(native_shapes=job["shapes"], tail_shapes=[]))
        print(json.dumps(dict(batches=len(jobs), shapes=sum(len(j["shapes"]) for j in jobs), per_gpu_shapes=[sum(len(j["shapes"]) for j in jobs if j["gpu"]==g) for g in (0,1)])))
        return
    if args.gpu is None:
        p.error("--gpu required")
    for repo, commit in ((FLY, "3ed2ca5f68ecc2fd972cb23d4da366364e91d261"), (VLLM, "3c138b588506cf2196e5f183d3fe893d973ec2e3")):
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        assert actual == commit, (str(repo), actual)
        assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip(), str(repo)
    script_hash = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert script_hash == "d381f39f98f955a8153bac0e779735b80d35647868c566468f17eb63a3389646", "Unexpected harness revision"
    rocm = subprocess.check_output([str(ROOT / ".venv/bin/rocm-sdk"), "path", "--root"], text=True).strip()
    env = dict(os.environ)
    env.update(ROCM_PATH=rocm, FLYDSL_GPU_ARCH="gfx1201", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VLLM_MXFP4_EMULATION_DEQUANT_AT_LOAD="0")
    env["PATH"] = os.pathsep.join((str(ROOT / ".venv/bin"), str(Path(rocm) / "bin"), env.get("PATH", "")))
    env["CXX"] = str(ROOT / ".venv/bin/x86_64-conda-linux-gnu-c++")
    env["PYTORCH_ROCM_ARCH"] = "gfx1201"
    env["MAX_JOBS"] = "4"
    env["PYTHONPATH"] = os.pathsep.join(map(str, (FLY / "python", FLY, VLLM, Path(rocm) / "share/amd_smi")))
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        env.pop(key, None)
    work = OUT / f"gpu{args.gpu}" / "vllm"
    work.mkdir(parents=True, exist_ok=True)
    assigned = [j for j in jobs if j["gpu"] == args.gpu]
    completed = 0
    for pass_id in range(1, args.passes + 1):
        order = list(assigned)
        random.Random(1373 + pass_id).shuffle(order)
        for job in order:
            name = f"pass{pass_id}-{job['id']}"
            status_path = work / (name + ".status.json")
            output = work / (name + ".json")
            log = work / (name + ".log")
            if status_path.exists():
                old = json.loads(status_path.read_text())
                if old["status"] == "success" or not args.retry_failed:
                    continue
            assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == script_hash, "Harness changed during run"
            if output.exists():
                output.rename(output.with_suffix(f".attempt-{int(time.time())}.json"))
            cmd = [str(PYTHON), str(SCRIPT), "--shape-manifest", str(OUT / "vllm-manifests" / (job["id"] + ".json")), "--gpu", str(args.gpu), "--pass-id", str(pass_id), "--seed", str(20261115 + pass_id), "--rounds", "16", "--flush-mib", "256", "--output", str(output)]
            record = dict(job=job, pass_id=pass_id, command=cmd, harness_sha256=script_hash, started=time.time(), status="running")
            save(status_path, record)
            with log.open("w") as handle:
                try:
                    rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=args.timeout).returncode
                except subprocess.TimeoutExpired:
                    rc = 124
            error = None
            row_count = success_count = unsupported_count = 0
            try:
                data = json.loads(output.read_text())
                rows = data["results"]
                row_count = len(rows)
                expected = {tuple(s) for s in job["shapes"]}
                assert {tuple(r["shape_mnk"]) for r in rows} == expected and len(rows) == len(expected), "coverage mismatch"
                assert data["gpu"] == args.gpu and data["rounds"] == 16, "metadata mismatch"
                assert data["device_capabilities"]["cache_size_bytes"] == 64*1024**2, "cache mismatch"
                for row in rows:
                    if row["status"] not in ("ok", "emulation_unsupported_flydsl_ok"):
                        continue
                    assert all(math.isfinite(x) for x in row["correctness"].values()), "nonfinite correctness metric"
                    assert row["correctness"]["w4a8_flydsl_own_ref_relative_rms"] <= .005
                    if row["status"] == "emulation_unsupported_flydsl_ok":
                        m, n, k = row["shape_mnk"]
                        assert m*k % 64 or n*k % 512, "unexpected unsupported case"
                        unsupported_count += 1
                        backends = ("w4a8_flydsl",)
                    else:
                        backends = ("w4a4_emulation", "w4a8_flydsl")
                        assert row["correctness"]["w4a4_emulation_own_ref_relative_rms"] <= .005
                    for backend in backends:
                        for regime in ("hot", "rotating"):
                            timing = row["timings"]["regimes"][backend][regime]
                            assert len(timing["samples_us"]) == 16
                            assert all(math.isfinite(x) and x > 0 for x in timing["samples_us"])
                        assert row["timings"]["graph_changed_input_relative_rms"][backend] <= .005
                    success_count += 1
                assert success_count == row_count, f"{row_count-success_count} shape failures"
            except Exception as exc:
                error = str(exc)
            record.update(status="success" if rc == 0 and error is None else "failed", exit_code=rc, finished=time.time(), seconds=time.time()-record["started"], rows=row_count, successful_rows=success_count, emulation_unsupported_rows=unsupported_count, validation_error=error)
            save(status_path, record)
            with (work / "ledger.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps({k: record[k] for k in ("status", "pass_id", "seconds", "rows", "successful_rows", "validation_error")}, default=str), flush=True)
            completed += 1
            if args.limit_batches and completed >= args.limit_batches:
                return


if __name__ == "__main__":
    main()
