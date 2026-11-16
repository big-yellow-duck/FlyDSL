# Reproduce the arbitrary-shape sweep

The [full report](REPORT.md) measures FlyDSL `3ed2ca5f68ecc2fd972cb23d4da366364e91d261` and vLLM `3c138b588506cf2196e5f183d3fe893d973ec2e3`. Use those revisions to reproduce the published numbers. Later E5M2-only code is a different implementation.

## Inspect or recompute the published results without a GPU

```bash
git clone --branch pr1374-benchmark-results-20261116 \
  https://github.com/big-yellow-duck/FlyDSL.git FlyDSL-benchmark-results
cd FlyDSL-benchmark-results/reports/flydsl-comprehensive-20261116

# Verify the published artifact checksums, then restore the original raw layout.
sha256sum -c SHA256SUMS
tar -xzf raw/gpu0.tar.gz
tar -xzf raw/gpu1.tar.gz

# These aggregators use only the Python standard library.
python summarize_native.py
python summarize_vllm.py
```

Expected native totals: 1,266 successful grouped jobs; 1,242 unique auto-selected native pairs; 600 FlyDSL-only N-tail shapes; and 120 additional shape/policy combinations. There are 9,972 method rows across three passes. Expected vLLM totals: 102 successful batches; 1,158 paired shapes; 684 explicit emulation rejections with validated FlyDSL timing; and 5,526 shape rows across three passes. Check for missing or failed cases, not just the speedup aggregate.

The raw archives contain all native/vLLM samples, logs, status files and ledgers. Each method has 16 timing samples per cache regime per pass. Unsupported emulation cases retain the exact rejection message. CSVs preserve per-shape latency summaries and all three pass ratios. The [shape manifest](../flydsl-native-sweep-plan-20261116/shape-manifest.json) lists every shape, and the [original plan](../flydsl-native-sweep-plan-20261116/rdna4_mxfp4_sweep.md) explains the cohorts. Its “proposal” status is retained as historical planning evidence; the report and ledgers establish completion.

## Run representative native comparisons

With a built FlyDSL runtime matching the measured source revision, ROCm PyTorch, and AMD SMI Python bindings, run from the FlyDSL source checkout. Set `ROCM_PATH` to your ROCm SDK root.

```bash
export FLYDSL_GPU_ARCH=gfx1201
export PYTHONPATH="$PWD/python:$PWD:$ROCM_PATH/share/amd_smi${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p /tmp/flydsl-pr1374-native

# --nk means N x K; every invocation measures both cache conditions.
for pass in 1 2 3; do
  python tests/perf/bench_rdna4_mxfp4.py \
    --gpu 0 --m 1 8 16 32 128 512 2048 8192 \
    --nk 4096x4096 17408x5120 --native-fp8 \
    --output "/tmp/flydsl-pr1374-native/pass${pass}.json"
done

# Largest observed native hot-cache regression.
python tests/perf/bench_rdna4_mxfp4.py \
  --gpu 0 --m 8 --nk 528x8192 --native-fp8 \
  --output /tmp/flydsl-pr1374-native/regression.json

# N-tail correctness/timing; no unsupported native baseline is invented.
python tests/perf/bench_rdna4_mxfp4.py \
  --gpu 0 --m 1 7 73 257 --nk 513x1120 \
  --output /tmp/flydsl-pr1374-native/tails.json

python -m pytest tests/kernels/test_rdna4_mxfp4.py -q
```

These shapes are illustrative subsets, not the full aggregate.

## Run the complete sweep on two GPUs

The frozen runners expect a workspace containing:

```text
workspace/
├── .venv/bin/python
├── FlyDSL-pr1374-review/          # measured FlyDSL revision, clean checkout
├── vllm-flydsl-mxfp4-rdna4/       # measured vLLM revision, clean checkout
└── reports/
    ├── flydsl-native-sweep-plan-20261116/
    └── flydsl-repro-.../         # fresh runner/harness files and outputs
```

Provision `.venv` and the compiled libraries before running: the recorded environment used Torch `2.13.0+rocm10.0.0`, Quark `0.12.post1`, the `rocm-sdk` command, AMD SMI bindings, and `x86_64-conda-linux-gnu-c++`. The FlyDSL compiler must match the measured source. Exact identities and paths are recorded in [provenance](provenance.json), [package metadata](package-versions.json), and [runtime module provenance](gpu0/vllm-runtime-provenance.json). The recorded vLLM Python source used an existing archived `_C` extension; these commands do not recreate that binary or install a stock vLLM package. Reproduction on another software build must record that difference.

The vLLM harness supplies the source namespace adapter used in this experiment. It times real `apply_weights` implementations with synthetic weights and activations. It does not require checkpoint weight downloads and does not measure model throughput.

From the root of the results checkout, set `WORKSPACE` to the absolute path of your prepared workspace:

```bash
set -e
ARTIFACTS="$PWD/reports/flydsl-comprehensive-20261116"
WORKSPACE=/absolute/path/to/prepared/workspace

# If the source directories already exist, verify their revisions instead.
git clone https://github.com/big-yellow-duck/FlyDSL.git "$WORKSPACE/FlyDSL-pr1374-review"
git -C "$WORKSPACE/FlyDSL-pr1374-review" checkout --detach 3ed2ca5f68ecc2fd972cb23d4da366364e91d261
git clone https://github.com/big-yellow-duck/vllm.git "$WORKSPACE/vllm-flydsl-mxfp4-rdna4"
git -C "$WORKSPACE/vllm-flydsl-mxfp4-rdna4" checkout --detach 3c138b588506cf2196e5f183d3fe893d973ec2e3

mkdir -p "$WORKSPACE/reports"
cp -a "$ARTIFACTS/../flydsl-native-sweep-plan-20261116" "$WORKSPACE/reports/"
RUN_DIR=$(mktemp -d "$WORKSPACE/reports/flydsl-repro-XXXXXX")
mkdir -p "$RUN_DIR/vllm-audit"
cp "$ARTIFACTS"/{run_native_sweep.py,run_vllm_sweep.py,summarize_native.py,summarize_vllm.py} "$RUN_DIR/"
cp "$ARTIFACTS/vllm-audit/bench_mxfp4_pair.py" "$RUN_DIR/vllm-audit/"
PYTHON="$WORKSPACE/.venv/bin/python"

run_pair() {
  local runner="$1" p0 p1 s0=0 s1=0
  "$PYTHON" "$RUN_DIR/$runner" --gpu 0 --passes 3 > "$RUN_DIR/$runner.gpu0.log" 2>&1 &
  p0=$!
  "$PYTHON" "$RUN_DIR/$runner" --gpu 1 --passes 3 > "$RUN_DIR/$runner.gpu1.log" 2>&1 &
  p1=$!
  wait "$p0" || s0=$?
  wait "$p1" || s1=$?
  test "$s0" -eq 0 && test "$s1" -eq 0
}

"$PYTHON" "$RUN_DIR/run_native_sweep.py" --prepare
run_pair run_native_sweep.py
"$PYTHON" "$RUN_DIR/summarize_native.py"

"$PYTHON" "$RUN_DIR/run_vllm_sweep.py" --prepare
run_pair run_vllm_sweep.py
"$PYTHON" "$RUN_DIR/summarize_vllm.py"
```

Use two otherwise idle gfx1201 GPUs with sufficient memory. The runners keep both devices visible, assign each case to one GPU, verify source revisions, and retain the same assignment across passes. They explicitly disable load-time emulation weight decoding. The fresh output directory prevents archived successful statuses from causing the run to skip measurements. Verify all expected counts and failures in the resulting summaries.

The native runner retains the measured revision's extra policy-sensitivity jobs for historical reproduction. Those extra jobs are excluded from the headline auto-selection aggregate; the later audit found identical configs on all 60 sensitivity shapes.
