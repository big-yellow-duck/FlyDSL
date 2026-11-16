import json
import runpy
import sys
import time
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[4]
script = root / 'FlyDSL-pr1374-review/tests/perf/bench_rdna4_mxfp4.py'
out = Path(__file__).resolve().parent / 'largest-native-profile.json'
torch.cuda.set_device(0)
torch.cuda.synchronize(0)
torch.cuda.reset_peak_memory_stats(0)
start = time.time()
print(json.dumps({
    'profile': 'supplemental_largest_native_shape',
    'shape_mnk': [8192, 16384, 16384],
    'gpu': 0,
    'torch_version': torch.__version__,
    'hip_version': torch.version.hip,
    'baseline_allocated_bytes': torch.cuda.memory_allocated(0),
    'baseline_reserved_bytes': torch.cuda.memory_reserved(0),
    'started_unix': start,
}), flush=True)
sys.argv = [
    str(script), '--gpu', '0', '--m', '8192', '--nk', '16384x16384',
    '--native-fp8', '--output', str(out),
]
runpy.run_path(str(script), run_name='__main__')
torch.cuda.synchronize(0)
print(json.dumps({
    'profile': 'supplemental_largest_native_shape',
    'shape_mnk': [8192, 16384, 16384],
    'gpu': 0,
    'peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
    'peak_reserved_bytes': torch.cuda.max_memory_reserved(0),
    'final_allocated_bytes': torch.cuda.memory_allocated(0),
    'final_reserved_bytes': torch.cuda.memory_reserved(0),
    'elapsed_seconds': time.time() - start,
    'benchmark_json': str(out),
}), flush=True)
