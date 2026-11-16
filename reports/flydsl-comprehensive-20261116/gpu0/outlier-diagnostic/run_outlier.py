import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parents[4]
repo = root / 'FlyDSL-pr1374-review'
python = root / '.venv/bin/python'
bench = repo / 'tests/perf/bench_rdna4_mxfp4.py'
configs = root / 'reports/flydsl-comprehensive-20261116/outlier-diagnostic/configs.json'
out = Path(__file__).resolve().parent
rocm = subprocess.check_output([str(root / '.venv/bin/rocm-sdk'), 'path', '--root'], text=True).strip()
env = dict(os.environ)
env.update(
    ROCM_PATH=rocm,
    FLYDSL_GPU_ARCH='gfx1201',
    PYTHONPATH=os.pathsep.join(map(str, (repo / 'python', repo, Path(rocm) / 'share/amd_smi'))),
    OMP_NUM_THREADS='1',
    MKL_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',
)
for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
    env.pop(key, None)
for pass_id in (1, 2, 3):
    name = f'pass{pass_id}'
    output = out / f'{name}.json'
    log = out / f'{name}.log'
    status_path = out / f'{name}.status.json'
    cmd = [str(python), str(bench), '--gpu', '0', '--m', '1', '8', '--nk', '528x8192', '1040x4096', '--native-fp8', '--cache-policy', 'auto', '--configs', str(configs), '--output', str(output)]
    record = {
        'pass_id': pass_id,
        'command': cmd,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'source_clean': not subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip(),
        'benchmark_sha256': hashlib.sha256(bench.read_bytes()).hexdigest(),
        'configs_sha256': hashlib.sha256(configs.read_bytes()).hexdigest(),
        'started': time.time(),
        'status': 'running',
    }
    status_path.write_text(json.dumps(record, indent=2) + '\n')
    with log.open('w') as handle:
        run = subprocess.run(cmd, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT)
    record.update(status='success' if run.returncode == 0 else 'failed', exit_code=run.returncode, finished=time.time(), seconds=time.time() - record['started'])
    try:
        rows = json.loads(output.read_text())
        record['rows'] = len(rows)
        record['shape_config_coverage'] = len({(tuple(row['shape']), row['config']['engine'], row['config'].get('table_mode')) for row in rows})
        record['row_count_valid'] = len(rows) == 12 and record['shape_config_coverage'] == 12
        if not record['row_count_valid']:
            record['status'] = 'failed'
            record['validation_error'] = 'expected 12 shape/config rows'
    except Exception as exc:
        record['rows'] = None
        record['status'] = 'failed'
        record['validation_error'] = repr(exc)
    status_path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record.get(k) for k in ('pass_id', 'status', 'exit_code', 'seconds', 'rows', 'shape_config_coverage', 'validation_error')}), flush=True)
    if record['status'] != 'success':
        raise SystemExit(run.returncode or 1)
