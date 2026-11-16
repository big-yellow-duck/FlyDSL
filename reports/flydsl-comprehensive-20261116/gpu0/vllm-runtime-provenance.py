import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

names = [
    'vllm',
    'vllm._custom_ops',
    'vllm.model_executor.kernels.linear.mxfp4.emulation',
    'vllm.model_executor.kernels.linear.mxfp4.flydsl',
    'vllm._C',
]
mods = []
for name in names:
    try:
        module = importlib.import_module(name)
        filename = getattr(module, '__file__', None)
        resolved = str(Path(filename).resolve()) if filename else None
        sha256 = None
        if resolved and Path(resolved).is_file():
            digest = hashlib.sha256()
            with open(resolved, 'rb') as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    digest.update(block)
            sha256 = digest.hexdigest()
        mods.append({'name': name, 'file': filename, 'resolved_file': resolved,
                     'sha256': sha256, 'error': None})
    except Exception as e:
        mods.append({'name': name, 'file': None, 'resolved_file': None,
                     'sha256': None, 'error': repr(e)})
print(json.dumps({
    'environment': {k: os.environ.get(k) for k in (
        'ROCM_PATH', 'FLYDSL_GPU_ARCH', 'PYTHONPATH', 'VLLM_MXFP4_EMULATION_DEQUANT_AT_LOAD',
        'PYTORCH_ROCM_ARCH', 'MAX_JOBS', 'CXX', 'PATH',
    )},
    'sys_executable': sys.executable,
    'sys_path': sys.path,
    'modules': mods,
}, indent=2))
