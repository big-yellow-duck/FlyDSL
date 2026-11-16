"""Resumeable, GPU-partitioned execution of the frozen native sweep manifest."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REPO = ROOT / 'FlyDSL-pr1374-review'
PYTHON = ROOT / '.venv/bin/python'
PLAN = ROOT / 'reports/flydsl-native-sweep-plan-20261116/shape-manifest.json'
SCRIPT = REPO / 'tests/perf/bench_rdna4_mxfp4.py'

def jobs():
    manifest=json.loads(PLAN.read_text())
    entries={}
    for shape in manifest['native_shapes']:
        entries[(True,'auto',*shape)]=1
    for shape in manifest['tail_shapes']:
        entries[(False,'auto',*shape)]=1
    for shape in manifest['policy_sensitivity_shapes']:
        for policy in ('hot','rotating'):entries[(True,policy,*shape)]=1
    grouped=collections.defaultdict(list)
    for native,policy,m,n,k in entries:grouped[(native,policy,n,k)].append(m)
    result=[]
    for (native,policy,n,k),ms in sorted(grouped.items()):
        ms=sorted(ms)
        ident=f'{"native" if native else "tails"}-{policy}-n{n}-k{k}'
        result.append(dict(id=ident,native=native,policy=policy,n=n,k=k,ms=ms,
                           weight=len(ms)*3 + sum(m*n*k for m in ms)/1e9))
    loads=[0.,0.]
    for job in sorted(result,key=lambda j:(-j['weight'],j['id'])):
        gpu=min(range(2),key=lambda i:(loads[i],i))
        job['gpu']=gpu;loads[gpu]+=job['weight']
    return result,loads

def atomic_json(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,1]);p.add_argument('--prepare',action='store_true');p.add_argument('--passes',type=int,default=3);p.add_argument('--timeout',type=int,default=1800);p.add_argument('--limit-jobs',type=int);p.add_argument('--retry-failed',action='store_true');a=p.parse_args()
    all_jobs,loads=jobs()
    if a.prepare:
        atomic_json(OUT/'native-jobs.json',dict(plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),jobs=all_jobs,estimated_weights=loads))
        print(json.dumps({'jobs':len(all_jobs),'shapes_per_pass':sum(len(j['ms']) for j in all_jobs),'per_gpu_jobs':[sum(j['gpu']==g for j in all_jobs) for g in range(2)],'weights':loads}));return
    if a.gpu is None:p.error('--gpu required')
    current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    if current!='3ed2ca5f68ecc2fd972cb23d4da366364e91d261':raise RuntimeError('Source changed: '+current)
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise RuntimeError('Source checkout is dirty')
    env=dict(os.environ)
    rocm=subprocess.check_output([str(ROOT/'.venv/bin/rocm-sdk'),'path','--root'],text=True).strip()
    env['ROCM_PATH']=rocm;env['FLYDSL_GPU_ARCH']='gfx1201'
    env['PYTHONPATH']=os.pathsep.join([str(REPO/'python'),str(REPO),str(Path(rocm)/'share/amd_smi')])
    env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1'
    env.pop('CUDA_VISIBLE_DEVICES',None);env.pop('HIP_VISIBLE_DEVICES',None);env.pop('ROCR_VISIBLE_DEVICES',None)
    # Keep the physical-to-logical mapping common to both workers.
    work=OUT/f'gpu{a.gpu}'/'native';work.mkdir(parents=True,exist_ok=True)
    assigned=[j for j in all_jobs if j['gpu']==a.gpu]
    done=0;start=time.time()
    for pass_id in range(1,a.passes+1):
        order=list(assigned);random.Random(1373+pass_id).shuffle(order)
        for job in order:
            key=f'pass{pass_id}-{job["id"]}'
            status=work/(key+'.status.json');output=work/(key+'.json');log=work/(key+'.log')
            if status.exists():
                old=json.loads(status.read_text())
                if old.get('status')=='success' or not a.retry_failed:continue
            ms=list(job['ms']);random.Random(f'{1373+pass_id}-{job["id"]}').shuffle(ms)
            cmd=[str(PYTHON),str(SCRIPT),'--gpu',str(a.gpu),'--m',*map(str,ms),'--nk',f'{job["n"]}x{job["k"]}','--cache-policy',job['policy'],'--output',str(output)]
            if job['native']:cmd.append('--native-fp8')
            # Preserve partial rows from prior attempts separately if retrying.
            if output.exists():output.rename(output.with_suffix(f'.attempt-{int(time.time())}.json'))
            t0=time.time();record=dict(job=job,pass_id=pass_id,command=cmd,source_commit=current,started=t0)
            atomic_json(status,dict(record,status='running'))
            with log.open('w') as f:
                try:
                    run=subprocess.run(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=a.timeout)
                    rc=run.returncode
                except subprocess.TimeoutExpired:rc=124
            row_count=0;validation_error=None
            try:
                rows=json.loads(output.read_text());row_count=len(rows)
                expected={(m,job['n'],job['k'],engine) for m in ms for engine in (['mxfp4','torch_fp8'] if job['native'] else ['mxfp4'])}
                observed={(r['shape'][0],r['shape'][1],r['shape'][2],'torch_fp8' if r['config']['engine']=='torch_fp8' else 'mxfp4') for r in rows}
                assert expected==observed and len(rows)==len(expected), 'row coverage mismatch'
                assert all(r['gpu']==a.gpu for r in rows),'wrong GPU'
                assert all(r['capacities']['cache_size_bytes']==64*1024**2 for r in rows),'cache capacity mismatch'
                assert all(len(r['samples_us'][reg])==16 and r[reg+'_us']>0 for r in rows for reg in ('hot','rotating')),'invalid samples'
            except Exception as e:validation_error=str(e)
            success=rc==0 and validation_error is None
            record.update(status='success' if success else 'failed',exit_code=rc,finished=time.time(),seconds=time.time()-t0,rows=row_count,validation_error=validation_error)
            atomic_json(status,record)
            with (work/'ledger.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            done+=1
            print(json.dumps(dict(gpu=a.gpu,job=key,status=record['status'],seconds=round(record['seconds'],2),rows=row_count,completed_this_run=done,elapsed=round(time.time()-start,1))),flush=True)
            if a.limit_jobs and done>=a.limit_jobs:return
    print(json.dumps(dict(gpu=a.gpu,complete=True,elapsed=time.time()-start)),flush=True)
if __name__=='__main__':main()
