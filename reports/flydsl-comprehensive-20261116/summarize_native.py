"""Aggregate validated paired rows without mixing GPUs or revisions."""
import collections,csv,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
plan=json.loads((ROOT.parent/'flydsl-native-sweep-plan-20261116/shape-manifest.json').read_text())
cohorts={k:{tuple(s) for s in v} for k,v in plan['cohorts'].items()}
paired=collections.defaultdict(dict);tail=collections.defaultdict(dict);failures=[];statuses=[]
for gpu in (0,1):
 for p in sorted((ROOT/f'gpu{gpu}'/'native').glob('*.status.json')):
  status=json.loads(p.read_text());statuses.append(status)
  if status['status']!='success':
   if status['status']=='failed':failures.append({'file':str(p),'status':status})
   continue
  data=json.loads(p.with_name(p.name.replace('.status.json','.json')).read_text())
  assert all(len(r['samples_us'][reg])==16 and all(math.isfinite(x) and x>0 for x in r['samples_us'][reg]) for r in data for reg in ('hot','rotating')), str(p)
  byshape=collections.defaultdict(dict)
  for row in data:byshape[tuple(row['shape'])]['native_fp8' if row['config']['engine']=='torch_fp8' else 'mxfp4']=row
  for shape,methods in byshape.items():
   identity=(shape,status['job']['policy'],gpu)
   if status['job']['native']:
    assert set(methods)=={'mxfp4','native_fp8'}
    paired[identity][status['pass_id']]=methods
   else:tail[identity][status['pass_id']]=methods['mxfp4']

def band(m):return 'M<=16' if m<=16 else '17<=M<=128' if m<=128 else '129<=M<=1024' if m<=1024 else 'M>1024'
def gm(xs):return math.exp(statistics.mean(map(math.log,xs))) if xs else None
rows=[]
for (shape,policy,gpu),passes in sorted(paired.items()):
 if set(passes)!={1,2,3}:continue
 configs={json.dumps(v['mxfp4']['config'],sort_keys=True) for v in passes.values()};assert len(configs)==1
 for regime in ('hot','rotating'):
  ratios=[passes[p]['native_fp8'][regime+'_us']/passes[p]['mxfp4'][regime+'_us'] for p in (1,2,3)]
  row=dict(m=shape[0],n=shape[1],k=shape[2],policy=policy,gpu=gpu,regime=regime,m_band=band(shape[0]),cohorts=';'.join(k for k,v in cohorts.items() if shape in v),engine=passes[1]['mxfp4']['config']['engine'],mxfp4_us=statistics.median(passes[p]['mxfp4'][regime+'_us'] for p in (1,2,3)),native_fp8_us=statistics.median(passes[p]['native_fp8'][regime+'_us'] for p in (1,2,3)),speedup=statistics.median(ratios),min_pass_speedup=min(ratios),max_pass_speedup=max(ratios),classification='consistent_win_gt5pct' if min(ratios)>1.05 else 'consistent_loss_gt5pct' if max(ratios)<.95 else 'small_or_inconsistent',**{f'pass{p}_speedup':ratios[p-1] for p in (1,2,3)})
  rows.append(row)
if rows:
 with (ROOT/'native-paired.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
tailrows=[]
for (shape,policy,gpu),passes in sorted(tail.items()):
 if set(passes)!={1,2,3}:continue
 tailrows.append(dict(m=shape[0],n=shape[1],k=shape[2],gpu=gpu,policy=policy,hot_us=statistics.median(v['hot_us'] for v in passes.values()),rotating_us=statistics.median(v['rotating_us'] for v in passes.values())))
if tailrows:
 with (ROOT/'native-tail-only.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(tailrows[0]));w.writeheader();w.writerows(tailrows)
def describe(sub):
 return dict(count=len(sub),geomean_speedup=gm([r['speedup'] for r in sub]),median_speedup=statistics.median(r['speedup'] for r in sub) if sub else None,min_speedup=min((r['speedup'] for r in sub),default=None),max_speedup=max((r['speedup'] for r in sub),default=None),consistent_wins=sum(r['classification']=='consistent_win_gt5pct' for r in sub),consistent_losses=sum(r['classification']=='consistent_loss_gt5pct' for r in sub),small_or_inconsistent=sum(r['classification']=='small_or_inconsistent' for r in sub))
summary={'planned_evaluations':5886,'status_counts':dict(collections.Counter(s['status'] for s in statuses)),'completed_native_shape_policy_gpu_keys':len(rows)//2,'completed_tail_keys':len(tailrows),'failed_jobs':failures,'auto':{},'by_cohort':{},'by_m_band':{},'by_policy':{},'worst_auto':{}}
for reg in ('hot','rotating'):
 auto=[r for r in rows if r['policy']=='auto' and r['regime']==reg]
 summary['auto'][reg]=describe(auto)
 summary['by_cohort'][reg]={c:describe([r for r in auto if c in r['cohorts'].split(';')]) for c in cohorts if c!='G-n-tails'}
 summary['by_m_band'][reg]={b:describe([r for r in auto if r['m_band']==b]) for b in ('M<=16','17<=M<=128','129<=M<=1024','M>1024')}
 summary['by_policy'][reg]={p:describe([r for r in rows if r['policy']==p and r['regime']==reg]) for p in ('auto','hot','rotating')}
 summary['worst_auto'][reg]=sorted(auto,key=lambda r:r['speedup'])[:15]
(ROOT/'native-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['status_counts','completed_native_shape_policy_gpu_keys','completed_tail_keys','auto']},indent=2))
