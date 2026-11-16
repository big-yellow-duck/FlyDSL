import json,time
from pathlib import Path
root=Path(__file__).resolve().parent
out={}
for gpu in [0,1]:
 files=list((root/f'gpu{gpu}/native').glob('*.status.json'))
 states=[]
 for p in files:
  try: states.append(json.loads(p.read_text()))
  except Exception: pass
 complete=[s for s in states if s['status'] in ('success','failed')]
 successes=[s for s in complete if s['status']=='success']
 out[str(gpu)]={'success':len(successes),'failed':len(complete)-len(successes),'running':[s['job']['id'] for s in states if s['status']=='running'],'rows':sum(s.get('rows',0) for s in successes),'evaluations':sum(len(s['job']['ms']) for s in successes),'passes':{str(i):sum(s['pass_id']==i for s in successes) for i in [1,2,3]}}
print(json.dumps(out,indent=2))
