"""Summarize supplemental decoder toggles separately from the frozen sweeps."""
import collections
import csv
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent / "outlier-diagnostic"
RAW = ROOT.parent / "gpu0/outlier-diagnostic"
expected = {(m,n,k) for m in (1,8) for n,k in ((528,8192),(1040,4096))}
data = collections.defaultdict(dict)
for pass_id in (1,2,3):
    status = json.loads((RAW / f"pass{pass_id}.status.json").read_text())
    assert status["status"] == "success" and status["exit_code"] == 0
    rows = json.loads((RAW / f"pass{pass_id}.json").read_text())
    assert len(rows) == 12
    seen = set()
    for row in rows:
        shape = tuple(row["shape"])
        method = "native_fp8" if row["config"]["engine"] == "torch_fp8" else row["config"]["table_mode"]
        assert shape in expected and method in ("alu","lut","native_fp8")
        assert (shape,method) not in seen
        seen.add((shape,method))
        assert row["gpu"] == 0 and row["capacities"]["cache_size_bytes"] == 64*1024**2
        assert all(len(row["samples_us"][reg])==16 and all(math.isfinite(v) and v>0 for v in row["samples_us"][reg]) for reg in ("hot","rotating"))
        data[shape,method][pass_id] = row
    assert seen == {(shape,method) for shape in expected for method in ("alu","lut","native_fp8")}
result = []
for shape in sorted(expected):
    default = [method for method in ("alu","lut") if data[shape,method][1]["analytical"]]
    assert len(default)==1
    for regime in ("hot","rotating"):
        alu = [data[shape,"alu"][p][regime+"_us"] for p in (1,2,3)]
        lut = [data[shape,"lut"][p][regime+"_us"] for p in (1,2,3)]
        native = [data[shape,"native_fp8"][p][regime+"_us"] for p in (1,2,3)]
        ratios = [l/a for l,a in zip(lut,alu)]
        result.append(dict(m=shape[0],n=shape[1],k=shape[2],regime=regime,default_decoder=default[0],
                           alu_us=statistics.median(alu),lut_us=statistics.median(lut),native_fp8_us=statistics.median(native),
                           lut_over_alu=statistics.median(ratios),min_pass_lut_over_alu=min(ratios),max_pass_lut_over_alu=max(ratios),
                           native_over_alu=statistics.median(n/a for n,a in zip(native,alu)),
                           native_over_lut=statistics.median(n/l for n,l in zip(native,lut))))
with (ROOT / "decoder-comparison.csv").open("w") as f:
    writer=csv.DictWriter(f,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)
(ROOT / "summary.json").write_text(json.dumps(dict(scope="Supplemental diagnostic, excluded from main aggregates; no source or selector changes",passes=3,shapes=4,rows=result),indent=2)+"\n")
print(json.dumps(result,indent=2))
