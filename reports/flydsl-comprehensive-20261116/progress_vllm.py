"""Read completed and partial vLLM batch coverage without GPU work."""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
result = {}
for gpu in (0, 1):
    statuses = collections.Counter()
    rows = collections.Counter()
    passes = collections.Counter()
    running = []
    for path in sorted((ROOT / f"gpu{gpu}" / "vllm").glob("*.status.json")):
        status = json.loads(path.read_text())
        statuses[status["status"]] += 1
        if status["status"] == "running":
            running.append(path.stem)
        output = path.with_name(path.name.replace(".status.json", ".json"))
        if output.exists():
            try:
                data = json.loads(output.read_text())
            except json.JSONDecodeError:
                continue  # Incremental output may be in the middle of a write.
            rows.update(r["status"] for r in data["results"])
            passes[status["pass_id"]] += len(data["results"])
    result[gpu] = dict(batches=dict(statuses), row_statuses=dict(rows), rows_per_pass=dict(passes), running=running)
print(json.dumps(result, indent=2))
