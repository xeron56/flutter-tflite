"""Extract sample timings and tuning results from a Flutter device test log."""
import json
from pathlib import Path
import sys

entries = {}
for line in Path(sys.argv[1]).read_text(errors='replace').splitlines():
    for tag in ['INDRA_DEVICE', 'INDRA_RESULT', 'INDRA_BENCHMARK']:
        marker = tag + ' '
        if marker in line:
            value = json.loads(line.split(marker, 1)[1])
            entries.setdefault(tag.removeprefix('INDRA_').lower(), []).append(value)
output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('benchmark-results.json')
output.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + '\n')
print(output)
