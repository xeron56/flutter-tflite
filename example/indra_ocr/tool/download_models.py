"""Fetch the pinned Indra weights; curl uses the host's TLS trust store."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / 'tool/models.lock.json').read_text())
DEST = ROOT / 'android/app/src/main/assets/models'

def fetch(item):
    name, spec = item
    dest = (ROOT / 'tool/model_sources' if name.startswith('rec') and name.endswith('.onnx') else DEST) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    def valid(path):
        return path.exists() and path.stat().st_size == spec['size'] and (
            not spec['sha256'] or hashlib.sha256(path.read_bytes()).hexdigest() == spec['sha256'])
    if not valid(dest):
        part = dest.with_suffix(dest.suffix + '.part')
        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                        '--retry', '3', '--max-time', '180',
                        f"https://huggingface.co/datalab-to/indra-ocr/resolve/{LOCK['revision']}/{name}",
                        '-o', str(part)], check=True)
        if not valid(part):
            part.unlink(missing_ok=True)
            raise RuntimeError(f'Integrity check failed: {name}')
        part.replace(dest)
    print(f'Verified {name}: {dest.stat().st_size:,} bytes', flush=True)

if __name__ == '__main__':
    DEST.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, LOCK['files'].items()))
