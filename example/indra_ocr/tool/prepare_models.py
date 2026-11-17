"""Prune LM-only outputs, retaining the exact FP32 greedy recognizer.
Run with uv run --python 3.12 --with onnx==1.23.2 --with onnxruntime==1.31.0 --with numpy==2.5.4 tool/prepare_models.py.
Source weights remain unchanged outside the APK. Validate output equivalence.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'android/app/src/main/assets/models'
SOURCES = ROOT / 'tool/model_sources'
OUTPUTS = ['ids', 'conf', 'lengths']

def prepare():
    report = {'kind': 'greedy-only FP32 graph pruning; no quantization', 'models': {}}
    for name in ['rec.onnx', 'rec_cjk.onnx', 'rec_rtl.onnx', 'rec_world.onnx']:
        source = SOURCES / name
        lock = json.loads((ROOT / 'tool/models.lock.json').read_text())
        if hashlib.sha256(source.read_bytes()).hexdigest() != lock['files'][name]['sha256']:
            raise RuntimeError(f'Source checksum mismatch: {name}; run download_models.py')
        destination = ASSETS / name
        temporary = destination.with_suffix('.tmp.onnx')
        onnx.utils.extract_model(str(source), str(temporary), ['x', 'widths'], OUTPUTS)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        original = ort.InferenceSession(str(source), options, providers=['CPUExecutionProvider'])
        pruned = ort.InferenceSession(str(temporary), options, providers=['CPUExecutionProvider'])
        height = original.get_inputs()[0].shape[2]
        rng = np.random.default_rng(42)
        # Dynamic batch sizes, asymmetric widths, zero-padding, and empty ink.
        for batch, width in [(1, 64), (3, 256), (2, 640)]:
            x = rng.random((batch, 1, height, width), dtype=np.float32)
            widths = np.array([width - 8 * i for i in range(batch)], dtype=np.int64)
            for i, w in enumerate(widths):
                x[i, :, :, w:] = 0
            if batch == 1:
                x.fill(0)
            inputs = {'x': x, 'widths': widths}
            expected = original.run(OUTPUTS, inputs)
            actual = pruned.run(OUTPUTS, inputs)
            for a, b in zip(expected, actual):
                np.testing.assert_allclose(a, b, rtol=0, atol=0)
        onnx.checker.check_model(str(temporary))
        temporary.replace(destination)
        before = onnx.load(str(source)); after = onnx.load(str(destination))
        report['models'][name] = {
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'nodes_before': len(before.graph.node), 'nodes_after': len(after.graph.node),
            'output_parity': 'bit-identical on 3 synthetic batches; device sample test also required',
        }
        print(name, report['models'][name], flush=True)
    (ASSETS / 'mobile.json').write_text(json.dumps(report, indent=2) + '\n')

if __name__ == '__main__':
    prepare()
