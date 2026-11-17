# Indra OCR for Android / Flutter

An offline Android implementation of Datalab's Indra v1.1.0 image OCR, with a Flutter UI and a Kotlin inference engine. Created for the REDMI K90 (Snapdragon 8 Elite / SM8750), with tuning performed on the actual device rather than assuming an NPU is faster.

## Run

```sh
cd /Users/mdshahidulislam/Documents/resource/flutter-tflite/example/indra_ocr
python3 tool/download_models.py
uv run --python 3.12 --with onnx==1.23.2 --with onnxruntime==1.31.0 --with numpy==2.5.4 tool/prepare_models.py
flutter pub get
flutter run --release -d 3343ba1a
```

The prepared models are already included locally; download/preparation is needed only to reproduce them. The downloader is repeatable, uses an immutable Hugging Face revision, verifies sizes and SHA-256 hashes, and replaces files atomically. No API key, server, or runtime model download is needed. The release app has no Internet permission. Camera opens the system camera app; gallery uses Android's picker.

This project targets ARM64 Android 8.0+ (API 26+). Build an APK for the K90:

```sh
flutter build apk --release --target-platform android-arm64
# build/app/outputs/flutter-apk/app-release.apk
```

Release shrinking retains the ONNX/OpenCV JNI classes through `android/app/proguard-rules.pro`. The release manifest explicitly removes dependency-contributed networking permissions; debug/profile retain the permissions needed by Flutter tooling.

The local release APK uses Flutter's debug signing key for installation/testing. Configure your own signing key before store distribution.

## Use

1. Choose an image, take a photo, or select an English, Bengali, or Korean sample.
2. Tap **Create document** to extract the recognized text as Markdown and open the document reader. Switch between **Preview**, **Markdown**, and **Source image**. The reader supports headings, lists, quotes, code, and Markdown tables; edit the source to correct formatting.
3. Use **Copy Markdown** or **Save .md file** in the document toolbar. Saving opens Android’s system file picker and writes UTF-8 without storage permission. The menu keeps the unchanged **Original OCR text** available. Edits stay while returning to the scanner; save a file before closing the app.
4. Enable **Small text mode** for dense pages. It runs a second detector pass, up to 2048 pixels, only when the detected lines are small.
5. Tap **Tune on this image**. The app compares CPU 1/2/4/6/8 threads, XNNPACK 4/6 threads, and NNAPI. Each candidate gets a cold run plus three warm runs. The fastest warm median with text identical to the CPU baseline is saved locally. Blank images cannot select a winner.

Tuning is workload-dependent. Use a typical full page, keep the phone cool, and retune if your document type changes. Timings include decoding, preprocessing, inference, and postprocessing in the native engine; they exclude image selection and Flutter rendering. The first scan includes loading the models used by that page.

## Performance implementation

- Recognizer graphs prune unused language-model TopK outputs while retaining FP32 weights and the exact greedy outputs; preparation verifies bit-identical results on three synthetic batches per recognizer. Original weights stay outside the APK in `tool/model_sources`.
- ONNX Runtime Android 1.31.0; OpenCV 4.12.0 for native image operations.
- One background worker, no model work on Flutter's UI thread, and no concurrent OCR queue growth.
- Lazy persistent model sessions, versioned on-device model files, deterministic close of results/tensors, bounded image decoding (4096-pixel longest side).
- Indra's grayscale ink normalization, detector padding, connected components, line margins, tall Latin context windows, dark-text inversion, script routing and CTC greedy decoding.
- Recognition grouped by similar widths, up to 16 lines / 8192 padded columns per batch. Avoids wasting inference on padding and limits temporary memory.
- CPU arena disabled, following upstream's variable-shape memory-growth finding; spin-wait disabled to reduce idle power and contention.
- EXIF transforms and model-based 90/180/270-degree orientation correction; Korean jamo normalized to NFC, Indra-style RTL visual-to-logical conversion, geometric recursive reading order.
- Real per-stage times and thread/backend labels. Development logs use the `IndraOCR` tag; ordinary OCR logs contain timings/counts, not document text.

## GPU / NPU support

The K90 has an Adreno GPU and a Hexagon NPU, but having them does not make an arbitrary ONNX graph compatible or faster. These Indra models use dynamic widths/batch shapes, and recognizers include operations that can split execution across providers. CPU is the portable default.

NNAPI is an **experimental request**, with its CPU device disabled and ONNX Runtime CPU fallback retained. Selecting it does **not** prove the GPU or NPU ran any particular layer. The UI explicitly says CPU fallback may occur. XNNPACK is a CPU provider and may also fall back for unsupported operations. Tuning rejects candidates with different decoded text, but this is a one-image consistency check, not a corpus-level accuracy evaluation.

Direct Qualcomm acceleration is possible, but is not implemented in this APK:

- **Adreno GPU:** QNN GPU accepts floating-point models (FP32/FP16). It needs a QNN-enabled Android ONNX Runtime build and compatible fixed-shape graphs.
- **Hexagon NPU / HTP:** QNN HTP needs quantized models with fixed shapes. Calibrate using representative English, Bengali, CJK and RTL document crops, then measure character error rates as well as latency.
- Indra’s variable batch sizes and line widths must become fixed buckets with padding and valid-length handling. Start with orientation/script models and detector, then the four recognizers; retain the verified CPU implementation for unsupported shapes/operators.
- Bundle the SDK’s appropriate Android runtime libraries, inspect provider assignments and profiling, and initially disable CPU fallback to expose unsupported graphs. Cache compiled contexts to reduce startup cost. Select acceleration only after end-to-end measurements on the K90, including preprocessing, transfers, cold start, and sustained heat.

See [ONNX Runtime’s QNN documentation](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html). GPU + NPU is not an automatic combined speed boost: partitioning and transfers can outweigh acceleration for these small models. The existing CPU/8 path remains the fastest **measured** configuration on this phone.

## Markdown document workflow (v1.1)

All nonempty recognized lines are included, in native reading order. A local geometric formatter infers headings from line height/spacing, groups paragraphs, and recognizes bullet/numbered lists. It escapes literal Markdown/HTML punctuation so recognized symbols are not accidentally interpreted as formatting. There is no summarization, translation, extra model, or network call. The original OCR string is retained separately, including original bullet characters.

The reader provides selectable text, adjustable type size, light/dark themes, and a zoomable source image. Markdown edits update the preview and exported file; they do not modify the original OCR text. Markdown images are displayed as placeholders and tapping a link copies it, keeping the app offline. Automatic layout is heuristic: tables are supported in the viewer/editor, but scan-to-table reconstruction is not implemented. OCR omissions/errors still need review against the image.

Validation: eight Flutter unit/widget tests cover all-line preservation, multilingual text, literal punctuation, inferred lists/headings, preview/edit/copy/export, original text, save cancel/error recovery, and a 360-pixel screen at 2× text scale.

The v1.1 release was also tested on the K90: the complete 35-line Korean sample exported through Android’s file picker, and every original recognized line was verified in the UTF-8 file. See [`tool/document-k90-validation.json`](tool/document-k90-validation.json).

## Scope compared with the Python package

This is a fast native **image-to-text port**, not the entire desktop document parser. It bundles detection, orientation, script classification, and all four recognizers (Latin, CJK, RTL, world, including Bengali). It returns text, line boxes/confidence, and timings over the platform channel.

It does not include n-gram language-model beam correction, learned layout/table reconstruction, PDF text extraction/rasterization, optional math decoding, or arbitrary-angle deskew. For best results, photograph the document straight and close. Handwriting, complex columns/tables and skewed scans can be less accurate than upstream's complete pipeline. No claim is made that upstream benchmark accuracy/speed carries over to this port.

## Checks

```sh
flutter analyze
flutter test
(cd android && ./gradlew :app:testDebugUnitTest -Ptarget-platform=android-arm64)
flutter test integration_test/ocr_device_test.dart -d 3343ba1a
flutter drive --profile --driver=test_driver/integration_test.dart --target=integration_test/ocr_device_test.dart -d 3343ba1a
```

The device test executes the real bundled models on English, Bengali, Korean, dark, rotated, and blank inputs; checks failure recovery; and benchmarks providers. It prints `INDRA_RESULT` and `INDRA_BENCHMARK` JSON using bundled/generated samples only. The UI test covers the empty state, disabled recognition before an image is chosen, and a small screen at 2× text scale.

## Measured on this K90 (2026-10-25)

**OCR engine baseline release (v1.0):** through the real app UI, the 35-line Korean page took **652.8 ms** on 8 CPU threads (warm median), compared with 749.4 ms on CPU/4, 754.5 ms for requested NNAPI/4, and 1317.0 ms for XNNPACK/4. All eight candidates completed and returned the same text; CPU/8 was saved. Detailed release measurements and APK hash: [`tool/benchmark-k90-release.json`](tool/benchmark-k90-release.json).

The release APK also passed permission inspection (no Internet/network-state permission), ARM64-only packaging inspection, and ZIP 16 KB alignment verification. Three native unit tests passed (RTL digit order, combining-mark roundtrip, column reading order).

The earlier **profile-build** benchmark below is retained separately; do not treat it as the release latency or compare runs as a controlled speedup experiment.

Physical device `2510DRK44C`, SM8750, Android 16, approximately 12 GB RAM. Flutter **profile** build, native ONNX/OpenCV pipeline, bundled 1653×2337 Korean scan (35 detected lines), normal mode. Each entry is the median of three warm runs after one warm-up/cold-session run. All candidates produced identical text on this image.

| Requested engine | CPU threads | Warm median |
| --- | ---: | ---: |
| CPU | 1 | 1095.1 ms |
| CPU | 2 | 758.1 ms |
| CPU | 4 | 673.9 ms |
| CPU | 6 | 606.2 ms |
| CPU | 8 | 569.6 ms |
| XNNPACK | 4 | 718.9 ms |
| XNNPACK | 6 | 754.3 ms |
| NNAPI | 4 | 676.8 ms |

The SM8750 default is therefore 8 CPU threads; other chips start with 4. The phone can retune and save a different configuration for its workload. These numbers are sample measurements, not guaranteed latency on arbitrary documents or a proof of NPU/GPU use. The phone was USB-connected and charging, and these short runs do not characterize sustained thermal behavior.

The six correctness samples matched the **entire text output** from the original unpruned FP32 models, including the Korean document. First/cold sample times on 4 threads were 291 ms for English and 215 ms for Bengali (the latter includes loading its recognizer). Warm dark/rotated English samples took 165/146 ms. These small generated pages should not be compared directly with the full-page benchmark.

The v1.0 release APK was additionally exercised through its real UI on the K90: Bengali sample text was correct, with 368 ms first-run and 146 ms warm latency. Native RTL and column-order unit tests also pass.

Full results: [`tool/benchmark-k90.json`](tool/benchmark-k90.json). Original-graph debug baseline: [`tool/benchmark-baseline.json`](tool/benchmark-baseline.json). The baseline and final runs used different build modes and thermal conditions; they are not a controlled speedup experiment.

## Source and licensing

- [Indra source](https://github.com/datalab-to/indra), referenced commit `e20355319b5aa5705a876a8d2b40aca606d7df5f`.
- [Indra models](https://huggingface.co/datalab-to/indra-ocr), v1.1.0 / `bd02167fdf2005a19962cc15188f8496a2c87d3b`.
- Adapted algorithms: `det_post.py`, `rec_preproc.py`, `pipeline.py`, `rtl.py`; upstream Apache-2.0 license in `third_party/INDRA_CODE_LICENSE`.
- Model weights use Datalab's **modified OpenRAIL-M** license, included at `android/app/src/main/assets/models/LICENSE`. It includes commercial/revenue and competitive-use restrictions; read the actual license before distributing or commercializing the app.
- The bundled Korean scan comes from upstream `examples/samples/scan.png`. Generated English/Bengali samples use Android Canvas.
- [Xiaomi REDMI K90](https://www.mi.com/shop/buy/detail?cfrom=search&product_id=22172)
- [ONNX Runtime mobile](https://onnxruntime.ai/docs/tutorials/mobile/), [NNAPI](https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html), [QNN](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
