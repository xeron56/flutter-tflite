import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

enum HardwareAccelerator {
  cpu,
  gpu,
  npu,
}

class OcrResult {
  final List<Point<int>> box;
  final String text;
  final double confidence;

  OcrResult({
    required this.box,
    required this.text,
    required this.confidence,
  });
}

class EasyOcrService {
  Interpreter? _detector;
  Interpreter? _recognizer;
  String _currentLanguage = 'en';
  HardwareAccelerator _currentAccelerator = HardwareAccelerator.cpu;

  double _detectorInferenceTimeMs = 0.0;
  double _recognizerInferenceTimeMs = 0.0;
  double _totalRecognizerTimeMs = 0.0;

  // EasyOCR English Gen2 character vocabulary (index 0 is CTCblank)
  static const String _enVocabulary =
      '0123456789!"#\$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';

  // EasyOCR Bengali Gen1 character vocabulary (index 0 is CTCblank)
  static const String _bnVocabulary =
      '0123456789!"#\$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ।ঁংঃঅআইঈউঊঋঌএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ািীুূৃেৈোৌ্ৎড়ঢ়য়০১২৩৪৫৬৭৮৯';

  HardwareAccelerator _detectorAccelerator = HardwareAccelerator.cpu;
  HardwareAccelerator _recognizerAccelerator = HardwareAccelerator.cpu;

  bool get isLoaded => _detector != null && _recognizer != null;
  String get currentLanguage => _currentLanguage;
  HardwareAccelerator get currentAccelerator => _currentAccelerator;
  HardwareAccelerator get detectorAccelerator => _detectorAccelerator;
  HardwareAccelerator get recognizerAccelerator => _recognizerAccelerator;
  double get detectorInferenceTimeMs => _detectorInferenceTimeMs;
  double get recognizerInferenceTimeMs => _recognizerInferenceTimeMs;
  double get totalRecognizerTimeMs => _totalRecognizerTimeMs;

  Future<void> loadModels(String language, {HardwareAccelerator accelerator = HardwareAccelerator.cpu}) async {
    _currentLanguage = language;
    _currentAccelerator = accelerator;
    _detector?.close();
    _recognizer?.close();
    _detector = null;
    _recognizer = null;

    final options = InterpreterOptions();
    switch (accelerator) {
      case HardwareAccelerator.cpu:
        // Default CPU options
        break;
      case HardwareAccelerator.gpu:
        if (Platform.isAndroid) {
          options.addDelegate(GpuDelegateV2());
        } else if (Platform.isIOS || Platform.isMacOS) {
          options.addDelegate(GpuDelegate());
        }
        break;
      case HardwareAccelerator.npu:
        if (Platform.isAndroid) {
          options.useNnApiForAndroid = true;
        }
        break;
    }

    // Load shared CRAFT detector
    try {
      if (accelerator != HardwareAccelerator.cpu) {
        _detector = await Interpreter.fromAsset(
          'assets/models/detector.tflite',
          options: options,
        );
        _detectorAccelerator = accelerator;
      } else {
        throw 'CPU requested';
      }
    } catch (e) {
      print('Failed to load detector on accelerator ${accelerator.name}: $e. Falling back to CPU.');
      _detector = await Interpreter.fromAsset(
        'assets/models/detector.tflite',
      );
      _detectorAccelerator = HardwareAccelerator.cpu;
    }

    // Load language-specific recognizer
    try {
      final recognizerPath = language == 'bn'
          ? 'assets/models/recognizer_bn.tflite'
          : 'assets/models/recognizer_en.tflite';
      
      if (accelerator != HardwareAccelerator.cpu) {
        _recognizer = await Interpreter.fromAsset(
          recognizerPath,
          options: options,
        );
        _recognizerAccelerator = accelerator;
      } else {
        throw 'CPU requested';
      }
    } catch (e) {
      print('Failed to load recognizer on accelerator ${accelerator.name}: $e. Falling back to CPU.');
      final recognizerPath = language == 'bn'
          ? 'assets/models/recognizer_bn.tflite'
          : 'assets/models/recognizer_en.tflite';
      _recognizer = await Interpreter.fromAsset(
        recognizerPath,
      );
      _recognizerAccelerator = HardwareAccelerator.cpu;
    }
  }

  void dispose() {
    _detector?.close();
    _recognizer?.close();
  }

  /// Runs the full OCR pipeline (CRAFT detection + CRNN recognition) on an image.
  Future<List<OcrResult>> runOcr(img.Image originalImage) async {
    if (!isLoaded) {
      throw StateError('EasyOCR models are not loaded. Call loadModels first.');
    }

    // --- STEP 1: Detection pre-processing (CRAFT detector input shape: [1, 608, 800, 3]) ---
    const int detH = 608;
    const int detW = 800;
    
    // Scale keeping aspect ratio
    final double scale = min(detW / originalImage.width, detH / originalImage.height);
    final int scaledW = (originalImage.width * scale).round();
    final int scaledH = (originalImage.height * scale).round();
    
    // Resize image
    final img.Image resized = img.copyResize(originalImage, width: scaledW, height: scaledH);
    
    // Create black padded canvas of size 800x608
    final img.Image paddedCanvas = img.Image(width: detW, height: detH);
    final int padLeft = (detW - scaledW) ~/ 2;
    final int padTop = (detH - scaledH) ~/ 2;
    
    img.compositeImage(paddedCanvas, resized, dstX: padLeft, dstY: padTop);

    // Prepare flat Float32List for detector input (1 * 608 * 800 * 3)
    final Float32List detInputBuffer = Float32List(1 * detH * detW * 3);
    int bufferIdx = 0;
    for (int y = 0; y < detH; y++) {
      for (int x = 0; x < detW; x++) {
        final pixel = paddedCanvas.getPixel(x, y);
        detInputBuffer[bufferIdx++] = pixel.r / 255.0;
        detInputBuffer[bufferIdx++] = pixel.g / 255.0;
        detInputBuffer[bufferIdx++] = pixel.b / 255.0;
      }
    }

    // Run detector model
    final Float32List detOutputBuffer = Float32List(1 * 304 * 400 * 2);
    final detStopwatch = Stopwatch()..start();
    _detector!.run(detInputBuffer.buffer, detOutputBuffer.buffer);
    detStopwatch.stop();
    _detectorInferenceTimeMs = detStopwatch.elapsedMicroseconds / 1000.0;

    // --- STEP 2: Post-processing CRAFT output using Connected Components ---
    const int outH = 304;
    const int outW = 400;

    // Thresholds
    const double lowText = 0.4;
    const double linkThreshold = 0.4;
    const double textThreshold = 0.7;

    // Build thresholded combined map
    final List<bool> combinedMap = List.filled(outH * outW, false);
    for (int i = 0; i < outH * outW; i++) {
      final double textVal = detOutputBuffer[i * 2];
      final double linkVal = detOutputBuffer[i * 2 + 1];
      
      final bool textScore = textVal >= lowText;
      final bool linkScore = linkVal >= linkThreshold;
      combinedMap[i] = textScore || linkScore;
    }

    // Connected Components labeling via Breadth-First Search (BFS)
    final Uint8List visited = Uint8List(outH * outW);
    final List<List<Point<int>>> components = [];

    final List<Point<int>> neighbors = [
      const Point(-1, -1), const Point(0, -1), const Point(1, -1),
      const Point(-1, 0),                       const Point(1, 0),
      const Point(-1, 1),  const Point(0, 1),  const Point(1, 1),
    ];

    for (int y = 0; y < outH; y++) {
      for (int x = 0; x < outW; x++) {
        final int idx = y * outW + x;
        if (combinedMap[idx] && visited[idx] == 0) {
          // New component found
          final List<Point<int>> component = [];
          final List<Point<int>> queue = [Point(x, y)];
          visited[idx] = 1;

          int head = 0;
          while (head < queue.length) {
            final curr = queue[head++];
            component.add(curr);

            for (final dir in neighbors) {
              final int nx = curr.x + dir.x;
              final int ny = curr.y + dir.y;
              if (nx >= 0 && nx < outW && ny >= 0 && ny < outH) {
                final int nidx = ny * outW + nx;
                if (combinedMap[nidx] && visited[nidx] == 0) {
                  visited[nidx] = 1;
                  queue.add(Point(nx, ny));
                }
              }
            }
          }
          components.add(component);
        }
      }
    }

    // Filter and extract bounding boxes
    final List<Rectangle<int>> boundingBoxes = [];
    for (final component in components) {
      if (component.length < 10) continue; // Area size filter

      // Check max text value in this component
      double maxTextVal = -1.0;
      int minX = outW, maxX = -1, minY = outH, maxY = -1;
      for (final p in component) {
        final double textVal = detOutputBuffer[(p.y * outW + p.x) * 2];
        if (textVal > maxTextVal) maxTextVal = textVal;

        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
      }

      if (maxTextVal < textThreshold) continue; // Max text threshold filter

      // Add a slight margin (10%)
      final int w = maxX - minX + 1;
      final int h = maxY - minY + 1;
      final int marginX = (w * 0.1).round();
      final int marginY = (h * 0.1).round();

      final int xmin = max(0, minX - marginX);
      final int xmax = min(outW - 1, maxX + marginX);
      final int ymin = max(0, minY - marginY);
      final int ymax = min(outH - 1, maxY + marginY);

      boundingBoxes.add(Rectangle(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));
    }

    // Map bounding boxes back to the original image dimensions
    final List<List<Point<int>>> mappedBoxes = [];
    for (final box in boundingBoxes) {
      // CRAFT output is 304x400, model input was 608x800. So we multiply output coordinates by 2
      final int xminInput = box.left * 2;
      final int xmaxInput = (box.left + box.width) * 2;
      final int yminInput = box.top * 2;
      final int ymaxInput = (box.top + box.height) * 2;

      // Inverse of scale and pad
      final int xminOrig = max(0, ((xminInput - padLeft) / scale).round());
      final int xmaxOrig = min(originalImage.width - 1, ((xmaxInput - padLeft) / scale).round());
      final int yminOrig = max(0, ((yminInput - padTop) / scale).round());
      final int ymaxOrig = min(originalImage.height - 1, ((ymaxInput - padTop) / scale).round());

      if (xmaxOrig > xminOrig && ymaxOrig > yminOrig) {
        mappedBoxes.add([
          Point(xminOrig, yminOrig),
          Point(xmaxOrig, yminOrig),
          Point(xmaxOrig, ymaxOrig),
          Point(xminOrig, ymaxOrig),
        ]);
      }
    }

    // Sort boxes top-to-bottom for reading order
    mappedBoxes.sort((a, b) => a[0].y.compareTo(b[0].y));

    // --- STEP 3: Text Recognition ---
    final List<OcrResult> ocrResults = [];
    final vocabulary = _currentLanguage == 'bn' ? _bnVocabulary : _enVocabulary;
    final int vocabSize = _currentLanguage == 'bn' ? 170 : 97;

    // Verify recognizer output dimension before running to prevent crash on mismatched models
    final outputTensorShape = _recognizer!.getOutputTensors().first.shape;
    final int actualModelVocabSize = outputTensorShape[2];
    if (actualModelVocabSize != vocabSize) {
      throw StateError(
        'Model vocabulary size mismatch! Selected language is $_currentLanguage '
        '(expected vocab size $vocabSize), but loaded model has vocab size $actualModelVocabSize. '
        'Please replace recognizer_bn.tflite with the correct Bengali TFLite model.',
      );
    }

    final recStopwatch = Stopwatch();
    double totalRecTimeMs = 0.0;
    int runCount = 0;

    for (final box in mappedBoxes) {
      final int xmin = box[0].x;
      final int ymin = box[0].y;
      final int w = box[1].x - xmin;
      final int h = box[2].y - ymin;

      if (w <= 0 || h <= 0) continue;

      // Crop text region
      final img.Image cropped = img.copyCrop(originalImage, x: xmin, y: ymin, width: w, height: h);

      // Preprocess for recognizer: input shape [1, 64, 800, 1] (grayscale, pad-right)
      const int recH = 64;
      const int recW = 800;

      // Maintain aspect ratio scaling to height 64
      final double recScale = recH / cropped.height;
      final int scaledRecW = min(recW, (cropped.width * recScale).round());
      
      final img.Image recResized = img.copyResize(cropped, width: scaledRecW, height: recH);

      // Convert to grayscale and place on left of 800x64 canvas.
      // Use top-left pixel as background color padding
      final firstPixel = recResized.getPixel(0, 0);
      final double padPixelVal = (0.299 * firstPixel.r + 0.587 * firstPixel.g + 0.114 * firstPixel.b) / 255.0;

      final Float32List recInputBuffer = Float32List(1 * recH * recW * 1);
      
      for (int y = 0; y < recH; y++) {
        for (int x = 0; x < recW; x++) {
          final int idx = y * recW + x;
          if (x < scaledRecW) {
            final pixel = recResized.getPixel(x, y);
            // Grayscale formula
            final double gray = (0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b) / 255.0;
            recInputBuffer[idx] = gray;
          } else {
            recInputBuffer[idx] = padPixelVal; // pad right
          }
        }
      }

      // Run recognizer
      final Float32List recOutputBuffer = Float32List(1 * 199 * vocabSize);
      recStopwatch.reset();
      recStopwatch.start();
      _recognizer!.run(recInputBuffer.buffer, recOutputBuffer.buffer);
      recStopwatch.stop();
      totalRecTimeMs += recStopwatch.elapsedMicroseconds / 1000.0;
      runCount++;

      // --- STEP 4: Greedy CTC Decoding ---
      final List<int> decodedIndices = [];
      final List<double> decodedScores = [];

      for (int t = 0; t < 199; t++) {
        int bestIdx = 0;
        double maxProb = -1.0;
        
        // Find class index with max probability at timestep t
        // (Softmax is not strictly needed for argmax, as raw logits maintain relative order)
        for (int c = 0; c < vocabSize; c++) {
          final double val = recOutputBuffer[t * vocabSize + c];
          if (val > maxProb) {
            maxProb = val;
            bestIdx = c;
          }
        }

        decodedIndices.add(bestIdx);
        decodedScores.add(maxProb);
      }

      // Collapse consecutive duplicates and remove blanks (index 0)
      final List<int> collapsedIndices = [];
      final List<double> collapsedScores = [];
      
      int prevIdx = -1;
      for (int i = 0; i < decodedIndices.length; i++) {
        final int idx = decodedIndices[i];
        if (idx != prevIdx) {
          if (idx != 0) {
            collapsedIndices.add(idx);
            collapsedScores.add(decodedScores[i]);
          }
          prevIdx = idx;
        }
      }

      // Decode indices to text
      final StringBuffer sb = StringBuffer();
      double totalScore = 0.0;
      for (int i = 0; i < collapsedIndices.length; i++) {
        final int idx = collapsedIndices[i];
        // Vocabulary mapping (index 1 maps to char 0 of string vocabulary)
        if (idx - 1 >= 0 && idx - 1 < vocabulary.length) {
          sb.write(vocabulary[idx - 1]);
          totalScore += collapsedScores[i];
        }
      }

      String textResult = sb.toString().trim();
      
      // Postprocessing cleanups standard to EasyOCR
      if (textResult.endsWith(']') || textResult.endsWith('|')) {
        textResult = textResult.substring(0, textResult.length - 1).trim();
      }

      if (textResult.isNotEmpty) {
        final double confidence = collapsedIndices.isEmpty ? 0.0 : totalScore / collapsedIndices.length;
        ocrResults.add(OcrResult(
          box: box,
          text: textResult,
          confidence: confidence,
        ));
      }
    }

    _totalRecognizerTimeMs = totalRecTimeMs;
    _recognizerInferenceTimeMs = runCount > 0 ? totalRecTimeMs / runCount : 0.0;

    return ocrResults;
  }
}
