import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

enum HardwareAccelerator {
  cpu,
  gpu,
  npu,
}

class RetinaFaceAnchor {
  final double cx;
  final double cy;
  final double pw;
  final double ph;
  RetinaFaceAnchor(this.cx, this.cy, this.pw, this.ph);
}

class FaceDetection {
  final Rect box; // Bounding box relative to the original image dimensions
  final List<Point<double>> landmarks; // 5 landmarks (left eye, right eye, nose, left mouth, right mouth)
  final double score;

  FaceDetection({
    required this.box,
    required this.landmarks,
    required this.score,
  });
}

class EyeGazeResult {
  final Rect? faceBox;
  final Rect? eyeBox;
  final Offset eyeCenter;
  final double pitchRad;
  final double yawRad;
  final String side; // "left" or "right"
  final double score; // Face detection score, if any

  EyeGazeResult({
    this.faceBox,
    this.eyeBox,
    required this.eyeCenter,
    required this.pitchRad,
    required this.yawRad,
    required this.side,
    this.score = 1.0,
  });

  double get pitchDeg => pitchRad * 180.0 / pi;
  double get yawDeg => yawRad * 180.0 / pi;
}

class EyeGazeService {
  Interpreter? _detectorInterpreter;
  Interpreter? _gazeInterpreter;
  bool _isLocalFileLoaded = false;
  HardwareAccelerator _currentAccelerator = HardwareAccelerator.cpu;
  
  late final List<RetinaFaceAnchor> _anchors;

  bool get isLoaded => _detectorInterpreter != null && _gazeInterpreter != null;
  bool get isLocalFileLoaded => _isLocalFileLoaded;
  HardwareAccelerator get currentAccelerator => _currentAccelerator;

  // Local model paths for rapid macOS testing
  static const String _localDetectorPath =
      '/Users/mdshahidulislam/Documents/resource/flutter-tflite/allmodel/sixd_repnet-tflite-float/face_detector.tflite';
  static const String _localGazePath =
      '/Users/mdshahidulislam/Documents/resource/flutter-tflite/allmodel/eyegaze-tflite-float/eyegaze.tflite';

  EyeGazeService() {
    // Precompute RetinaFace anchors (16800 anchors for 640x640 input)
    _anchors = _generateAnchors();
  }

  /// Generates the anchor prior boxes for 640x640 input size
  static List<RetinaFaceAnchor> _generateAnchors() {
    final anchors = <RetinaFaceAnchor>[];
    final minSizes = [
      [16, 32],
      [64, 128],
      [256, 512]
    ];
    final steps = [8, 16, 32];
    for (int k = 0; k < steps.length; k++) {
      final int step = steps[k];
      final int fh = (640.0 / step).ceil();
      final int fw = (640.0 / step).ceil();
      final currentMinSizes = minSizes[k];
      for (int i = 0; i < fh; i++) {
        for (int j = 0; j < fw; j++) {
          for (final minSize in currentMinSizes) {
            final double sKx = minSize / 640.0;
            final double sKy = minSize / 640.0;
            final double cx = (j + 0.5) * step / 640.0;
            final double cy = (i + 0.5) * step / 640.0;
            anchors.add(RetinaFaceAnchor(cx, cy, sKx, sKy));
          }
        }
      }
    }
    return anchors;
  }

  /// Loads both models with specified hardware accelerator.
  Future<void> loadModels({HardwareAccelerator accelerator = HardwareAccelerator.cpu}) async {
    _detectorInterpreter?.close();
    _gazeInterpreter?.close();
    _detectorInterpreter = null;
    _gazeInterpreter = null;
    _currentAccelerator = accelerator;

    final options = InterpreterOptions();
    switch (accelerator) {
      case HardwareAccelerator.cpu:
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

    final localDetFile = File(_localDetectorPath);
    final localGazeFile = File(_localGazePath);

    if (localDetFile.existsSync() && localGazeFile.existsSync()) {
      try {
        _detectorInterpreter = Interpreter.fromFile(localDetFile, options: options);
        _gazeInterpreter = Interpreter.fromFile(localGazeFile, options: options);
        _isLocalFileLoaded = true;
        debugPrint('RetinaFace and EyeGaze models loaded successfully from local files.');
        return;
      } catch (e) {
        debugPrint('Failed to load from local file paths: $e. Falling back to assets.');
      }
    }

    // Fallback to assets
    _detectorInterpreter = await Interpreter.fromAsset(
      'assets/models/face_detector.tflite',
      options: options,
    );
    _gazeInterpreter = await Interpreter.fromAsset(
      'assets/models/eyegaze.tflite',
      options: options,
    );
    _isLocalFileLoaded = false;
    debugPrint('RetinaFace and EyeGaze models loaded successfully from assets.');
  }

  void dispose() {
    _detectorInterpreter?.close();
    _gazeInterpreter?.close();
    _detectorInterpreter = null;
    _gazeInterpreter = null;
  }

  /// Entry point for static image bytes
  Future<List<EyeGazeResult>> estimateGaze(
    Uint8List imageBytes, {
    double scoreThreshold = 0.85,
    double nmsThreshold = 0.4,
    double eyeCropScale = 0.35,
    bool rawEyeCropMode = false,
    String rawEyeSide = 'left',
    required Function(double preTime, double detTime, double postTime, double gazeTime) onStatsUpdated,
  }) async {
    final decodeStopwatch = Stopwatch()..start();
    final img.Image? originalImage = img.decodeImage(imageBytes);
    decodeStopwatch.stop();
    final double decodeTime = decodeStopwatch.elapsedMicroseconds / 1000.0;

    if (originalImage == null) {
      throw ArgumentError('Failed to decode input image.');
    }

    return estimateGazeFromImage(
      originalImage,
      scoreThreshold: scoreThreshold,
      nmsThreshold: nmsThreshold,
      eyeCropScale: eyeCropScale,
      rawEyeCropMode: rawEyeCropMode,
      rawEyeSide: rawEyeSide,
      onStatsUpdated: (pre, det, post, gaze) {
        onStatsUpdated(pre + decodeTime, det, post, gaze);
      },
    );
  }

  /// Processes image from decoded object (efficient for camera stream)
  Future<List<EyeGazeResult>> estimateGazeFromImage(
    img.Image originalImage, {
    double scoreThreshold = 0.85,
    double nmsThreshold = 0.4,
    double eyeCropScale = 0.35,
    bool rawEyeCropMode = false,
    String rawEyeSide = 'left',
    required Function(double preTime, double detTime, double postTime, double gazeTime) onStatsUpdated,
  }) async {
    if (!isLoaded) {
      throw StateError('Models are not loaded. Call loadModels first.');
    }

    final preStopwatch = Stopwatch()..start();
    final int origW = originalImage.width;
    final int origH = originalImage.height;

    // --- CASE 1: Raw Eye Crop Mode (Pre-cropped eye image input) ---
    if (rawEyeCropMode) {
      preStopwatch.stop();
      final double preTime = preStopwatch.elapsedMicroseconds / 1000.0;

      final gazeStopwatch = Stopwatch()..start();
      // Preprocess the eye crop
      final img.Image processedCrop = _preprocessEyeCrop(originalImage, rawEyeSide);
      
      // Run EyeGaze inference
      final double pitchRad, yawRad;
      try {
        final gazeAngles = _runGazeModel(processedCrop, rawEyeSide);
        pitchRad = gazeAngles[0];
        yawRad = gazeAngles[1];
      } catch (e) {
        gazeStopwatch.stop();
        rethrow;
      }
      gazeStopwatch.stop();
      final double gazeTime = gazeStopwatch.elapsedMicroseconds / 1000.0;

      onStatsUpdated(preTime, 0.0, 0.0, gazeTime);

      return [
        EyeGazeResult(
          eyeCenter: Offset(origW / 2, origH / 2),
          pitchRad: pitchRad,
          yawRad: yawRad,
          side: rawEyeSide,
          score: 1.0,
        ),
      ];
    }

    // --- CASE 2: Full Face Mode (Auto-detection & Eye-cropping) ---
    // 1. Resize & Preprocess original image for face detection
    final img.Image resizedDetectorImage = img.copyResize(
      originalImage,
      width: 640,
      height: 640,
      interpolation: img.Interpolation.linear,
    );

    final Float32List detectorInput = Float32List(1 * 640 * 640 * 3);
    int offset = 0;
    for (final pixel in resizedDetectorImage) {
      detectorInput[offset++] = pixel.rNormalized.toDouble();
      detectorInput[offset++] = pixel.gNormalized.toDouble();
      detectorInput[offset++] = pixel.bNormalized.toDouble();
    }
    preStopwatch.stop();
    final double preTime = preStopwatch.elapsedMicroseconds / 1000.0;

    // 2. Run Face Detection
    final detStopwatch = Stopwatch()..start();
    
    // Allocate outputs for RetinaFace
    final Float32List bboxRegressions = Float32List(1 * 16800 * 4);
    final Float32List classifications = Float32List(1 * 16800 * 2);
    final Float32List landmarkRegressions = Float32List(1 * 16800 * 10);

    // Dynamically query outputs indices to ensure correct mapping
    int bboxIdx = 0;
    int classIdx = 1;
    int landmIdx = 2;
    final outputTensors = _detectorInterpreter!.getOutputTensors();
    for (int i = 0; i < outputTensors.length; i++) {
      final tensor = _detectorInterpreter!.getOutputTensor(i);
      final shape = tensor.shape;
      if (shape.length == 3 && shape[1] == 16800) {
        if (shape[2] == 4) {
          bboxIdx = i;
        } else if (shape[2] == 2) {
          classIdx = i;
        } else if (shape[2] == 10) {
          landmIdx = i;
        }
      }
    }

    final inputs = [detectorInput.buffer];
    final outputs = {
      bboxIdx: bboxRegressions.buffer,
      classIdx: classifications.buffer,
      landmIdx: landmarkRegressions.buffer,
    };

    _detectorInterpreter!.runForMultipleInputs(inputs, outputs);
    detStopwatch.stop();
    final double detTime = detStopwatch.elapsedMicroseconds / 1000.0;

    // 3. Post-process Face Detections (Decoding & NMS)
    final postStopwatch = Stopwatch()..start();
    final List<FaceDetection> rawDetections = _decodeDetections(
      bboxRegressions,
      classifications,
      landmarkRegressions,
      origW,
      origH,
      scoreThreshold,
    );

    final List<FaceDetection> faces = _runNMS(rawDetections, nmsThreshold);
    postStopwatch.stop();
    final double postTime = postStopwatch.elapsedMicroseconds / 1000.0;

    // 4. Crop Eyes & Run EyeGaze on each face
    final gazeStopwatch = Stopwatch()..start();
    final List<EyeGazeResult> results = [];

    for (final face in faces) {
      final double faceW = face.box.width;

      // Extract left and right eye coordinates from RetinaFace landmarks
      // In RetinaFace 5 landmarks: index 0 = left eye, 1 = right eye (from image layout perspective)
      // Note: "left eye" landmark 0 is on the left side of the image (which corresponds to the person's right eye, but we treat it as side='left' or layout-left).
      // Let's call them "left" (index 0) and "right" (index 1) based on image layout.
      final Point<double> leftEyeLm = face.landmarks[0];
      final Point<double> rightEyeLm = face.landmarks[1];

      // Left Eye Processing
      if (leftEyeLm.x > 0 && leftEyeLm.y > 0) {
        final double eyeCropW = faceW * eyeCropScale;
        final double eyeCropH = eyeCropW * (96.0 / 160.0);
        
        final img.Image? leftCrop = _cropSafe(originalImage, leftEyeLm.x, leftEyeLm.y, eyeCropW, eyeCropH);
        if (leftCrop != null) {
          final img.Image processedLeft = _preprocessEyeCrop(leftCrop, 'left');
          final gazeAngles = _runGazeModel(processedLeft, 'left');
          
          results.add(EyeGazeResult(
            faceBox: face.box,
            eyeBox: Rect.fromCenter(
              center: Offset(leftEyeLm.x, leftEyeLm.y),
              width: eyeCropW,
              height: eyeCropH,
            ),
            eyeCenter: Offset(leftEyeLm.x, leftEyeLm.y),
            pitchRad: gazeAngles[0],
            yawRad: gazeAngles[1],
            side: 'left',
            score: face.score,
          ));
        }
      }

      // Right Eye Processing
      if (rightEyeLm.x > 0 && rightEyeLm.y > 0) {
        final double eyeCropW = faceW * eyeCropScale;
        final double eyeCropH = eyeCropW * (96.0 / 160.0);
        
        final img.Image? rightCrop = _cropSafe(originalImage, rightEyeLm.x, rightEyeLm.y, eyeCropW, eyeCropH);
        if (rightCrop != null) {
          final img.Image processedRight = _preprocessEyeCrop(rightCrop, 'right');
          final gazeAngles = _runGazeModel(processedRight, 'right');
          
          results.add(EyeGazeResult(
            faceBox: face.box,
            eyeBox: Rect.fromCenter(
              center: Offset(rightEyeLm.x, rightEyeLm.y),
              width: eyeCropW,
              height: eyeCropH,
            ),
            eyeCenter: Offset(rightEyeLm.x, rightEyeLm.y),
            pitchRad: gazeAngles[0],
            yawRad: gazeAngles[1], // Negated yaw is handled inside _runGazeModel
            side: 'right',
            score: face.score,
          ));
        }
      }
    }
    gazeStopwatch.stop();
    final double gazeTime = gazeStopwatch.elapsedMicroseconds / 1000.0;

    onStatsUpdated(preTime, detTime, postTime, gazeTime);

    return results;
  }

  /// Helper to safely crop image regions with boundary checking
  img.Image? _cropSafe(img.Image src, double cx, double cy, double w, double h) {
    int xMin = (cx - w / 2).round();
    int yMin = (cy - h / 2).round();
    int cropW = w.round();
    int cropH = h.round();

    xMin = xMin.clamp(0, src.width - 1);
    yMin = yMin.clamp(0, src.height - 1);

    if (xMin + cropW > src.width) {
      cropW = src.width - xMin;
    }
    if (yMin + cropH > src.height) {
      cropH = src.height - yMin;
    }

    if (cropW <= 0 || cropH <= 0) return null;

    return img.copyCrop(src, x: xMin, y: yMin, width: cropW, height: cropH);
  }

  /// Equalizes histogram of an 8-bit single-channel/grayscale image.
  img.Image _equalizeHistogram(img.Image grayImage) {
    final histogram = List<int>.filled(256, 0);
    final totalPixels = grayImage.width * grayImage.height;

    for (final pixel in grayImage) {
      final int val = pixel.r.toInt().clamp(0, 255);
      histogram[val]++;
    }

    final cdf = List<int>.filled(256, 0);
    cdf[0] = histogram[0];
    for (int i = 1; i < 256; i++) {
      cdf[i] = cdf[i - 1] + histogram[i];
    }

    int minCdf = 0;
    for (int i = 0; i < 256; i++) {
      if (cdf[i] > 0) {
        minCdf = cdf[i];
        break;
      }
    }

    if (totalPixels - minCdf <= 0) {
      return grayImage;
    }

    final mapping = List<int>.filled(256, 0);
    for (int i = 0; i < 256; i++) {
      mapping[i] = (((cdf[i] - minCdf) / (totalPixels - minCdf)) * 255).round().clamp(0, 255);
    }

    for (final pixel in grayImage) {
      final int val = pixel.r.toInt().clamp(0, 255);
      final int newVal = mapping[val];
      pixel.r = newVal;
      pixel.g = newVal;
      pixel.b = newVal;
    }

    return grayImage;
  }

  /// Preprocesses an eye crop (resize, grayscale, equalize, horizontal flip if right eye)
  img.Image _preprocessEyeCrop(img.Image crop, String side) {
    // 1. Resize to W=160, H=96
    img.Image resized = img.copyResize(
      crop,
      width: 160,
      height: 96,
      interpolation: img.Interpolation.linear,
    );

    // 2. Grayscale conversion
    resized = img.grayscale(resized);

    // 3. Histogram equalization
    resized = _equalizeHistogram(resized);

    // 4. Horizontal flip for right eye
    if (side == 'right') {
      resized = img.flip(resized, direction: img.FlipDirection.horizontal);
    }

    return resized;
  }

  /// Runs the EyeGaze model on the preprocessed eye image and returns [pitch, yaw] in radians
  List<double> _runGazeModel(img.Image preprocessed, String side) {
    // Input is shape [1, 96, 160] (batch/channel, height, width)
    final Float32List gazeInput = Float32List(1 * 96 * 160);
    int offset = 0;
    for (int y = 0; y < 96; y++) {
      for (int x = 0; x < 160; x++) {
        final pixel = preprocessed.getPixel(x, y);
        gazeInput[offset++] = pixel.rNormalized.toDouble();
      }
    }

    // Allocate outputs for EyeGaze
    final Float32List heatmaps = Float32List(1 * 3 * 34 * 48 * 80);
    final Float32List landmarks = Float32List(1 * 34 * 2);
    final Float32List gazePitchYaw = Float32List(1 * 2);

    // Query outputs indices
    int heatmapsIdx = 0;
    int landmarksIdx = 1;
    int gazeIdx = 2;
    final outputTensors = _gazeInterpreter!.getOutputTensors();
    for (int i = 0; i < outputTensors.length; i++) {
      final tensor = _gazeInterpreter!.getOutputTensor(i);
      final shape = tensor.shape;
      if (shape.length == 5 && shape[2] == 34) {
        heatmapsIdx = i;
      } else if (shape.length == 3 && shape[1] == 34) {
        landmarksIdx = i;
      } else if (shape.length == 2 && shape[1] == 2) {
        gazeIdx = i;
      }
    }

    final inputs = [gazeInput.buffer];
    final outputs = {
      heatmapsIdx: heatmaps.buffer,
      landmarksIdx: landmarks.buffer,
      gazeIdx: gazePitchYaw.buffer,
    };

    _gazeInterpreter!.runForMultipleInputs(inputs, outputs);

    double pitch = gazePitchYaw[0];
    double yaw = gazePitchYaw[1];

    // Post-process yaw: if right eye, negate it
    if (side == 'right') {
      yaw = -yaw;
    }

    return [pitch, yaw];
  }

  /// Decodes RetinaFace network outputs
  List<FaceDetection> _decodeDetections(
    Float32List bboxReg,
    Float32List classReg,
    Float32List landmReg,
    int imgW,
    int imgH,
    double scoreThreshold,
  ) {
    final List<FaceDetection> list = [];
    final double widthD = imgW.toDouble();
    final double heightD = imgH.toDouble();

    for (int idx = 0; idx < 16800; idx++) {
      // Index 1 contains face probability
      final double score = classReg[idx * 2 + 1];
      if (score < scoreThreshold) continue;

      final RetinaFaceAnchor anchor = _anchors[idx];

      final double dx = bboxReg[idx * 4 + 0];
      final double dy = bboxReg[idx * 4 + 1];
      final double dw = bboxReg[idx * 4 + 2];
      final double dh = bboxReg[idx * 4 + 3];

      // Decode box center & size (normalized scale)
      final double bx = anchor.cx + dx * 0.1 * anchor.pw;
      final double by = anchor.cy + dy * 0.1 * anchor.ph;
      final double bw = anchor.pw * exp(dw * 0.2);
      final double bh = anchor.ph * exp(dh * 0.2);

      // Convert to bounding box (x1, y1, x2, y2)
      final double x1 = (bx - bw / 2.0) * widthD;
      final double y1 = (by - bh / 2.0) * heightD;
      final double x2 = (bx + bw / 2.0) * widthD;
      final double y2 = (by + bh / 2.0) * heightD;

      final double left = max(0.0, x1);
      final double top = max(0.0, y1);
      final double right = min(widthD, x2);
      final double bottom = min(heightD, y2);

      if (right <= left || bottom <= top) continue;

      // Decode 5 landmarks
      final List<Point<double>> landmarks = [];
      for (int i = 0; i < 5; i++) {
        final double ldx = landmReg[idx * 10 + i * 2 + 0];
        final double ldy = landmReg[idx * 10 + i * 2 + 1];
        final double lx = (anchor.cx + ldx * 0.1 * anchor.pw) * widthD;
        final double ly = (anchor.cy + ldy * 0.1 * anchor.ph) * heightD;
        landmarks.add(Point(lx, ly));
      }

      list.add(FaceDetection(
        box: Rect.fromLTRB(left, top, right, bottom),
        landmarks: landmarks,
        score: score,
      ));
    }
    return list;
  }

  /// Performs Non-Maximum Suppression (NMS) to remove overlapping bounding boxes
  List<FaceDetection> _runNMS(List<FaceDetection> detections, double nmsThreshold) {
    if (detections.isEmpty) return [];

    // Sort detections by score in descending order
    detections.sort((a, b) => b.score.compareTo(a.score));

    final List<FaceDetection> selected = [];
    final List<bool> active = List<bool>.filled(detections.length, true);

    for (int i = 0; i < detections.length; i++) {
      if (!active[i]) continue;

      final face = detections[i];
      selected.add(face);

      for (int j = i + 1; j < detections.length; j++) {
        if (!active[j]) continue;

        final double iou = _calculateIoU(face.box, detections[j].box);
        if (iou > nmsThreshold) {
          active[j] = false;
        }
      }
    }
    return selected;
  }

  double _calculateIoU(Rect r1, Rect r2) {
    final double intersectionX1 = max(r1.left, r2.left);
    final double intersectionY1 = max(r1.top, r2.top);
    final double intersectionX2 = min(r1.right, r2.right);
    final double intersectionY2 = min(r1.bottom, r2.bottom);

    final double intersectionWidth = max(0.0, intersectionX2 - intersectionX1);
    final double intersectionHeight = max(0.0, intersectionY2 - intersectionY1);
    final double intersectionArea = intersectionWidth * intersectionHeight;

    final double unionArea = r1.width * r1.height + r2.width * r2.height - intersectionArea;

    if (unionArea <= 0.0) return 0.0;
    return intersectionArea / unionArea;
  }
}
