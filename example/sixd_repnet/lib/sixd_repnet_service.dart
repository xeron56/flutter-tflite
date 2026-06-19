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

class FacePose {
  final Rect box;
  final List<Point<double>> landmarks;
  final double score;
  final double pitch; // degrees
  final double yaw;   // degrees
  final double roll;  // degrees

  FacePose({
    required this.box,
    required this.landmarks,
    required this.score,
    required this.pitch,
    required this.yaw,
    required this.roll,
  });
}

class SixDRepNetService {
  Interpreter? _detectorInterpreter;
  Interpreter? _poseInterpreter;
  bool _isLocalFileLoaded = false;
  HardwareAccelerator _currentAccelerator = HardwareAccelerator.cpu;
  
  late final List<RetinaFaceAnchor> _anchors;

  bool get isLoaded => _detectorInterpreter != null && _poseInterpreter != null;
  bool get isLocalFileLoaded => _isLocalFileLoaded;
  HardwareAccelerator get currentAccelerator => _currentAccelerator;

  // Local model paths for rapid macOS testing
  static const String _localDetectorPath =
      '/Users/mdshahidulislam/Documents/resource/flutter-tflite/allmodel/sixd_repnet-tflite-float/face_detector.tflite';
  static const String _localPosePath =
      '/Users/mdshahidulislam/Documents/resource/flutter-tflite/allmodel/sixd_repnet-tflite-float/pose_estimator.tflite';

  SixDRepNetService() {
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
    _poseInterpreter?.close();
    _detectorInterpreter = null;
    _poseInterpreter = null;
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
    final localPoseFile = File(_localPosePath);

    if (localDetFile.existsSync() && localPoseFile.existsSync()) {
      try {
        _detectorInterpreter = Interpreter.fromFile(localDetFile, options: options);
        _poseInterpreter = Interpreter.fromFile(localPoseFile, options: options);
        _isLocalFileLoaded = true;
        debugPrint('SixDRepNet & RetinaFace models loaded successfully from local files.');
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
    _poseInterpreter = await Interpreter.fromAsset(
      'assets/models/pose_estimator.tflite',
      options: options,
    );
    _isLocalFileLoaded = false;
    debugPrint('SixDRepNet & RetinaFace models loaded successfully from assets.');
  }

  void dispose() {
    _detectorInterpreter?.close();
    _poseInterpreter?.close();
    _detectorInterpreter = null;
    _poseInterpreter = null;
  }

  /// Runs the full Head Pose Estimation pipeline:
  /// 1. Detect faces using RetinaFace
  /// 2. For each face, crop and run SixDRepNet pose estimation
  /// Returns a list of FacePose containing bounding boxes, landmarks, and pitch/yaw/roll angles.
  Future<List<FacePose>> estimatePose(
    Uint8List imageBytes, {
    double scoreThreshold = 0.85,
    double nmsThreshold = 0.4,
    required Function(double preTime, double detTime, double postTime, double poseTime) onStatsUpdated,
  }) async {
    if (!isLoaded) {
      throw StateError('Models are not loaded. Call loadModels first.');
    }

    final totalStopwatch = Stopwatch()..start();

    // 1. Decode original image
    final preStopwatch = Stopwatch()..start();
    final img.Image? originalImage = img.decodeImage(imageBytes);
    if (originalImage == null) {
      throw ArgumentError('Failed to decode input image.');
    }
    final int origW = originalImage.width;
    final int origH = originalImage.height;

    // Resize to detector input size (640x640)
    final img.Image resizedDetectorImage = img.copyResize(
      originalImage,
      width: 640,
      height: 640,
      interpolation: img.Interpolation.linear,
    );

    // Preprocess: Map values to [0.0, 1.0] and flatten to Float32List
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
    
    // Allocate outputs
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

    // 4. Crop & Estimate Pose for each face
    final poseStopwatch = Stopwatch()..start();
    final List<FacePose> results = [];

    // Fallback: If no faces detected, treat entire image as a face
    final List<FaceDetection> facesToProcess = faces.isNotEmpty
        ? faces
        : [
            FaceDetection(
              box: Rect.fromLTRB(0, 0, origW.toDouble(), origH.toDouble()),
              landmarks: List.generate(5, (_) => const Point(0.0, 0.0)),
              score: 1.0,
            )
          ];

    for (final face in facesToProcess) {
      final double bw = face.box.width;
      final double bh = face.box.height;

      // 20% padding matching upstream 6DRepNet
      final int xMinP = max(0, (face.box.left - 0.2 * bw).toInt());
      final int yMinP = max(0, (face.box.top - 0.2 * bh).toInt());
      final int xMaxP = min(origW, (face.box.right + 0.2 * bw).toInt());
      final int yMaxP = min(origH, (face.box.bottom + 0.2 * bh).toInt());

      final int cropW = xMaxP - xMinP;
      final int cropH = yMaxP - yMinP;

      if (cropW <= 0 || cropH <= 0) continue;

      // Crop face
      final img.Image crop = img.copyCrop(
        originalImage,
        x: xMinP,
        y: yMinP,
        width: cropW,
        height: cropH,
      );

      // Resize crop to 224x224
      final img.Image resizedCrop = img.copyResize(
        crop,
        width: 224,
        height: 224,
        interpolation: img.Interpolation.linear,
      );

      // Preprocess crop: values normalized to [0, 1]
      final Float32List poseInput = Float32List(1 * 224 * 224 * 3);
      int poseOffset = 0;
      for (final pixel in resizedCrop) {
        poseInput[poseOffset++] = pixel.rNormalized.toDouble();
        poseInput[poseOffset++] = pixel.gNormalized.toDouble();
        poseInput[poseOffset++] = pixel.bNormalized.toDouble();
      }

      // Run Pose Estimator
      final Float32List rotationMatrix = Float32List(1 * 3 * 3);
      _poseInterpreter!.run(poseInput.buffer, rotationMatrix.buffer);

      // Extract rotation matrix values (row-major)
      final double r00 = rotationMatrix[0];
      final double r10 = rotationMatrix[3];
      final double r11 = rotationMatrix[4];
      final double r12 = rotationMatrix[5];
      final double r20 = rotationMatrix[6];
      final double r21 = rotationMatrix[7];
      final double r22 = rotationMatrix[8];

      // Convert rotation matrix to Euler angles (in radians)
      double sy = sqrt(r00 * r00 + r10 * r10);
      if (sy < 1e-12) sy = 1e-12; // clamp min to 1e-12

      final bool singular = sy < 1e-6;
      double pitch, yaw, roll;

      if (!singular) {
        pitch = atan2(r21, r22);
        yaw = atan2(-r20, sy);
        roll = atan2(r10, r00);
      } else {
        pitch = atan2(-r12, r11);
        yaw = atan2(-r20, sy);
        roll = 0.0;
      }

      // Convert to degrees
      final double pitchDeg = pitch * 180.0 / pi;
      final double yawDeg = yaw * 180.0 / pi;
      final double rollDeg = roll * 180.0 / pi;

      results.add(FacePose(
        box: face.box,
        landmarks: face.landmarks,
        score: face.score,
        pitch: pitchDeg,
        yaw: yawDeg,
        roll: rollDeg,
      ));
    }
    poseStopwatch.stop();
    final double poseTime = poseStopwatch.elapsedMicroseconds / 1000.0;

    totalStopwatch.stop();
    onStatsUpdated(preTime, detTime, postTime, poseTime);

    return results;
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
