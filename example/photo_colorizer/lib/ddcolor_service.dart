import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

enum HardwareAccelerator {
  cpu,
  gpu,
  npu,
}

class PreprocessResult {
  final Float32List modelInput;
  final Float32List origL;
  final int width;
  final int height;

  PreprocessResult({
    required this.modelInput,
    required this.origL,
    required this.width,
    required this.height,
  });
}

class PostprocessInput {
  final Float32List outputAb;
  final Float32List origL;
  final int width;
  final int height;

  PostprocessInput({
    required this.outputAb,
    required this.origL,
    required this.width,
    required this.height,
  });
}

class DdColorService {
  Interpreter? _interpreter;
  bool _isLocalFileLoaded = false;
  HardwareAccelerator _currentAccelerator = HardwareAccelerator.cpu;

  bool get isLoaded => _interpreter != null;
  bool get isLocalFileLoaded => _isLocalFileLoaded;
  HardwareAccelerator get currentAccelerator => _currentAccelerator;

  // Local model path for rapid macOS testing
  static const String _localModelPath =
      '/Users/mdshahidulislam/Documents/resource/flutter-tflite/allmodel/ddcolor-tflite-float/ddcolor.tflite';

  /// Loads the DDColor model with optional hardware acceleration.
  Future<void> loadModel({HardwareAccelerator accelerator = HardwareAccelerator.cpu}) async {
    _interpreter?.close();
    _interpreter = null;
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

    // Check if local file exists first (development optimization for macOS)
    final localFile = File(_localModelPath);
    if (localFile.existsSync()) {
      try {
        _interpreter = Interpreter.fromFile(
          localFile,
          options: options,
        );
        _isLocalFileLoaded = true;
        debugPrint('DDColor model loaded successfully from local file path.');
        return;
      } catch (e) {
        debugPrint('Failed to load model from local file path: $e. Falling back to assets.');
      }
    }

    // Fallback to assets
    _interpreter = await Interpreter.fromAsset(
      'assets/models/ddcolor.tflite',
      options: options,
    );
    _isLocalFileLoaded = false;
    debugPrint('DDColor model loaded successfully from assets.');
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }

  /// Runs the full photo colorization pipeline: Preprocess -> Inference -> Postprocess.
  Future<Uint8List> colorize(
    Uint8List originalImageBytes, {
    required Function(double preTime, double infTime, double postTime) onStatsUpdated,
  }) async {
    if (!isLoaded) {
      throw StateError('DDColor model is not loaded. Call loadModel first.');
    }

    final totalStopwatch = Stopwatch()..start();

    // 1. PREPROCESSING (Run in Isolate)
    final preStopwatch = Stopwatch()..start();
    final preprocessResult = await compute(_preprocessTask, originalImageBytes);
    preStopwatch.stop();
    final double preTime = preStopwatch.elapsedMicroseconds / 1000.0;

    // 2. MODEL INFERENCE (Run on main isolate, because Interpreter uses C memory pointers)
    final infStopwatch = Stopwatch()..start();
    final int outputLength = 1 * 2 * 256 * 256;

    final outputBuffer = Float32List(outputLength);
    _interpreter!.run(preprocessResult.modelInput.buffer, outputBuffer.buffer);
    infStopwatch.stop();
    final double infTime = infStopwatch.elapsedMicroseconds / 1000.0;

    // 3. POSTPROCESSING (Run in Isolate)
    final postStopwatch = Stopwatch()..start();
    final postprocessInput = PostprocessInput(
      outputAb: outputBuffer,
      origL: preprocessResult.origL,
      width: preprocessResult.width,
      height: preprocessResult.height,
    );
    final Uint8List colorizedBytes = await compute(_postprocessTask, postprocessInput);
    postStopwatch.stop();
    final double postTime = postStopwatch.elapsedMicroseconds / 1000.0;

    totalStopwatch.stop();
    onStatsUpdated(preTime, infTime, postTime);

    return colorizedBytes;
  }
}

// ==========================================
// PURE DART COLOR CONVERSION & MATH UTILS
// ==========================================

double pivotRGBToXYZ(double val) {
  return (val > 0.04045)
      ? math.pow((val + 0.055) / 1.055, 2.4).toDouble()
      : (val / 12.92);
}

double pivotXYZToRGB(double val) {
  return (val > 0.0031308)
      ? (1.055 * math.pow(val, 1.0 / 2.4) - 0.055)
      : (12.92 * val);
}

/// Helper to precompute weights for fast bilinear interpolation.
class InterpolationWeights {
  final List<int> lowIndices;
  final List<int> highIndices;
  final List<double> weights;

  InterpolationWeights(int srcDim, int dstDim)
      : lowIndices = List.filled(dstDim, 0),
        highIndices = List.filled(dstDim, 0),
        weights = List.filled(dstDim, 0.0) {
    final double scale = srcDim.toDouble() / dstDim.toDouble();
    for (int i = 0; i < dstDim; i++) {
      final double srcPos = i * scale;
      final int low = srcPos.floor().clamp(0, srcDim - 1);
      final int high = (low + 1).clamp(0, srcDim - 1);
      lowIndices[i] = low;
      highIndices[i] = high;
      weights[i] = srcPos - low;
    }
  }
}

/// Bilinear interpolation to resize channel.
Float32List resizeChannelBilinear(Float32List src, int srcW, int srcH, int dstW, int dstH) {
  final weightsX = InterpolationWeights(srcW, dstW);
  final weightsY = InterpolationWeights(srcH, dstH);
  final dst = Float32List(dstW * dstH);

  for (int y = 0; y < dstH; y++) {
    final int y1 = weightsY.lowIndices[y];
    final int y2 = weightsY.highIndices[y];
    final double dy = weightsY.weights[y];
    final int y1Offset = y1 * srcW;
    final int y2Offset = y2 * srcW;
    final int dstRowOffset = y * dstW;

    for (int x = 0; x < dstW; x++) {
      final int x1 = weightsX.lowIndices[x];
      final int x2 = weightsX.highIndices[x];
      final double dx = weightsX.weights[x];

      final double val11 = src[y1Offset + x1];
      final double val21 = src[y1Offset + x2];
      final double val12 = src[y2Offset + x1];
      final double val22 = src[y2Offset + x2];

      final double valY1 = val11 * (1.0 - dx) + val21 * dx;
      final double valY2 = val12 * (1.0 - dx) + val22 * dx;

      dst[dstRowOffset + x] = valY1 * (1.0 - dy) + valY2 * dy;
    }
  }
  return dst;
}

/// Preprocessing task running in isolate.
PreprocessResult _preprocessTask(Uint8List bytes) {
  final img.Image? image = img.decodeImage(bytes);
  if (image == null) {
    throw ArgumentError('Failed to decode original image.');
  }

  final int width = image.width;
  final int height = image.height;

  // 1. Extract L channel at original resolution
  final origL = Float32List(width * height);
  int idx = 0;
  for (final pixel in image) {
    final double r = pixel.rNormalized.toDouble();
    final double g = pixel.gNormalized.toDouble();
    final double b = pixel.bNormalized.toDouble();

    // RGB -> XYZ
    final double pr = pivotRGBToXYZ(r);
    final double pg = pivotRGBToXYZ(g);
    final double pb = pivotRGBToXYZ(b);

    final double Y = pr * 0.212671 + pg * 0.715160 + pb * 0.072169;

    // XYZ -> Lab
    final double fy = (Y > 0.008856) ? math.pow(Y, 1.0 / 3.0).toDouble() : (7.787 * Y + 16.0 / 116.0);
    origL[idx++] = (116.0 * fy) - 16.0;
  }

  // 2. Resize to 256x256
  final img.Image resizedImage = img.copyResize(image, width: 256, height: 256);

  // 3. Extract resized L channel and build gray RGB input for the model
  final Float32List modelInput = Float32List(1 * 256 * 256 * 3);
  int offset = 0;

  for (final pixel in resizedImage) {
    final double r = pixel.rNormalized.toDouble();
    final double g = pixel.gNormalized.toDouble();
    final double b = pixel.bNormalized.toDouble();

    // RGB -> XYZ -> LAB (L channel only)
    final double pr = pivotRGBToXYZ(r);
    final double pg = pivotRGBToXYZ(g);
    final double pb = pivotRGBToXYZ(b);

    final double Y = pr * 0.212671 + pg * 0.715160 + pb * 0.072169;
    final double fy = (Y > 0.008856) ? math.pow(Y, 1.0 / 3.0).toDouble() : (7.787 * Y + 16.0 / 116.0);
    final double L = (116.0 * fy) - 16.0;

    // LAB [L, a=0, b=0] back to RGB:
    final double lFy = (L + 16.0) / 116.0;
    
    // a=0, b=0 => fx = fy = fz
    final double fx = lFy;
    final double fz = lFy;

    final double fx3 = fx * fx * fx;
    final double fy3 = lFy * lFy * lFy;
    final double fz3 = fz * fz * fz;

    final double xVal = (fx3 > 0.008856) ? fx3 : (fx - 16.0 / 116.0) / 7.787;
    final double yVal = (fy3 > 0.008856) ? fy3 : (lFy - 16.0 / 116.0) / 7.787;
    final double zVal = (fz3 > 0.008856) ? fz3 : (fz - 16.0 / 116.0) / 7.787;

    // XYZ values using D65 reference white
    final double X = xVal * 0.950456;
    final double yXyz = yVal * 1.0;
    final double Z = zVal * 1.088754;

    // XYZ -> RGB
    final double rVal = X * 3.240479 + yXyz * -1.537150 + Z * -0.498535;
    final double gVal = X * -0.969256 + yXyz * 1.875992 + Z * 0.041556;
    final double bVal = X * 0.055648 + yXyz * -0.204043 + Z * 1.057311;

    final double modelR = pivotXYZToRGB(rVal).clamp(0.0, 1.0);
    final double modelG = pivotXYZToRGB(gVal).clamp(0.0, 1.0);
    final double modelB = pivotXYZToRGB(bVal).clamp(0.0, 1.0);

    modelInput[offset++] = modelR;
    modelInput[offset++] = modelG;
    modelInput[offset++] = modelB;
  }

  return PreprocessResult(
    modelInput: modelInput,
    origL: origL,
    width: width,
    height: height,
  );
}

/// Postprocessing task running in isolate.
Uint8List _postprocessTask(PostprocessInput input) {
  final int width = input.width;
  final int height = input.height;

  // Extract a and b channels from model output of shape [1, 2, 256, 256]
  final int channelLength = 256 * 256;
  final Float32List aChannel = Float32List.sublistView(input.outputAb, 0, channelLength);
  final Float32List bChannel = Float32List.sublistView(input.outputAb, channelLength, 2 * channelLength);

  // Resize a & b channels to original width & height using bilinear interpolation
  final Float32List resizedA = resizeChannelBilinear(aChannel, 256, 256, width, height);
  final Float32List resizedB = resizeChannelBilinear(bChannel, 256, 256, width, height);

  // Reconstruct RGB image by merging origL with resizedA & resizedB
  final img.Image colorized = img.Image(width: width, height: height);

  int idx = 0;
  for (final pixel in colorized) {
    final double L = input.origL[idx];
    final double a = resizedA[idx];
    final double b = resizedB[idx];

    // LAB -> XYZ
    final double fy = (L + 16.0) / 116.0;
    final double fx = a / 500.0 + fy;
    final double fz = fy - b / 200.0;

    final double fx3 = fx * fx * fx;
    final double fy3 = fy * fy * fy;
    final double fz3 = fz * fz * fz;

    final double xVal = (fx3 > 0.008856) ? fx3 : (fx - 16.0 / 116.0) / 7.787;
    final double yVal = (fy3 > 0.008856) ? fy3 : (fy - 16.0 / 116.0) / 7.787;
    final double zVal = (fz3 > 0.008856) ? fz3 : (fz - 16.0 / 116.0) / 7.787;

    final double X = xVal * 0.950456;
    final double Y = yVal * 1.0;
    final double Z = zVal * 1.088754;

    // XYZ -> RGB
    final double rVal = X * 3.240479 + Y * -1.537150 + Z * -0.498535;
    final double gVal = X * -0.969256 + Y * 1.875992 + Z * 0.041556;
    final double bVal = X * 0.055648 + Y * -0.204043 + Z * 1.057311;

    final int red = (pivotXYZToRGB(rVal).clamp(0.0, 1.0) * 255.0).round();
    final int green = (pivotXYZToRGB(gVal).clamp(0.0, 1.0) * 255.0).round();
    final int blue = (pivotXYZToRGB(bVal).clamp(0.0, 1.0) * 255.0).round();

    pixel.r = red;
    pixel.g = green;
    pixel.b = blue;
    pixel.a = 255;

    idx++;
  }

  return Uint8List.fromList(img.encodeJpg(colorized));
}
