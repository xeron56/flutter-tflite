import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

enum HardwareAccelerator {
  cpu,
  gpu,
  npu,
}

class PreprocessResult {
  final Float32List modelInputImage;
  final Float32List modelInputMask;

  PreprocessResult({
    required this.modelInputImage,
    required this.modelInputMask,
  });
}

class PreprocessInput {
  final Uint8List originalImageBytes;
  final Uint8List maskBytes;

  PreprocessInput({
    required this.originalImageBytes,
    required this.maskBytes,
  });
}

class PostprocessInput {
  final Float32List modelOutput;
  final Uint8List originalImageBytes;
  final Uint8List maskBytes;
  final bool highFidelityComposite;

  PostprocessInput({
    required this.modelOutput,
    required this.originalImageBytes,
    required this.maskBytes,
    required this.highFidelityComposite,
  });
}

class LamaDilatedService {
  Interpreter? _interpreter;
  bool _isLocalFileLoaded = false;
  HardwareAccelerator _currentAccelerator = HardwareAccelerator.cpu;

  bool get isLoaded => _interpreter != null;
  bool get isLocalFileLoaded => _isLocalFileLoaded;
  HardwareAccelerator get currentAccelerator => _currentAccelerator;

  // Local model path for rapid macOS testing
  static const String _localModelPath =
      '/Users/mdshahidulislam/Documents/resource/flutter-tflite/allmodel/lama_dilated-tflite-float/lama_dilated.tflite';

  /// Loads the LaMa-Dilated model with optional hardware acceleration.
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
        debugPrint('LaMa model loaded successfully from local file path.');
        return;
      } catch (e) {
        debugPrint('Failed to load model from local file path: $e. Falling back to assets.');
      }
    }

    // Fallback to assets
    _interpreter = await Interpreter.fromAsset(
      'assets/models/lama_dilated.tflite',
      options: options,
    );
    _isLocalFileLoaded = false;
    debugPrint('LaMa model loaded successfully from assets.');
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }

  /// Runs the full inpainting pipeline: Preprocess -> Inference -> Postprocess.
  Future<Uint8List> inpaint(
    Uint8List originalImageBytes,
    Uint8List maskBytes, {
    required bool highFidelityComposite,
    required Function(double preTime, double infTime, double postTime) onStatsUpdated,
  }) async {
    if (!isLoaded) {
      throw StateError('LaMa-Dilated model is not loaded. Call loadModel first.');
    }

    // 1. PREPROCESSING (Background Isolate)
    final preStopwatch = Stopwatch()..start();
    final preprocessInput = PreprocessInput(
      originalImageBytes: originalImageBytes,
      maskBytes: maskBytes,
    );
    final preprocessResult = await compute(_preprocessTask, preprocessInput);
    preStopwatch.stop();
    final double preTime = preStopwatch.elapsedMicroseconds / 1000.0;

    // 2. INFERENCE (Main Isolate)
    final infStopwatch = Stopwatch()..start();
    // Inputs:
    // 0: [1, 512, 512, 3] float32 image
    // 1: [1, 512, 512, 1] float32 mask
    // Output:
    // 0: [1, 512, 512, 3] float32 painted_image
    final int outputLength = 1 * 512 * 512 * 3;
    final outputBuffer = Float32List(outputLength);

    final inputs = [
      preprocessResult.modelInputImage.buffer,
      preprocessResult.modelInputMask.buffer,
    ];
    final outputs = {
      0: outputBuffer.buffer,
    };

    _interpreter!.runForMultipleInputs(inputs, outputs);
    infStopwatch.stop();
    final double infTime = infStopwatch.elapsedMicroseconds / 1000.0;

    // 3. POSTPROCESSING (Background Isolate)
    final postStopwatch = Stopwatch()..start();
    final postprocessInput = PostprocessInput(
      modelOutput: outputBuffer,
      originalImageBytes: originalImageBytes,
      maskBytes: maskBytes,
      highFidelityComposite: highFidelityComposite,
    );
    final Uint8List inpaintedBytes = await compute(_postprocessTask, postprocessInput);
    postStopwatch.stop();
    final double postTime = postStopwatch.elapsedMicroseconds / 1000.0;

    onStatsUpdated(preTime, infTime, postTime);

    return inpaintedBytes;
  }
}

/// Preprocessing task running in background isolate.
PreprocessResult _preprocessTask(PreprocessInput input) {
  final img.Image? originalImage = img.decodeImage(input.originalImageBytes);
  final img.Image? maskImage = img.decodeImage(input.maskBytes);

  if (originalImage == null) {
    throw ArgumentError('Failed to decode original image.');
  }
  if (maskImage == null) {
    throw ArgumentError('Failed to decode mask image.');
  }

  // 1. Preprocess Image: Resize to 512x512 and scale pixels to [0.0, 1.0] (RGB)
  final img.Image resizedImage = img.copyResize(
    originalImage,
    width: 512,
    height: 512,
    interpolation: img.Interpolation.linear,
  );

  final Float32List modelInputImage = Float32List(1 * 512 * 512 * 3);
  int imgOffset = 0;
  for (final pixel in resizedImage) {
    modelInputImage[imgOffset++] = pixel.rNormalized.toDouble();
    modelInputImage[imgOffset++] = pixel.gNormalized.toDouble();
    modelInputImage[imgOffset++] = pixel.bNormalized.toDouble();
  }

  // 2. Preprocess Mask: Resize to 512x512 and threshold to [0.0, 1.0] single channel
  final img.Image resizedMask = img.copyResize(
    maskImage,
    width: 512,
    height: 512,
    interpolation: img.Interpolation.nearest,
  );

  final Float32List modelInputMask = Float32List(1 * 512 * 512 * 1);
  int maskOffset = 0;
  for (final pixel in resizedMask) {
    // Convert to grayscale: check if red value (or any) is > 127
    // Active drawing is white (r = 255), background is black (r = 0)
    // The mask value should be 1.0 for regions to be inpainted and 0.0 for keeping.
    final double val = pixel.r > 127 ? 1.0 : 0.0;
    modelInputMask[maskOffset++] = val;
  }

  return PreprocessResult(
    modelInputImage: modelInputImage,
    modelInputMask: modelInputMask,
  );
}

/// Postprocessing task running in background isolate.
Uint8List _postprocessTask(PostprocessInput input) {
  // 1. Create a 512x512 image from the model output
  final img.Image modelOutputImage = img.Image(width: 512, height: 512);
  int outOffset = 0;
  for (final pixel in modelOutputImage) {
    // Model output is Float32 in range [0.0, 1.0]
    final double r = input.modelOutput[outOffset++];
    final double g = input.modelOutput[outOffset++];
    final double b = input.modelOutput[outOffset++];

    pixel.r = (r.clamp(0.0, 1.0) * 255.0).round();
    pixel.g = (g.clamp(0.0, 1.0) * 255.0).round();
    pixel.b = (b.clamp(0.0, 1.0) * 255.0).round();
    pixel.a = 255;
  }

  if (!input.highFidelityComposite) {
    // Just return the raw 512x512 inpainted image
    return Uint8List.fromList(img.encodeJpg(modelOutputImage));
  }

  // 2. High-Fidelity Composite:
  // Decode original image and mask
  final img.Image? originalImage = img.decodeImage(input.originalImageBytes);
  final img.Image? maskImage = img.decodeImage(input.maskBytes);

  if (originalImage == null || maskImage == null) {
    return Uint8List.fromList(img.encodeJpg(modelOutputImage));
  }

  // Resize modelOutputImage back to original size
  final img.Image upscaledInpainted = img.copyResize(
    modelOutputImage,
    width: originalImage.width,
    height: originalImage.height,
    interpolation: img.Interpolation.linear,
  );

  // Resize maskImage to match original size if it doesn't already
  final img.Image finalMask = (maskImage.width == originalImage.width && maskImage.height == originalImage.height)
      ? maskImage
      : img.copyResize(
          maskImage,
          width: originalImage.width,
          height: originalImage.height,
          interpolation: img.Interpolation.nearest,
        );

  // Blend originalImage with upscaledInpainted using finalMask
  // For each pixel in the original image, if mask > 0, blend
  for (final pixel in originalImage) {
    final int x = pixel.x;
    final int y = pixel.y;

    final maskPixel = finalMask.getPixel(x, y);
    final double maskVal = maskPixel.r / 255.0; // grayscale mask value 0..1

    if (maskVal > 0.0) {
      final inpaintedPixel = upscaledInpainted.getPixel(x, y);
      
      // Interpolate between original pixel and inpainted pixel based on mask weight
      pixel.r = (maskVal * inpaintedPixel.r + (1.0 - maskVal) * pixel.r).round();
      pixel.g = (maskVal * inpaintedPixel.g + (1.0 - maskVal) * pixel.g).round();
      pixel.b = (maskVal * inpaintedPixel.b + (1.0 - maskVal) * pixel.b).round();
    }
  }

  return Uint8List.fromList(img.encodeJpg(originalImage));
}
