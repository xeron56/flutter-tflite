import 'dart:convert';
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

class TrocrService {
  Interpreter? _encoder;
  Interpreter? _decoder;
  final List<String> _vocab = [];

  HardwareAccelerator _encoderAccelerator = HardwareAccelerator.cpu;
  HardwareAccelerator _decoderAccelerator = HardwareAccelerator.cpu;
  HardwareAccelerator _currentAccelerator = HardwareAccelerator.cpu;

  double _encoderTimeMs = 0.0;
  double _decoderTimeMs = 0.0;
  int _decodedTokenCount = 0;

  bool get isLoaded => _encoder != null && _decoder != null;
  HardwareAccelerator get currentAccelerator => _currentAccelerator;
  HardwareAccelerator get encoderAccelerator => _encoderAccelerator;
  HardwareAccelerator get decoderAccelerator => _decoderAccelerator;
  double get encoderTimeMs => _encoderTimeMs;
  double get decoderTimeMs => _decoderTimeMs;
  int get decodedTokenCount => _decodedTokenCount;

  Future<void> _loadVocab() async {
    if (_vocab.isNotEmpty) return;
    final String jsonStr = await rootBundle.loadString('assets/models/vocab.json');
    final List<dynamic> rawList = json.decode(jsonStr) as List<dynamic>;
    _vocab.clear();
    for (final item in rawList) {
      _vocab.add(item as String);
    }
  }

  Future<void> loadModels({HardwareAccelerator accelerator = HardwareAccelerator.cpu}) async {
    _currentAccelerator = accelerator;
    _encoder?.close();
    _decoder?.close();
    _encoder = null;
    _decoder = null;

    await _loadVocab();

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

    // Load Encoder
    try {
      if (accelerator != HardwareAccelerator.cpu) {
        _encoder = await Interpreter.fromAsset(
          'assets/models/encoder.tflite',
          options: options,
        );
        _encoderAccelerator = accelerator;
      } else {
        throw 'CPU requested';
      }
    } catch (e) {
      print('Failed to load encoder on accelerator ${accelerator.name}: $e. Falling back to CPU.');
      _encoder = await Interpreter.fromAsset(
        'assets/models/encoder.tflite',
      );
      _encoderAccelerator = HardwareAccelerator.cpu;
    }

    // Load Decoder
    try {
      if (accelerator != HardwareAccelerator.cpu) {
        _decoder = await Interpreter.fromAsset(
          'assets/models/decoder.tflite',
          options: options,
        );
        _decoderAccelerator = accelerator;
      } else {
        throw 'CPU requested';
      }
    } catch (e) {
      print('Failed to load decoder on accelerator ${accelerator.name}: $e. Falling back to CPU.');
      _decoder = await Interpreter.fromAsset(
        'assets/models/decoder.tflite',
      );
      _decoderAccelerator = HardwareAccelerator.cpu;
    }
  }

  Future<void> loadModelsForTesting({
    required File encoderFile,
    required File decoderFile,
    required File vocabFile,
  }) async {
    final String jsonStr = await vocabFile.readAsString();
    final List<dynamic> rawList = json.decode(jsonStr) as List<dynamic>;
    _vocab.clear();
    for (final item in rawList) {
      _vocab.add(item as String);
    }
    _encoder = Interpreter.fromBuffer(encoderFile.readAsBytesSync());
    _decoder = Interpreter.fromBuffer(decoderFile.readAsBytesSync());
    _encoderAccelerator = HardwareAccelerator.cpu;
    _decoderAccelerator = HardwareAccelerator.cpu;
    _currentAccelerator = HardwareAccelerator.cpu;
  }

  void dispose() {
    _encoder?.close();
    _decoder?.close();
  }

  Future<String> runOcr(img.Image originalImage) async {
    try {
      return await _runOcrCore(originalImage);
    } catch (e) {
      if (_currentAccelerator != HardwareAccelerator.cpu) {
        print('Inference failed on ${_currentAccelerator.name.toUpperCase()}: $e. Automatically falling back to CPU.');
        await loadModels(accelerator: HardwareAccelerator.cpu);
        return await _runOcrCore(originalImage);
      }
      rethrow;
    }
  }

  Future<String> _runOcrCore(img.Image originalImage) async {
    if (!isLoaded) {
      throw StateError('TrOCR models are not loaded. Call loadModels first.');
    }

    // --- STEP 1: Image Preprocessing (384x384, RGB, normalized [0.0, 1.0]) ---
    const int encSize = 384;
    final img.Image resized = img.copyResize(originalImage, width: encSize, height: encSize);
    
    final Float32List encInputBuffer = Float32List(1 * encSize * encSize * 3);
    int bufferIdx = 0;
    for (int y = 0; y < encSize; y++) {
      for (int x = 0; x < encSize; x++) {
        final pixel = resized.getPixel(x, y);
        // Place in channel-last format (H, W, C)
        encInputBuffer[bufferIdx++] = pixel.r / 255.0;
        encInputBuffer[bufferIdx++] = pixel.g / 255.0;
        encInputBuffer[bufferIdx++] = pixel.b / 255.0;
      }
    }

    // --- STEP 2: Pre-allocate cross attention cache buffers for Encoder output ---
    // Shape: [1, 8, 578, 32]. Size = 1 * 8 * 578 * 32 = 147968 floats.
    final List<Float32List> crossAttnKeys = List.generate(6, (_) => Float32List(147968));
    final List<Float32List> crossAttnVals = List.generate(6, (_) => Float32List(147968));

    // Map encoder outputs by name to match correctly
    final encoderOutputTensors = _encoder!.getOutputTensors();
    final Map<int, Object> encoderOutputsMap = {};
    for (int i = 0; i < encoderOutputTensors.length; i++) {
      final name = encoderOutputTensors[i].name;
      final RegExp numReg = RegExp(r'\d+');
      final match = numReg.allMatches(name).last.group(0);
      final int layer = int.parse(match!);
      if (name.contains('key') || name.contains('k_proj')) {
        encoderOutputsMap[i] = crossAttnKeys[layer].buffer;
      } else {
        encoderOutputsMap[i] = crossAttnVals[layer].buffer;
      }
    }

    // Run encoder
    final encStopwatch = Stopwatch()..start();
    _encoder!.runForMultipleInputs([encInputBuffer.buffer], encoderOutputsMap);
    encStopwatch.stop();
    _encoderTimeMs = encStopwatch.elapsedMicroseconds / 1000.0;

    // --- STEP 3: Setup Autoregressive Decoder Caches ---
    // Caches are shape [1, 8, 19, 32]. Size = 1 * 8 * 19 * 32 = 4864 floats.
    final List<Float32List> attnKeys = List.generate(6, (_) => Float32List(4864));
    final List<Float32List> attnVals = List.generate(6, (_) => Float32List(4864));

    // Output Caches are shape [1, 8, 20, 32]. Size = 1 * 8 * 20 * 32 = 5120 floats.
    final List<Float32List> attnKeysOut = List.generate(6, (_) => Float32List(5120));
    final List<Float32List> attnValsOut = List.generate(6, (_) => Float32List(5120));

    final Int32List decoderInputIds = Int32List(1);
    final Int32List decoderIndex = Int32List(1);
    final Int32List nextTokenOut = Int32List(1);

    // Initial state
    decoderInputIds[0] = 2; // start_token_id
    decoderIndex[0] = 0;

    // Map decoder inputs by name
    final decoderInputTensors = _decoder!.getInputTensors();
    final List<Object> decoderInputsList = List<Object>.filled(decoderInputTensors.length, 0);
    for (int i = 0; i < decoderInputTensors.length; i++) {
      final name = decoderInputTensors[i].name;
      if (name == 'input_ids') {
        decoderInputsList[i] = decoderInputIds.buffer;
      } else if (name == 'index') {
        decoderInputsList[i] = decoderIndex.buffer;
      } else if (name.startsWith('kv_') && name.contains('cross') && (name.endsWith('key') || name.contains('key'))) {
        final RegExp numReg = RegExp(r'\d+');
        final match = numReg.allMatches(name).first.group(0);
        final int layer = int.parse(match!);
        decoderInputsList[i] = crossAttnKeys[layer].buffer;
      } else if (name.startsWith('kv_') && name.contains('cross') && (name.endsWith('val') || name.contains('val'))) {
        final RegExp numReg = RegExp(r'\d+');
        final match = numReg.allMatches(name).first.group(0);
        final int layer = int.parse(match!);
        decoderInputsList[i] = crossAttnVals[layer].buffer;
      } else if (name.startsWith('kv_') && !name.contains('cross') && (name.endsWith('key') || name.contains('key'))) {
        final RegExp numReg = RegExp(r'\d+');
        final match = numReg.allMatches(name).first.group(0);
        final int layer = int.parse(match!);
        decoderInputsList[i] = attnKeys[layer].buffer;
      } else if (name.startsWith('kv_') && !name.contains('cross') && (name.endsWith('val') || name.contains('val'))) {
        final RegExp numReg = RegExp(r'\d+');
        final match = numReg.allMatches(name).first.group(0);
        final int layer = int.parse(match!);
        decoderInputsList[i] = attnVals[layer].buffer;
      } else {
        print('Warning: unexpected decoder input tensor: $name');
      }
    }

    // Safety validation check to prevent silent downstream errors
    for (int i = 0; i < decoderInputTensors.length; i++) {
      if (decoderInputsList[i] == 0) {
        throw StateError(
          'Failed to map decoder input tensor at index $i: "${decoderInputTensors[i].name}".'
        );
      }
    }

    // Map decoder outputs by name
    final decoderOutputTensors = _decoder!.getOutputTensors();
    final Map<int, Object> decoderOutputsMap = {};
    for (int i = 0; i < decoderOutputTensors.length; i++) {
      final name = decoderOutputTensors[i].name;
      if (name == 'next_token') {
        decoderOutputsMap[i] = nextTokenOut.buffer;
      } else if (name.startsWith('kv_cache_key_')) {
        final RegExp numReg = RegExp(r'\d+');
        final match = numReg.allMatches(name).first.group(0);
        final int layer = int.parse(match!);
        decoderOutputsMap[i] = attnKeysOut[layer].buffer;
      } else if (name.startsWith('kv_cache_val_')) {
        final RegExp numReg = RegExp(r'\d+');
        final match = numReg.allMatches(name).first.group(0);
        final int layer = int.parse(match!);
        decoderOutputsMap[i] = attnValsOut[layer].buffer;
      } else {
        print('Warning: unexpected decoder output tensor: $name');
      }
    }

    final List<int> predictedTokens = [];
    final decStopwatch = Stopwatch();
    double totalDecTimeMs = 0.0;
    int decSteps = 0;

    // Loop for autoregressive generation
    const int maxLen = 20;
    for (int t = 0; t < maxLen; t++) {
      decoderIndex[0] = t;

      decStopwatch.reset();
      decStopwatch.start();
      _decoder!.runForMultipleInputs(decoderInputsList, decoderOutputsMap);
      decStopwatch.stop();
      totalDecTimeMs += decStopwatch.elapsedMicroseconds / 1000.0;
      decSteps++;

      final int nextToken = nextTokenOut[0];
      if (nextToken == 2 && t > 0) {
        break; // EOS reached
      }
      predictedTokens.add(nextToken);

      decoderInputIds[0] = nextToken;

      // Slice the attention cache outputs (1, 8, 20, 32) -> (1, 8, 19, 32)
      for (int layer = 0; layer < 6; layer++) {
        _sliceAndCopyCache(attnKeysOut[layer], attnKeys[layer]);
        _sliceAndCopyCache(attnValsOut[layer], attnVals[layer]);
      }
    }

    _decoderTimeMs = decSteps > 0 ? totalDecTimeMs / decSteps : 0.0;
    _decodedTokenCount = decSteps;

    return _decodeTokens(predictedTokens);
  }

  void _sliceAndCopyCache(Float32List src, Float32List dst) {
    for (int h = 0; h < 8; h++) {
      final int srcOffset = h * 20 * 32 + 32; // Skip first sequence token
      final int dstOffset = h * 19 * 32;
      final int copyLength = 19 * 32;
      dst.setRange(dstOffset, dstOffset + copyLength, src.sublist(srcOffset, srcOffset + copyLength));
    }
  }

  String _decodeTokens(List<int> tokens) {
    final StringBuffer buffer = StringBuffer();
    for (final token in tokens) {
      if (token >= 0 && token < _vocab.length) {
        final tokenString = _vocab[token];
        if (tokenString == '<s>' || tokenString == '</s>' || tokenString == '<pad>' || tokenString == '<unk>') {
          continue;
        }
        buffer.write(tokenString);
      }
    }
    return buffer.toString().replaceAll('\u2581', ' ').trim();
  }
}
