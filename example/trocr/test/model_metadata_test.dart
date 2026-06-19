import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:trocr/trocr_service.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('Print TrOCR Model Tensor Names', () async {
    // Load encoder
    final encoderFile = File('assets/models/encoder.tflite');
    final encoder = Interpreter.fromBuffer(encoderFile.readAsBytesSync());
    print('=== ENCODER INPUTS ===');
    for (var t in encoder.getInputTensors()) {
      print('Name: "${t.name}", Shape: ${t.shape}, Type: ${t.type}');
    }
    print('=== ENCODER OUTPUTS ===');
    for (var t in encoder.getOutputTensors()) {
      print('Name: "${t.name}", Shape: ${t.shape}, Type: ${t.type}');
    }
    encoder.close();

    // Load decoder
    final decoderFile = File('assets/models/decoder.tflite');
    final decoder = Interpreter.fromBuffer(decoderFile.readAsBytesSync());
    print('=== DECODER INPUTS ===');
    for (var t in decoder.getInputTensors()) {
      print('Name: "${t.name}", Shape: ${t.shape}, Type: ${t.type}');
    }
    print('=== DECODER OUTPUTS ===');
    for (var t in decoder.getOutputTensors()) {
      print('Name: "${t.name}", Shape: ${t.shape}, Type: ${t.type}');
    }
    decoder.close();
  });

  test('Run full TrOCR OCR pipeline on dummy image', () async {
    final service = TrocrService();
    await service.loadModelsForTesting(
      encoderFile: File('assets/models/encoder.tflite'),
      decoderFile: File('assets/models/decoder.tflite'),
      vocabFile: File('assets/models/vocab.json'),
    );

    // Create a 384x384 dummy image
    final dummyImage = img.Image(width: 384, height: 384);
    
    print('Running OCR...');
    final result = await service.runOcr(dummyImage);
    print('OCR Result: "$result"');
    
    service.dispose();
  });
}
