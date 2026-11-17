import 'package:flutter/services.dart';

class OcrLine {
  const OcrLine({
    required this.text,
    required this.box,
    required this.confidence,
  });
  factory OcrLine.fromMap(Map map) => OcrLine(
    text: map['text'] as String,
    box: (map['box'] as List).map((n) => (n as num).toDouble()).toList(),
    confidence: (map['confidence'] as num).toDouble(),
  );
  final String text;
  final List<double> box;
  final double confidence;
  double get top => box[1];
  double get bottom => box[3];
  double get height => bottom - top;
}

class OcrResult {
  const OcrResult({
    required this.text,
    required this.timings,
    required this.lines,
    required this.backend,
    required this.rotation,
    this.regions = const [],
  });
  factory OcrResult.fromMap(Map<Object?, Object?> map) => OcrResult(
    text: map['text']! as String,
    timings: (map['timings']! as Map).map(
      (k, v) => MapEntry(k as String, (v as num).toDouble()),
    ),
    lines: (map['lines']! as List).length,
    regions: (map['lines']! as List)
        .map((l) => OcrLine.fromMap(l as Map))
        .toList(),
    backend: map['backend']! as String,
    rotation: map['rotation']! as int,
  );
  final List<OcrLine> regions;
  final String text;
  final Map<String, double> timings;
  final int lines;
  final String backend;
  final int rotation;
}

class OcrService {
  static const channel = MethodChannel('dev.local.indra/ocr');
  Future<String?> exportMarkdown(String markdown) =>
      channel.invokeMethod<String>('exportMarkdown', {'markdown': markdown});
  Future<Map<Object?, Object?>> info() async =>
      (await channel.invokeMapMethod<Object?, Object?>('info'))!;
  Future<String> sample([String kind = 'english']) async =>
      (await channel.invokeMethod<String>('sample', {'kind': kind}))!;
  Future<OcrResult> recognize(String path, {bool quality = false}) async =>
      OcrResult.fromMap(
        (await channel.invokeMapMethod<Object?, Object?>('recognize', {
          'path': path,
          'quality': quality,
        }))!,
      );
  Future<Map<Object?, Object?>> benchmark(
    String path, {
    bool quality = false,
  }) async => (await channel.invokeMapMethod<Object?, Object?>('benchmark', {
    'path': path,
    'quality': quality,
  }))!;
}
