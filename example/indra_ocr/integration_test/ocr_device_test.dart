import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:indra_ocr/ocr_service.dart';

import '../test_support/expected_text.dart';

void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  final service = OcrService();
  testWidgets('Real offline Indra inference and device tuning', (tester) async {
    final info = await service.info();
    final samples = <Map<String, Object?>>[];
    // ignore: avoid_print
    print('INDRA_DEVICE ${jsonEncode(info)}');
    await OcrService.channel.invokeMethod('configure', {
      'backend': 'cpu',
      'threads': 4,
    });
    for (final kind in [
      'english',
      'blank',
      'dark',
      'rotated',
      'bengali',
      'scan',
    ]) {
      final path = await service.sample(kind);
      final result = await service.recognize(path);
      expect(
        result.text,
        expectedText[kind],
        reason: "Optimized model changed $kind text",
      );
      samples.add({
        "sample": kind,
        "text": result.text,
        "timings": result.timings,
        "rotation": result.rotation,
      });
      // ignore: avoid_print
      print(
        'INDRA_RESULT ${jsonEncode({'sample': kind, 'text': result.text, 'timings': result.timings, 'rotation': result.rotation})}',
      );
      if (kind == 'blank') {
        expect(result.text, isEmpty);
      } else if (kind == 'bengali') {
        expect(result.text, contains('বাংলা'));
      } else if (kind == 'scan') {
        expect(result.text.toUpperCase(), contains('SEOUL'));
        expect(RegExp(r'[가-힣]').hasMatch(result.text), isTrue);
      } else {
        expect(result.text, contains('Indra'));
        expect(result.text, contains('125.50'));
      }
      expect(result.timings['total'], greaterThan(0));
    }
    final path = await service.sample('scan');
    final tuning = await service.benchmark(path);
    // ignore: avoid_print
    print('INDRA_BENCHMARK ${jsonEncode(tuning)}');
    expect(tuning['medianMs'] as num, greaterThan(0));
    binding.reportData = {
      'device': info,
      'samples': samples,
      'benchmark': tuning,
      'mode': 'profile',
      'date': '2026-10-25',
    };
    final repeated = await service.recognize(path, quality: true);
    expect(repeated.text.toUpperCase(), contains('SEOUL'));
    await expectLater(
      service.recognize('/missing/image.png'),
      throwsA(isA<Exception>()),
    );
    expect(
      (await service.recognize(await service.sample('english'))).text,
      contains('Indra'),
    );
  }, timeout: const Timeout(Duration(minutes: 12)));
}
