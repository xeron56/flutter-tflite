import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:indra_ocr/main.dart';
import 'package:indra_ocr/ocr_service.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  setUp(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
          OcrService.channel,
          (call) async => {
            'soc': 'SM8750',
            'android': '16',
            'engine': 'CPU · 4 threads',
          },
        );
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
          const MethodChannel('plugins.flutter.io/image_picker'),
          (call) async => null,
        );
  });
  testWidgets('Empty state and small screen at large text scale', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(360, 800);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    await tester.pumpWidget(const IndraApp());
    await tester.pumpAndSettle();
    expect(find.text('From scan to document.'), findsOneWidget);
    await tester.scrollUntilVisible(find.byKey(const Key('recognize')), 200);
    expect(
      tester.widget<FilledButton>(find.byKey(const Key('recognize'))).onPressed,
      isNull,
    );
    expect(tester.takeException(), isNull);
    tester.platformDispatcher.textScaleFactorTestValue = 2;
    addTearDown(tester.platformDispatcher.clearTextScaleFactorTestValue);
    await tester.pumpAndSettle();
    expect(tester.takeException(), isNull);
  });
}
