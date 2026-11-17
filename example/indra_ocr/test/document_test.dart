import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:indra_ocr/document_formatter.dart';
import 'package:indra_ocr/document_screen.dart';
import 'package:indra_ocr/main.dart';
import 'package:indra_ocr/ocr_service.dart';

OcrResult document(List<String> text) => OcrResult(
  text: text.join('\n'),
  timings: {'total': 150},
  lines: text.length,
  backend: 'CPU',
  rotation: 0,
  regions: [
    for (var i = 0; i < text.length; i++)
      OcrLine(
        text: text[i],
        box: [0, i * 30, 500, i * 30 + 20],
        confidence: 0.9,
      ),
  ],
);

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test('Wide line spacing stays in a paragraph until a larger gap', () {
    const input = OcrResult(
      text: '',
      timings: {},
      lines: 5,
      backend: 'CPU',
      rotation: 0,
      regions: [
        OcrLine(text: 'Report', box: [0, 0, 200, 40], confidence: 1),
        OcrLine(text: 'First line', box: [0, 100, 200, 120], confidence: 1),
        OcrLine(text: 'Second line', box: [0, 150, 200, 170], confidence: 1),
        OcrLine(text: 'Third line', box: [0, 200, 200, 220], confidence: 1),
        OcrLine(text: 'Next paragraph', box: [0, 320, 200, 340], confidence: 1),
      ],
    );
    expect(
      DocumentFormatter.format(input),
      '# Report\n\nFirst line  \nSecond line  \nThird line\n\nNext paragraph\n',
    );
  });
  test(
    'Every recognized line survives formatting, including multilingual text',
    () {
      final input = document([
        'Title',
        'বাংলাদেশ একটি সুন্দর দেশ।',
        '한국어 문서',
        'Total 125.50',
        'Final line',
      ]);
      final result = DocumentFormatter.format(input);
      for (final line in input.regions) {
        expect(result, contains(DocumentFormatter.escape(line.text)));
      }
      expect(result, endsWith('Final line\n'));
    },
  );
  test('Layout detects a heading and lists while keeping original OCR', () {
    final input = OcrResult(
      text: 'Report\n• First\n• Second',
      timings: {},
      lines: 3,
      backend: 'CPU',
      rotation: 0,
      regions: const [
        OcrLine(text: 'Report', box: [0, 0, 200, 40], confidence: 1),
        OcrLine(text: '• First', box: [0, 80, 200, 100], confidence: 1),
        OcrLine(text: '• Second', box: [0, 110, 200, 130], confidence: 1),
      ],
    );
    expect(DocumentFormatter.format(input), '# Report\n\n- First\n- Second\n');
    expect(input.text, contains('• First'));
    expect(DocumentFormatter.format(document([])), isEmpty);
  });
  testWidgets(
    'OCR punctuation renders literally rather than as Markdown or HTML',
    (tester) async {
      const text =
          'Text *stars* <b>tag</b> &copy; [x](url) ![image](path) | 3.5';
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: MarkdownBody(
              data: DocumentFormatter.format(document([text])),
              selectable: true,
            ),
          ),
        ),
      );
      expect(find.text(text, findRichText: true), findsOneWidget);
    },
  );

  testWidgets(
    'Edit, read, copy and export use complete Markdown; original stays intact',
    (tester) async {
      final result = document(['First line', 'Last line']);
      var edited = '';
      String? copied;
      String? exported;
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(SystemChannels.platform, (call) async {
            if (call.method == 'Clipboard.setData') {
              copied = (call.arguments as Map)['text'] as String;
            }
            return null;
          });
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(OcrService.channel, (call) async {
            expect(call.method, 'exportMarkdown');
            exported = (call.arguments as Map)['markdown'] as String;
            return 'indra-document.md';
          });
      await tester.pumpWidget(
        MaterialApp(
          theme: IndraApp.theme(Brightness.light),
          home: DocumentScreen(
            result: result,
            imagePath: '/missing.png',
            initialMarkdown: DocumentFormatter.format(result),
            onChanged: (s) => edited = s,
          ),
        ),
      );
      await tester.tap(find.byKey(const Key('view_edit')));
      await tester.pumpAndSettle();
      const update =
          '# My document\n\n**Bold** and বাংলা\n\n- First\n- Last\n\n| A | B |\n|---|---|\n| 1 | 2 |';
      await tester.enterText(find.byKey(const Key('markdown_editor')), update);
      expect(edited, update);
      await tester.tap(find.byKey(const Key('view_read')));
      await tester.pumpAndSettle();
      expect(find.text('My document', findRichText: true), findsOneWidget);
      await tester.tap(find.byKey(const Key('copy_markdown')));
      await tester.pump();
      expect(copied, update);
      await tester.tap(find.byKey(const Key('save_markdown')));
      await tester.pumpAndSettle();
      expect(exported, update);
      await tester.tap(find.byTooltip('Document options'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Original OCR text'));
      await tester.pumpAndSettle();
      expect(find.text(result.text), findsOneWidget);
      expect(tester.takeException(), isNull);
    },
  );

  testWidgets('Narrow screen, dark theme and large text have no overflow', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(360, 800);
    tester.view.devicePixelRatio = 1;
    tester.platformDispatcher.textScaleFactorTestValue = 2;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    addTearDown(tester.platformDispatcher.clearTextScaleFactorTestValue);
    await tester.pumpWidget(
      MaterialApp(
        theme: IndraApp.theme(Brightness.dark),
        home: DocumentScreen(
          result: document(['Hello']),
          imagePath: '/missing.png',
          initialMarkdown: '# A readable document\n\nLast line',
          onChanged: (_) {},
        ),
      ),
    );
    await tester.pumpAndSettle();
    expect(tester.takeException(), isNull);
    await tester.tap(find.byKey(const Key('view_edit')));
    await tester.pumpAndSettle();
    expect(tester.takeException(), isNull);
  });

  testWidgets('Save cancellation is silent and errors allow retry', (
    tester,
  ) async {
    var calls = 0;
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(OcrService.channel, (call) async {
          if (calls++ == 0) return null;
          throw PlatformException(
            code: 'export_error',
            message: 'Storage unavailable',
          );
        });
    await tester.pumpWidget(
      MaterialApp(
        home: DocumentScreen(
          result: document(['Hello']),
          imagePath: '/missing.png',
          initialMarkdown: 'Hello',
          onChanged: (_) {},
        ),
      ),
    );
    await tester.tap(find.byKey(const Key('save_markdown')));
    await tester.pumpAndSettle();
    expect(find.byType(SnackBar), findsNothing);
    await tester.tap(find.byKey(const Key('save_markdown')));
    await tester.pumpAndSettle();
    expect(find.text('Storage unavailable'), findsOneWidget);
    expect(
      tester
          .widget<IconButton>(find.byKey(const Key('save_markdown')))
          .onPressed,
      isNotNull,
    );
  });
}
