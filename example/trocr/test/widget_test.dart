import 'package:flutter_test/flutter_test.dart';
import 'package:trocr/main.dart';

void main() {
  testWidgets('TrOCR Lite smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const MyApp());

    // Verify that the title and empty state text are displayed.
    expect(find.text('TrOCR Lite'), findsOneWidget);
    expect(find.text('No Image Selected'), findsOneWidget);
  });
}
