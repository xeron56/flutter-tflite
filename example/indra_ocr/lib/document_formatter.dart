import 'ocr_service.dart';

/// Conservative formatting only: never summarizes, translates or dehyphenates.
/// Geometric headings/paragraphs are guesses; original OCR text stays available.
abstract final class DocumentFormatter {
  static String escape(String text) => text.replaceAllMapped(
    RegExp(r'([\\`*_{}\[\]()#+\-.!|>~<&])'),
    (m) => '\\${m[0]}',
  );

  static String format(OcrResult result) {
    final lines = result.regions
        .where((line) => line.text.trim().isNotEmpty)
        .toList();
    if (lines.isEmpty) {
      return result.text.split('\n').map(escape).join('  \n').trim();
    }
    final heights = lines.map((l) => l.height).where((h) => h > 0).toList()
      ..sort();
    final bodyHeight = heights.isEmpty ? 1.0 : heights[heights.length ~/ 2];
    // Detector boxes tightly enclose ink, so normal line spacing can exceed
    // the ink height. Learn the page's typical gap before splitting paragraphs.
    final gaps = <double>[
      for (var i = 1; i < lines.length; i++)
        if (lines[i].top >= lines[i - 1].bottom)
          lines[i].top - lines[i - 1].bottom,
    ]..sort();
    final typicalGap = gaps.isEmpty ? 0.0 : gaps[gaps.length ~/ 2];
    final paragraphGap = typicalGap * 1.6 > bodyHeight * 0.8
        ? typicalGap * 1.6
        : bodyHeight * 0.8;
    final blocks = <String>[];
    final paragraph = <String>[];
    void flush() {
      if (paragraph.isNotEmpty) {
        blocks.add(paragraph.join('  \n'));
        paragraph.clear();
      }
    }

    var previousWasList = false;
    for (var i = 0; i < lines.length; i++) {
      final line = lines[i];
      final text = line.text.trim();
      final bullet = RegExp(r'^[-*+•◦▪‣○]\s+(.+)$').firstMatch(text);
      final number = RegExp(r'^(\d{1,3})[.)]\s+(.+)$').firstMatch(text);
      final gap = i == 0 ? 0.0 : line.top - lines[i - 1].bottom;
      if (bullet != null || number != null) {
        flush();
        final item = bullet != null
            ? '- ${escape(bullet[1]!)}'
            : '${number![1]}. ${escape(number[2]!)}';
        if (previousWasList && gap < bodyHeight * 1.2) {
          blocks[blocks.length - 1] += '\n$item';
        } else {
          blocks.add(item);
        }
        previousWasList = true;
        continue;
      }
      final firstTitle =
          i == 0 &&
          lines.length >= 3 &&
          text.length <= 80 &&
          lines[1].top - line.bottom > bodyHeight * 0.9;
      final largeHeading =
          text.length <= 160 && line.height >= bodyHeight * 1.4;
      if (firstTitle || largeHeading) {
        flush();
        blocks.add(
          '${firstTitle || line.height >= bodyHeight * 1.8 ? '#' : '##'} ${escape(text)}',
        );
      } else {
        if (gap > paragraphGap || gap < -bodyHeight || previousWasList) {
          flush();
        }
        paragraph.add(escape(text));
      }
      previousWasList = false;
    }
    flush();
    return '${blocks.join('\n\n')}\n';
  }
}
