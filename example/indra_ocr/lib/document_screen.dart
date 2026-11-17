import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';

import 'app_theme.dart';
import 'ocr_service.dart';

enum DocumentView { read, edit, image }

class DocumentScreen extends StatefulWidget {
  const DocumentScreen({
    super.key,
    required this.result,
    required this.imagePath,
    required this.initialMarkdown,
    required this.onChanged,
  });
  final OcrResult result;
  final String imagePath;
  final String initialMarkdown;
  final ValueChanged<String> onChanged;

  @override
  State<DocumentScreen> createState() => _DocumentScreenState();
}

class _DocumentScreenState extends State<DocumentScreen> {
  late final controller = TextEditingController(text: widget.initialMarkdown);
  DocumentView view = DocumentView.read;
  double fontSize = DocumentStyle.fontSize;
  bool saving = false;

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  void message(String text) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(text)));
  }

  Future<void> copy() async {
    await Clipboard.setData(ClipboardData(text: controller.text));
    message('Markdown copied');
  }

  Future<void> save() async {
    setState(() => saving = true);
    try {
      final name = await OcrService().exportMarkdown(controller.text);
      if (name != null) message('Saved $name');
    } on PlatformException catch (e) {
      message(e.message ?? 'Could not save the document. Please try again.');
    } finally {
      if (mounted) setState(() => saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Document'),
        actions: [
          IconButton(
            key: const Key('copy_markdown'),
            tooltip: 'Copy Markdown',
            onPressed: copy,
            icon: const Icon(Icons.copy_outlined),
          ),
          IconButton(
            key: const Key('save_markdown'),
            tooltip: 'Save .md file',
            onPressed: saving ? null : save,
            icon: const Icon(Icons.save_alt),
          ),
          PopupMenuButton<String>(
            tooltip: 'Document options',
            onSelected: (_) => Navigator.of(context).push(
              MaterialPageRoute<void>(
                builder: (_) => OriginalTextScreen(text: widget.result.text),
              ),
            ),
            itemBuilder: (_) => [
              const PopupMenuItem(
                value: 'original',
                child: Text('Original OCR text'),
              ),
            ],
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.symmetric(horizontal: AppSpacing.md),
              child: Row(
                children: [
                  for (final mode in DocumentView.values)
                    Padding(
                      padding: const EdgeInsets.only(right: AppSpacing.sm),
                      child: ChoiceChip(
                        key: Key('view_${mode.name}'),
                        label: Text(switch (mode) {
                          DocumentView.read => 'Preview',
                          DocumentView.edit => 'Markdown',
                          DocumentView.image => 'Source image',
                        }),
                        selected: view == mode,
                        onSelected: (_) => setState(() => view = mode),
                      ),
                    ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: AppSpacing.md),
              child: Row(
                children: [
                  Expanded(
                    child: Text(
                      view == DocumentView.read
                          ? '${widget.result.lines} lines · ${(widget.result.timings['total'] ?? 0).round()} ms · Offline'
                          : view == DocumentView.edit
                          ? 'Edit Markdown, then tap Preview to see the formatted document.'
                          : 'Pinch to zoom and compare the original.',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: theme.colorScheme.onSurfaceVariant,
                      ),
                    ),
                  ),
                  if (view == DocumentView.read) ...[
                    IconButton(
                      tooltip: 'Smaller text',
                      onPressed: fontSize <= 13
                          ? null
                          : () => setState(() => fontSize -= 2),
                      icon: const Icon(Icons.text_decrease),
                    ),
                    IconButton(
                      tooltip: 'Larger text',
                      onPressed: fontSize >= 25
                          ? null
                          : () => setState(() => fontSize += 2),
                      icon: const Icon(Icons.text_increase),
                    ),
                  ],
                ],
              ),
            ),
            if (saving) const LinearProgressIndicator(),
            const Divider(height: 1),
            Expanded(
              child: switch (view) {
                DocumentView.read => DocumentPaper(
                  markdown: controller.text,
                  fontSize: fontSize,
                ),
                DocumentView.edit => Padding(
                  padding: const EdgeInsets.all(AppSpacing.md),
                  child: TextField(
                    key: const Key('markdown_editor'),
                    controller: controller,
                    onChanged: widget.onChanged,
                    expands: true,
                    maxLines: null,
                    minLines: null,
                    textAlignVertical: TextAlignVertical.top,
                    autocorrect: false,
                    enableSuggestions: false,
                    style: theme.textTheme.bodyLarge?.copyWith(
                      fontFamily: 'monospace',
                      height: 1.5,
                    ),
                    decoration: const InputDecoration(
                      border: OutlineInputBorder(),
                      labelText: 'Markdown source',
                      alignLabelWithHint: true,
                    ),
                  ),
                ),
                DocumentView.image => InteractiveViewer(
                  minScale: 0.5,
                  maxScale: 6,
                  child: Center(
                    child: Image.file(
                      File(widget.imagePath),
                      cacheWidth: 2048,
                      fit: BoxFit.contain,
                      semanticLabel: 'Original scanned document',
                      errorBuilder: (_, _, _) =>
                          const Text('Source image is no longer available.'),
                    ),
                  ),
                ),
              },
            ),
          ],
        ),
      ),
    );
  }
}

class DocumentPaper extends StatelessWidget {
  const DocumentPaper({
    super.key,
    required this.markdown,
    required this.fontSize,
  });
  final String markdown;
  final double fontSize;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colors = theme.colorScheme;
    final body = theme.textTheme.bodyLarge!.copyWith(
      fontSize: fontSize,
      height: DocumentStyle.lineHeight,
      color: colors.onSurface,
    );
    final style = MarkdownStyleSheet.fromTheme(theme).copyWith(
      p: body,
      listBullet: body,
      h1: body.copyWith(
        fontSize: fontSize * 1.8,
        height: 1.25,
        fontWeight: FontWeight.w700,
      ),
      h2: body.copyWith(
        fontSize: fontSize * 1.4,
        height: 1.35,
        fontWeight: FontWeight.w700,
      ),
      h3: body.copyWith(fontSize: fontSize * 1.15, fontWeight: FontWeight.w600),
      h1Padding: const EdgeInsets.only(bottom: AppSpacing.md),
      h2Padding: const EdgeInsets.only(
        top: AppSpacing.md,
        bottom: AppSpacing.sm,
      ),
      blockSpacing: AppSpacing.md,
      blockquoteDecoration: BoxDecoration(
        color: colors.surfaceContainer,
        border: Border(left: BorderSide(color: colors.primary, width: 3)),
      ),
      code: body.copyWith(
        fontFamily: 'monospace',
        fontSize: fontSize * 0.9,
        backgroundColor: colors.surfaceContainer,
      ),
      codeblockDecoration: BoxDecoration(
        color: colors.surfaceContainer,
        borderRadius: BorderRadius.circular(AppRadius.md),
      ),
      tableBody: body,
    );
    return ColoredBox(
      color: colors.surfaceContainerLow,
      child: LayoutBuilder(
        builder: (context, constraints) => SingleChildScrollView(
          key: const Key('document_reader'),
          padding: EdgeInsets.all(
            constraints.maxWidth < 400 ? AppSpacing.sm : AppSpacing.md,
          ),
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(
                maxWidth: DocumentStyle.maxWidth,
              ),
              child: DecoratedBox(
                decoration: BoxDecoration(
                  color: theme.brightness == Brightness.light
                      ? DocumentStyle.paper
                      : colors.surface,
                  borderRadius: BorderRadius.circular(AppRadius.md),
                  border: Border.all(color: colors.outlineVariant),
                ),
                child: Padding(
                  padding: EdgeInsets.all(
                    constraints.maxWidth < 400
                        ? AppSpacing.md
                        : DocumentStyle.pagePadding,
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Text(
                        'EXTRACTED DOCUMENT',
                        style: theme.textTheme.labelSmall?.copyWith(
                          color: colors.primary,
                          letterSpacing: 1.8,
                        ),
                      ),
                      const SizedBox(height: AppSpacing.lg),
                      if (markdown.trim().isEmpty)
                        const Text(
                          'Your document is empty. Add text in Markdown.',
                        )
                      else
                        MarkdownBody(
                          data: markdown,
                          selectable: true,
                          styleSheet: style,
                          // Edited Markdown never fetches images or reads arbitrary file paths.
                          imageBuilder: (uri, title, alt) => Text(
                            '[Image: ${alt ?? title ?? 'not embedded'}]',
                          ),
                          onTapLink: (label, href, title) async {
                            if (href == null) return;
                            await Clipboard.setData(ClipboardData(text: href));
                            if (context.mounted) {
                              ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(content: Text('Link copied')),
                              );
                            }
                          },
                        ),
                      const SizedBox(height: AppSpacing.lg),
                      const Divider(),
                      Text(
                        'Formatting is inferred from the scan. You can correct it in Markdown; original OCR text is kept separately.',
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: colors.onSurfaceVariant,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class OriginalTextScreen extends StatelessWidget {
  const OriginalTextScreen({super.key, required this.text});
  final String text;
  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(
      title: const Text('Original OCR text'),
      actions: [
        IconButton(
          tooltip: 'Copy original text',
          icon: const Icon(Icons.copy),
          onPressed: () async {
            await Clipboard.setData(ClipboardData(text: text));
            if (context.mounted) {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Original text copied')),
              );
            }
          },
        ),
      ],
    ),
    body: SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(AppSpacing.lg),
        child: SelectableText(text),
      ),
    ),
  );
}
