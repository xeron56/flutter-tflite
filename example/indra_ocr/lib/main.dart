import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

import 'ocr_service.dart';
import 'app_theme.dart';
import 'document_formatter.dart';
import 'document_screen.dart';

void main() {
  LicenseRegistry.addLicense(() async* {
    yield LicenseEntryWithLineBreaks([
      'Datalab Indra source',
    ], await rootBundle.loadString('third_party/INDRA_CODE_LICENSE'));
    yield LicenseEntryWithLineBreaks(
      ['Datalab Indra model weights'],
      await rootBundle.loadString('android/app/src/main/assets/models/LICENSE'),
    );
  });
  runApp(const IndraApp());
}

class IndraApp extends StatelessWidget {
  const IndraApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'Indra OCR',
    debugShowCheckedModeBanner: false,
    theme: theme(Brightness.light),
    darkTheme: theme(Brightness.dark),
    home: const ScanScreen(),
  );
  static ThemeData theme(Brightness brightness) {
    final base = ThemeData(
      colorScheme: ColorScheme.fromSeed(
        seedColor: AppColors.seed,
        brightness: brightness,
      ),
      useMaterial3: true,
    );
    return base.copyWith(
      textTheme: AppTypography.apply(base.textTheme),
      cardTheme: CardThemeData(
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(AppRadius.md),
        ),
      ),
    );
  }
}

class ScanScreen extends StatefulWidget {
  const ScanScreen({super.key});
  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> {
  final service = OcrService();
  final picker = ImagePicker();
  Map<Object?, Object?>? device;
  String? path;
  String? error;
  OcrResult? result;
  String? markdown;
  Map<Object?, Object?>? benchmark;
  bool busy = false;
  bool quality = false;
  String status = '';

  @override
  void initState() {
    super.initState();
    initialize();
  }

  Future<void> initialize() async {
    try {
      final info = await service.info();
      final lost = await picker.retrieveLostData();
      if (!mounted) return;
      setState(() {
        device = info;
        if (lost.files?.isNotEmpty ?? false) path = lost.files!.first.path;
        if (lost.exception != null) error = lost.exception!.message;
      });
    } catch (e) {
      if (mounted) setState(() => error = readableError(e));
    }
  }

  String readableError(Object e) =>
      e is PlatformException ? e.message ?? e.code : e.toString();

  Future<void> task(String label, Future<void> Function() work) async {
    if (busy) return;
    setState(() {
      busy = true;
      status = label;
      error = null;
    });
    try {
      await work();
    } catch (e) {
      if (mounted) setState(() => error = readableError(e));
    } finally {
      if (mounted) {
        setState(() {
          busy = false;
          status = '';
        });
      }
    }
  }

  Future<void> pick(ImageSource source) => task('Opening images…', () async {
    final file = await picker.pickImage(source: source);
    if (file != null && mounted) {
      setState(() {
        path = file.path;
        result = null;
        benchmark = null;
      });
    }
  });

  Future<void> sample(String kind) => task('Preparing sample…', () async {
    final file = await service.sample(kind);
    if (mounted) {
      setState(() {
        path = file;
        result = null;
        benchmark = null;
      });
    }
  });

  Future<void> recognize() => task('Reading your document…', () async {
    // Clear the previous result so a failed run cannot look like fresh output.
    setState(() => result = null);
    final output = await service.recognize(path!, quality: quality);
    if (mounted) {
      setState(() {
        result = output;
        markdown = DocumentFormatter.format(output);
      });
      if (output.text.isNotEmpty) openDocument();
    }
  });

  void openDocument() {
    if (result == null || markdown == null) return;
    final document = result!;
    final sourcePath = path!;
    final sourceMarkdown = markdown!;
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => DocumentScreen(
          result: document,
          imagePath: sourcePath,
          initialMarkdown: sourceMarkdown,
          onChanged: (value) {
            if (mounted) setState(() => markdown = value);
          },
        ),
      ),
    );
  }

  Future<void> tune() =>
      task('Comparing engines · this can take a minute…', () async {
        setState(() {
          benchmark = null;
        });
        final output = await service.benchmark(path!, quality: quality);
        final info = await service.info();
        if (mounted) {
          setState(() {
            benchmark = output;
            device = info;
          });
        }
      });

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final type = Theme.of(context).textTheme;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Indra'),
        actions: [
          IconButton(
            key: const Key('about'),
            tooltip: 'About this app',
            onPressed: () => showAboutDialog(
              context: context,
              applicationName: 'Indra OCR',
              applicationVersion: '1.1 · models v1.1.0',
              children: const [
                Text(
                  'Offline Android port of Datalab Indra. Uses ONNX Runtime and OpenCV. '
                  'This fast port reads images with greedy decoding and geometric reading order. '
                  'It does not include upstream language-model correction, PDF parsing, table reconstruction, or math recognition. '
                  'Model weights use Datalab’s modified OpenRAIL-M license; see README and bundled LICENSE before commercial use.',
                ),
              ],
            ),
            icon: const Icon(Icons.info_outline),
          ),
        ],
      ),
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 840),
            child: CustomScrollView(
              slivers: [
                SliverPadding(
                  padding: const EdgeInsets.all(AppSpacing.lg),
                  sliver: SliverList.list(
                    children: [
                      Text(
                        'From scan to document.',
                        style: type.headlineMedium,
                      ),
                      const SizedBox(height: AppSpacing.sm),
                      Text(
                        'Read beautifully. Edit freely. Keep it offline.',
                        style: type.bodyLarge?.copyWith(
                          color: colors.onSurfaceVariant,
                        ),
                      ),
                      const SizedBox(height: AppSpacing.lg),
                      DeviceCard(device: device),
                      const SizedBox(height: AppSpacing.md),
                      if (path == null)
                        const EmptyDocument()
                      else
                        ClipRRect(
                          borderRadius: BorderRadius.circular(AppRadius.md),
                          child: Image.file(
                            File(path!),
                            key: ValueKey(path),
                            height: 250,
                            fit: BoxFit.contain,
                            semanticLabel: 'Selected document',
                            cacheWidth: 1000,
                            errorBuilder: (_, _, _) => const Padding(
                              padding: EdgeInsets.all(AppSpacing.md),
                              child: Text(
                                'Preview unavailable. Try another image.',
                              ),
                            ),
                          ),
                        ),
                      const SizedBox(height: AppSpacing.md),
                      Wrap(
                        spacing: AppSpacing.sm,
                        runSpacing: AppSpacing.sm,
                        children: [
                          FilledButton.tonalIcon(
                            key: const Key('gallery'),
                            onPressed: busy
                                ? null
                                : () => pick(ImageSource.gallery),
                            icon: const Icon(Icons.photo_library_outlined),
                            label: const Text('Choose image'),
                          ),
                          OutlinedButton.icon(
                            key: const Key('camera'),
                            onPressed: busy
                                ? null
                                : () => pick(ImageSource.camera),
                            icon: const Icon(Icons.photo_camera_outlined),
                            label: const Text('Camera'),
                          ),
                          PopupMenuButton<String>(
                            key: const Key('sample'),
                            enabled: !busy,
                            tooltip: 'Try a sample',
                            onSelected: sample,
                            itemBuilder: (_) => const [
                              PopupMenuItem(
                                value: 'english',
                                child: Text('English sample'),
                              ),
                              PopupMenuItem(
                                value: 'bengali',
                                child: Text('Bengali sample'),
                              ),
                              PopupMenuItem(
                                value: 'scan',
                                child: Text('Korean document'),
                              ),
                            ],
                            child: const Padding(
                              padding: EdgeInsets.all(AppSpacing.md),
                              child: Text('Try a sample ▾'),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: AppSpacing.md),
                      SwitchListTile.adaptive(
                        key: const Key('quality'),
                        contentPadding: EdgeInsets.zero,
                        title: const Text('Small text mode'),
                        subtitle: const Text(
                          'Adds a higher-resolution pass when needed.',
                        ),
                        value: quality,
                        onChanged: busy
                            ? null
                            : (v) => setState(() {
                                quality = v;
                                benchmark = null;
                              }),
                      ),
                      const SizedBox(height: AppSpacing.sm),
                      Wrap(
                        spacing: AppSpacing.sm,
                        runSpacing: AppSpacing.sm,
                        children: [
                          FilledButton.icon(
                            key: const Key('recognize'),
                            onPressed: busy || path == null ? null : recognize,
                            icon: const Icon(Icons.document_scanner_outlined),
                            label: const Text('Create document'),
                          ),
                          OutlinedButton.icon(
                            key: const Key('tune'),
                            onPressed: busy || path == null ? null : tune,
                            icon: const Icon(Icons.speed),
                            label: const Text('Tune on this image'),
                          ),
                        ],
                      ),
                      if (busy) ...[
                        const SizedBox(height: AppSpacing.lg),
                        const LinearProgressIndicator(),
                        const SizedBox(height: AppSpacing.sm),
                        Semantics(liveRegion: true, child: Text(status)),
                      ],
                      if (error != null) ...[
                        const SizedBox(height: AppSpacing.md),
                        Semantics(
                          liveRegion: true,
                          child: Text(
                            error!,
                            style: type.bodyMedium?.copyWith(
                              color: colors.error,
                            ),
                          ),
                        ),
                      ],
                      if (result != null) ...[
                        const SizedBox(height: AppSpacing.lg),
                        ResultCard(result: result!, onOpen: openDocument),
                      ],
                      if (benchmark != null) ...[
                        const SizedBox(height: AppSpacing.lg),
                        BenchmarkCard(data: benchmark!),
                      ],
                      const SizedBox(height: AppSpacing.lg),
                      Text(
                        'Tuning compares 3 warm runs per engine and saves the fastest setting with identical text. Results depend on the document and phone temperature.',
                        style: type.bodySmall?.copyWith(
                          color: colors.onSurfaceVariant,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class DeviceCard extends StatelessWidget {
  const DeviceCard({required this.device, super.key});
  final Map<Object?, Object?>? device;
  @override
  Widget build(BuildContext context) => Card(
    child: Padding(
      padding: const EdgeInsets.all(AppSpacing.md),
      child: Row(
        children: [
          const Icon(Icons.memory),
          const SizedBox(width: AppSpacing.md),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  device == null
                      ? 'Checking device…'
                      : '${device!['soc']} · Android ${device!['android']}',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                Text(
                  device == null
                      ? 'Native Android inference'
                      : '${device!['engine']} · bundled models',
                ),
              ],
            ),
          ),
        ],
      ),
    ),
  );
}

class EmptyDocument extends StatelessWidget {
  const EmptyDocument({super.key});
  @override
  Widget build(BuildContext context) => Card(
    child: Padding(
      padding: const EdgeInsets.all(AppSpacing.lg),
      child: Column(
        children: [
          Icon(
            Icons.description_outlined,
            size: 64,
            color: Theme.of(context).colorScheme.primary,
          ),
          const SizedBox(height: AppSpacing.md),
          Text(
            'Choose a document to begin',
            style: Theme.of(context).textTheme.titleLarge,
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: AppSpacing.sm),
          const Text(
            'Use a clear, straight photo or try a sample below.',
            textAlign: TextAlign.center,
          ),
        ],
      ),
    ),
  );
}

class ResultCard extends StatelessWidget {
  const ResultCard({required this.result, required this.onOpen, super.key});
  final OcrResult result;
  final VoidCallback onOpen;
  @override
  Widget build(BuildContext context) => Card(
    child: Padding(
      padding: const EdgeInsets.all(AppSpacing.md),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Wrap(
            spacing: AppSpacing.md,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              Text(
                'Your document is ready',
                style: Theme.of(context).textTheme.titleLarge,
              ),
              Text(
                '${result.timings['total']!.toStringAsFixed(0)} ms · ${result.lines} regions',
              ),
              IconButton(
                key: const Key('copy'),
                tooltip: 'Copy text',
                onPressed: result.text.isEmpty
                    ? null
                    : () async {
                        await Clipboard.setData(
                          ClipboardData(text: result.text),
                        );
                        if (context.mounted) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(content: Text('Text copied')),
                          );
                        }
                      },
                icon: const Icon(Icons.copy_outlined),
              ),
            ],
          ),
          Text(result.backend, style: Theme.of(context).textTheme.bodySmall),
          const SizedBox(height: AppSpacing.md),
          if (result.text.isEmpty)
            const Text(
              'No readable text found. Try a closer, sharper photo or small text mode.',
            )
          else ...[
            Text(result.text, maxLines: 3, overflow: TextOverflow.ellipsis),
            const SizedBox(height: AppSpacing.md),
            FilledButton.icon(
              key: const Key('open_document'),
              onPressed: onOpen,
              icon: const Icon(Icons.auto_stories_outlined),
              label: const Text('Open document'),
            ),
          ],
          const SizedBox(height: AppSpacing.md),
          ExpansionTile(
            tilePadding: EdgeInsets.zero,
            title: const Text('Processing details'),
            children: [
              for (final entry in result.timings.entries)
                ListTile(
                  dense: true,
                  title: Text(entry.key),
                  trailing: Text('${entry.value.toStringAsFixed(1)} ms'),
                ),
              ListTile(
                dense: true,
                title: const Text('Orientation correction'),
                trailing: Text('${result.rotation}°'),
              ),
            ],
          ),
        ],
      ),
    ),
  );
}

class BenchmarkCard extends StatelessWidget {
  const BenchmarkCard({required this.data, super.key});
  final Map<Object?, Object?> data;
  @override
  Widget build(BuildContext context) => Card(
    child: Padding(
      padding: const EdgeInsets.all(AppSpacing.md),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Tuning complete',
            style: Theme.of(context).textTheme.titleLarge,
          ),
          const SizedBox(height: AppSpacing.sm),
          Text('Saved: ${data['selected']}'),
          const SizedBox(height: AppSpacing.md),
          for (final row in (data['results']! as List).cast<Map>())
            Padding(
              padding: const EdgeInsets.only(bottom: AppSpacing.md),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    row['label'] as String,
                    style: Theme.of(context).textTheme.titleSmall,
                  ),
                  Text(
                    row['error'] != null
                        ? 'Unavailable: ${row['error']}'
                        : '${(row['medianMs'] as num).toStringAsFixed(0)} ms median · ${row['matchesCpuText'] == true ? 'same text' : 'different or empty text; excluded'}',
                  ),
                ],
              ),
            ),
          Text(
            data['note']! as String,
            style: Theme.of(context).textTheme.bodySmall,
          ),
        ],
      ),
    ),
  );
}
