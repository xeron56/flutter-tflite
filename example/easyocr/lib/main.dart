import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'easyocr_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  runApp(const EasyOcrApp());
}

class EasyOcrApp extends StatelessWidget {
  const EasyOcrApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EasyOCR Lite',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF12121A),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF6366F1),
          secondary: Color(0xFF10B981),
          surface: Color(0xFF1E1E2A),
          error: Color(0xFFEF4444),
        ),
      ),
      home: const OcrHomeScreen(),
    );
  }
}

class OcrHomeScreen extends StatefulWidget {
  const OcrHomeScreen({super.key});

  @override
  State<OcrHomeScreen> createState() => _OcrHomeScreenState();
}

class _OcrHomeScreenState extends State<OcrHomeScreen> {
  final EasyOcrService _ocrService = EasyOcrService();
  final ImagePicker _picker = ImagePicker();
  final TextEditingController _fullTextController = TextEditingController();

  File? _imageFile;
  img.Image? _decodedImage;
  List<OcrResult> _ocrResults = [];
  bool _isLoading = false;
  String _ocrLanguage = 'en';
  HardwareAccelerator _selectedAccelerator = HardwareAccelerator.cpu;
  bool _showFullTextTab = false;
  String? _statusText;

  @override
  void initState() {
    super.initState();
    _initOcrService();
  }

  Future<void> _initOcrService() async {
    setState(() {
      _isLoading = true;
      _statusText = 'Loading detection & English recognition models...';
    });
    try {
      await _ocrService.loadModels('en', accelerator: _selectedAccelerator);
      setState(() {
        _isLoading = false;
        _statusText = 'Models loaded successfully';
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _statusText = 'Error loading models: $e';
      });
      _showErrorDialog(
        'Failed to load models. Make sure detector.tflite and recognizer_en.tflite are in your assets. Details: $e',
      );
    }
  }

  Future<void> _toggleLanguage(String? newLang) async {
    if (newLang == null || newLang == _ocrLanguage) return;
    setState(() {
      _isLoading = true;
      _ocrLanguage = newLang;
      _statusText =
          'Loading ${newLang == 'bn' ? 'Bangla' : 'English'} recognizer...';
      _ocrResults = [];
    });

    try {
      await _ocrService.loadModels(newLang, accelerator: _selectedAccelerator);
      setState(() {
        _isLoading = false;
        _statusText =
            'Switched to ${newLang == 'bn' ? 'Bangla' : 'English'} successfully';
      });
      // Re-run OCR if image is already loaded
      if (_decodedImage != null) {
        _runOcr();
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
        _statusText = 'Error: $e';
      });
      _showErrorDialog(
        newLang == 'bn'
            ? 'Failed to load Bangla model. Make sure recognizer_bn.tflite is in your assets. Details: $e'
            : 'Failed to load English model. Make sure recognizer_en.tflite is in your assets. Details: $e',
      );
    }
  }

  Future<void> _changeAccelerator(HardwareAccelerator newAcc) async {
    if (newAcc == _selectedAccelerator) return;
    setState(() {
      _isLoading = true;
      _selectedAccelerator = newAcc;
      _statusText = 'Reloading models with ${newAcc.name.toUpperCase()}...';
      _ocrResults = [];
    });

    try {
      await _ocrService.loadModels(_ocrLanguage, accelerator: newAcc);
      
      final bool detectorFellback = newAcc != HardwareAccelerator.cpu &&
          _ocrService.detectorAccelerator == HardwareAccelerator.cpu;
      final bool recognizerFellback = newAcc != HardwareAccelerator.cpu &&
          _ocrService.recognizerAccelerator == HardwareAccelerator.cpu;

      String statusMsg = 'Models loaded';
      if (newAcc != HardwareAccelerator.cpu) {
        if (detectorFellback && recognizerFellback) {
          statusMsg = 'Accelerator failed: both models fell back to CPU';
          _selectedAccelerator = HardwareAccelerator.cpu;
          _showFallbackNotification(
            'Requested acceleration was not supported on this device. Both models fell back to CPU.',
          );
        } else if (detectorFellback) {
          statusMsg = 'Detector fell back to CPU';
          _showFallbackNotification(
            'CRAFT Detector model fell back to CPU. CRNN Recognizer is running on ${newAcc.name.toUpperCase()}.',
          );
        } else if (recognizerFellback) {
          statusMsg = 'Recognizer fell back to CPU';
          _showFallbackNotification(
            'CRNN Recognizer model fell back to CPU. CRAFT Detector is running on ${newAcc.name.toUpperCase()}.',
          );
        } else {
          statusMsg = 'Both models running on ${newAcc.name.toUpperCase()}';
        }
      }

      setState(() {
        _isLoading = false;
        _statusText = statusMsg;
      });
      if (_decodedImage != null) {
        _runOcr();
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
        _statusText = 'Failed to load on ${newAcc.name.toUpperCase()}: $e';
      });
      _showErrorDialog(
        'Failed to load models with ${newAcc.name.toUpperCase()} acceleration.\n\n'
        'Details: $e\n\n'
        'Falling back to CPU...',
      );
      _selectedAccelerator = HardwareAccelerator.cpu;
      _initOcrService();
    }
  }

  void _showFallbackNotification(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Row(
          children: [
            const Icon(Icons.info_outline, color: Colors.orange),
            const SizedBox(width: 8),
            Expanded(child: Text(message)),
          ],
        ),
        behavior: SnackBarBehavior.floating,
        duration: const Duration(seconds: 4),
      ),
    );
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? pickedFile = await _picker.pickImage(source: source);
      if (pickedFile == null) return;

      setState(() {
        _imageFile = File(pickedFile.path);
        _decodedImage = null;
        _ocrResults = [];
        _isLoading = true;
        _statusText = 'Decoding image...';
      });

      final bytes = await _imageFile!.readAsBytes();
      final decoded = img.decodeImage(bytes);

      if (decoded == null) {
        throw Exception('Failed to decode image pixels.');
      }

      setState(() {
        _decodedImage = decoded;
      });

      _runOcr();
    } catch (e) {
      setState(() {
        _isLoading = false;
        _statusText = 'Error picking image: $e';
      });
      _showErrorDialog('Could not decode the selected image file: $e');
    }
  }

  Future<void> _runOcr() async {
    if (_decodedImage == null) return;
    setState(() {
      _isLoading = true;
      _statusText = 'Running text detection & recognition...';
    });

    try {
      final results = await _ocrService.runOcr(_decodedImage!);
      setState(() {
        _ocrResults = results;
        _fullTextController.text = results.map((r) => r.text).join('\n');
        _isLoading = false;
        _statusText = 'Processed successfully: found ${results.length} lines';
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _statusText = 'OCR failed';
      });

      String errorMsg = e.toString();
      if (errorMsg.contains('vocabulary size mismatch')) {
        _showModelMismatchDialog();
      } else {
        _showErrorDialog('OCR Inference error: $e');
      }
    }
  }

  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Error'),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  void _showModelMismatchDialog() {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Row(
          children: const [
            Icon(Icons.warning_amber_rounded, color: Colors.orange, size: 28),
            SizedBox(width: 8),
            Text('Bangla Model Missing'),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: const [
            Text(
              'The selected model is still the English model placeholder (97 outputs) instead of the actual Bangla model (170 outputs).',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 12),
            Text(
              'To enable Bangla OCR:\n'
              '1. Export the EasyOCR Bangla model to TFLite (e.g. using Qualcomm AI Hub or conversion scripts).\n'
              '2. Replace `assets/models/recognizer_bn.tflite` in this project with your exported model.\n'
              '3. Rebuild and run the app.',
              style: TextStyle(height: 1.4),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('I Understand'),
          ),
        ],
      ),
    );
  }

  void _copyToClipboard() {
    if (_ocrResults.isEmpty) return;
    final String fullText = _ocrResults.map((r) => r.text).join('\n');
    Clipboard.setData(ClipboardData(text: fullText));
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Copied all text to clipboard'),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  @override
  void dispose() {
    _ocrService.dispose();
    _fullTextController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: const [
            Text(
              'EasyOCR Lite',
              style: TextStyle(
                fontWeight: FontWeight.w800,
                fontSize: 20,
                letterSpacing: -0.5,
              ),
            ),
            Text(
              'On-Device CRAFT + CRNN TFLite Pipeline',
              style: TextStyle(fontSize: 11, color: Colors.grey),
            ),
          ],
        ),
        actions: [
          if (_ocrResults.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.copy_rounded),
              onPressed: _copyToClipboard,
              tooltip: 'Copy all text',
            ),
        ],
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(
              horizontal: 16.0,
              vertical: 8.0,
            ),
            child: Column(
              children: [
                Container(
                  padding: const EdgeInsets.all(4),
                  decoration: BoxDecoration(
                    color: Theme.of(context).colorScheme.surface,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Row(
                    children: [
                      Expanded(
                        child: _LangOptionButton(
                          label: 'English',
                          isSelected: _ocrLanguage == 'en',
                          onTap: () => _toggleLanguage('en'),
                        ),
                      ),
                      Expanded(
                        child: _LangOptionButton(
                          label: 'বাংল (Bangla)',
                          isSelected: _ocrLanguage == 'bn',
                          onTap: () => _toggleLanguage('bn'),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 8),
                Container(
                  padding: const EdgeInsets.all(4),
                  decoration: BoxDecoration(
                    color: Theme.of(context).colorScheme.surface,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Row(
                    children: [
                      Expanded(
                        child: _AccOptionButton(
                          label: 'CPU',
                          isSelected: _selectedAccelerator == HardwareAccelerator.cpu,
                          onTap: () => _changeAccelerator(HardwareAccelerator.cpu),
                        ),
                      ),
                      Expanded(
                        child: _AccOptionButton(
                          label: 'GPU',
                          isSelected: _selectedAccelerator == HardwareAccelerator.gpu,
                          onTap: () => _changeAccelerator(HardwareAccelerator.gpu),
                        ),
                      ),
                      Expanded(
                        child: _AccOptionButton(
                          label: 'NPU (Hexagon)',
                          isSelected: _selectedAccelerator == HardwareAccelerator.npu,
                          onTap: () => _changeAccelerator(HardwareAccelerator.npu),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          Expanded(
            child: _imageFile == null
                ? _buildEmptyState()
                : _buildImageWorkspace(),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            decoration: BoxDecoration(
              color: const Color(0xFF161622),
              border: Border(
                top: BorderSide(color: Colors.white.withOpacity(0.05)),
              ),
            ),
            child: Row(
              children: [
                if (_isLoading)
                  const SizedBox(
                    width: 14,
                    height: 14,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                else
                  Icon(
                    Icons.circle,
                    size: 10,
                    color: _statusText != null && _statusText!.contains('Error')
                        ? Colors.red
                        : Theme.of(context).colorScheme.secondary,
                  ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    _statusText ?? 'Ready',
                    style: const TextStyle(fontSize: 12, color: Colors.grey),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.surface,
                shape: BoxShape.circle,
              ),
              child: Icon(
                Icons.document_scanner_outlined,
                size: 64,
                color: Theme.of(context).colorScheme.primary,
              ),
            ),
            const SizedBox(height: 24),
            const Text(
              'No Image Selected',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            const Text(
              'Take a photo or pick an image to run the EasyOCR pipeline on-device.',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.grey, height: 1.4),
            ),
            const SizedBox(height: 32),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _buildActionButton(
                  icon: Icons.camera_alt_rounded,
                  label: 'Camera',
                  onTap: () => _pickImage(ImageSource.camera),
                ),
                const SizedBox(width: 16),
                _buildActionButton(
                  icon: Icons.photo_library_rounded,
                  label: 'Gallery',
                  onTap: () => _pickImage(ImageSource.gallery),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildImageWorkspace() {
    return Column(
      children: [
        Expanded(
          flex: 3,
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Card(
              clipBehavior: Clip.antiAlias,
              color: Theme.of(context).colorScheme.surface,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
              ),
              child: Stack(
                fit: StackFit.expand,
                children: [
                  InteractiveViewer(
                    maxScale: 5.0,
                    minScale: 1.0,
                    child: Stack(
                      fit: StackFit.expand,
                      children: [
                        Image.file(_imageFile!, fit: BoxFit.contain),
                        if (_decodedImage != null && _ocrResults.isNotEmpty)
                          Positioned.fill(
                            child: CustomPaint(
                              painter: OcrBoxPainter(
                                results: _ocrResults,
                                imageWidth: _decodedImage!.width,
                                imageHeight: _decodedImage!.height,
                                primaryColor: Theme.of(context).colorScheme.primary,
                                secondaryColor: Theme.of(
                                  context,
                                ).colorScheme.secondary,
                              ),
                            ),
                          ),
                      ],
                    ),
                  ),
                  if (_decodedImage != null && !_isLoading)
                    Positioned(
                      top: 12,
                      right: 12,
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 8,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.black.withOpacity(0.75),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: Colors.white10),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Row(
                              children: [
                                const Icon(
                                  Icons.speed_rounded,
                                  size: 14,
                                  color: Color(0xFF10B981),
                                ),
                                const SizedBox(width: 4),
                                Text(
                                  'Backend: ${_ocrService.currentAccelerator.name.toUpperCase()}',
                                  style: const TextStyle(
                                    fontSize: 10,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.white70,
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 6),
                            Text(
                              'Detector: ${_ocrService.detectorInferenceTimeMs.toStringAsFixed(1)} ms (${_ocrService.detectorAccelerator.name.toUpperCase()})',
                              style: const TextStyle(
                                fontSize: 10,
                                color: Colors.white60,
                              ),
                            ),
                            Text(
                              'Recognizer: ${_ocrService.recognizerInferenceTimeMs.toStringAsFixed(1)} ms/box (${_ocrService.recognizerAccelerator.name.toUpperCase()})',
                              style: const TextStyle(
                                fontSize: 10,
                                color: Colors.white60,
                              ),
                            ),
                            Text(
                              'Total Rec: ${_ocrService.totalRecognizerTimeMs.toStringAsFixed(1)} ms (${_ocrResults.length} boxes)',
                              style: const TextStyle(
                                fontSize: 10,
                                color: Colors.white60,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  if (_isLoading)
                    Positioned.fill(
                      child: Container(
                        color: Colors.black.withOpacity(0.5),
                        child: const Center(child: CircularProgressIndicator()),
                      ),
                    ),
                ],
              ),
            ),
          ),
        ),
        Expanded(
          flex: 2,
          child: Container(
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.surface,
              borderRadius: const BorderRadius.vertical(
                top: Radius.circular(24),
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Padding(
                  padding: const EdgeInsets.only(
                    left: 20,
                    right: 12,
                    top: 16,
                    bottom: 8,
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Row(
                        children: [
                          _TabButton(
                            label: 'Lines',
                            isSelected: !_showFullTextTab,
                            onTap: () => setState(() => _showFullTextTab = false),
                          ),
                          const SizedBox(width: 8),
                          _TabButton(
                            label: 'Full Text',
                            isSelected: _showFullTextTab,
                            onTap: () => setState(() => _showFullTextTab = true),
                          ),
                        ],
                      ),
                      Row(
                        children: [
                          IconButton(
                            icon: const Icon(Icons.refresh_rounded, size: 20),
                            onPressed: _runOcr,
                            tooltip: 'Re-run OCR',
                          ),
                          IconButton(
                            icon: const Icon(
                              Icons.add_a_photo_outlined,
                              size: 20,
                            ),
                            onPressed: () => _pickImage(ImageSource.gallery),
                            tooltip: 'Pick another image',
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
                Expanded(
                  child: _ocrResults.isEmpty
                      ? Center(
                          child: Text(
                            _isLoading
                                ? 'Running OCR...'
                                : 'No text recognized',
                            style: const TextStyle(color: Colors.grey),
                          ),
                        )
                      : (_showFullTextTab
                          ? Padding(
                              padding: const EdgeInsets.only(
                                left: 16.0,
                                right: 16.0,
                                bottom: 16.0,
                              ),
                              child: Container(
                                decoration: BoxDecoration(
                                  color: Colors.black.withOpacity(0.2),
                                  borderRadius: BorderRadius.circular(12),
                                  border: Border.all(color: Colors.white10),
                                ),
                                child: Column(
                                  children: [
                                    Expanded(
                                      child: TextField(
                                        controller: _fullTextController,
                                        maxLines: null,
                                        style: const TextStyle(
                                          fontSize: 14,
                                          height: 1.5,
                                        ),
                                        decoration: const InputDecoration(
                                          contentPadding: EdgeInsets.all(12),
                                          border: InputBorder.none,
                                          hintText: 'Recognized text will appear here...',
                                        ),
                                      ),
                                    ),
                                    const Divider(height: 1, color: Colors.white10),
                                    Padding(
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 8.0,
                                        vertical: 4.0,
                                      ),
                                      child: Row(
                                        mainAxisAlignment: MainAxisAlignment.end,
                                        children: [
                                          TextButton.icon(
                                            onPressed: () {
                                              Clipboard.setData(
                                                ClipboardData(text: _fullTextController.text),
                                              );
                                              ScaffoldMessenger.of(context).showSnackBar(
                                                const SnackBar(
                                                  content: Text('Copied full text to clipboard'),
                                                  behavior: SnackBarBehavior.floating,
                                                  duration: Duration(seconds: 1),
                                                ),
                                              );
                                            },
                                            icon: const Icon(Icons.copy_rounded, size: 16),
                                            label: const Text(
                                              'Copy All',
                                              style: TextStyle(fontSize: 12),
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            )
                          : ListView.separated(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 16,
                                vertical: 8,
                              ),
                              itemCount: _ocrResults.length,
                              separatorBuilder: (context, index) =>
                                  const Divider(height: 1, color: Colors.white10),
                              itemBuilder: (context, index) {
                                final result = _ocrResults[index];
                                return ListTile(
                                  contentPadding: const EdgeInsets.symmetric(
                                    horizontal: 8,
                                    vertical: 4,
                                  ),
                                  title: Text(
                                    result.text,
                                    style: const TextStyle(
                                      fontWeight: FontWeight.w500,
                                      fontSize: 15,
                                    ),
                                  ),
                                  trailing: Container(
                                    padding: const EdgeInsets.symmetric(
                                      horizontal: 6,
                                      vertical: 2,
                                    ),
                                    decoration: BoxDecoration(
                                      color: Colors.white.withOpacity(0.08),
                                      borderRadius: BorderRadius.circular(4),
                                    ),
                                    child: Text(
                                      '${(result.confidence * 100).toStringAsFixed(0)}%',
                                      style: TextStyle(
                                        fontSize: 10,
                                        color: result.confidence > 0.8
                                            ? Theme.of(
                                                context,
                                              ).colorScheme.secondary
                                            : Colors.amber,
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ),
                                  onTap: () {
                                    Clipboard.setData(
                                      ClipboardData(text: result.text),
                                    );
                                    ScaffoldMessenger.of(context).showSnackBar(
                                      SnackBar(
                                        content: Text('Copied: "${result.text}"'),
                                        behavior: SnackBarBehavior.floating,
                                        duration: const Duration(seconds: 1),
                                      ),
                                    );
                                  },
                                );
                              },
                            )),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildActionButton({
    required IconData icon,
    required String label,
    required VoidCallback onTap,
  }) {
    return ElevatedButton.icon(
      style: ElevatedButton.styleFrom(
        backgroundColor: Theme.of(context).colorScheme.primary,
        foregroundColor: Colors.white,
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        elevation: 0,
      ),
      onPressed: onTap,
      icon: Icon(icon, size: 20),
      label: Text(label, style: const TextStyle(fontWeight: FontWeight.bold)),
    );
  }
}

class _LangOptionButton extends StatelessWidget {
  final String label;
  final bool isSelected;
  final VoidCallback onTap;

  const _LangOptionButton({
    required this.label,
    required this.isSelected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(vertical: 12),
        alignment: Alignment.center,
        decoration: BoxDecoration(
          color: isSelected
              ? Theme.of(context).colorScheme.primary
              : Colors.transparent,
          borderRadius: BorderRadius.circular(8),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: isSelected ? Colors.white : Colors.grey[400],
            fontWeight: FontWeight.bold,
            fontSize: 14,
          ),
        ),
      ),
    );
  }
}

class OcrBoxPainter extends CustomPainter {
  final List<OcrResult> results;
  final int imageWidth;
  final int imageHeight;
  final Color primaryColor;
  final Color secondaryColor;

  OcrBoxPainter({
    required this.results,
    required this.imageWidth,
    required this.imageHeight,
    required this.primaryColor,
    required this.secondaryColor,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (imageWidth == 0 || imageHeight == 0) return;

    final double scale = min(
      size.width / imageWidth,
      size.height / imageHeight,
    );
    final double renderedW = imageWidth * scale;
    final double renderedH = imageHeight * scale;
    final double dx = (size.width - renderedW) / 2;
    final double dy = (size.height - renderedH) / 2;

    final Paint borderPaint = Paint()
      ..color = primaryColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    final Paint fillPaint = Paint()
      ..color = primaryColor.withOpacity(0.12)
      ..style = PaintingStyle.fill;

    for (final result in results) {
      if (result.box.length < 4) continue;

      final Path path = Path();

      final List<Offset> offsets = result.box.map((p) {
        return Offset(dx + p.x * scale, dy + p.y * scale);
      }).toList();

      path.moveTo(offsets[0].dx, offsets[0].dy);
      path.lineTo(offsets[1].dx, offsets[1].dy);
      path.lineTo(offsets[2].dx, offsets[2].dy);
      path.lineTo(offsets[3].dx, offsets[3].dy);
      path.close();

      canvas.drawPath(path, fillPaint);
      canvas.drawPath(path, borderPaint);

      final TextPainter tp = TextPainter(
        text: TextSpan(
          text: result.text,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 10,
            fontWeight: FontWeight.bold,
            backgroundColor: Colors.black87,
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      tp.layout();
      tp.paint(canvas, Offset(offsets[0].dx, max(0.0, offsets[0].dy - 12)));
    }
  }

  @override
  bool shouldRepaint(covariant OcrBoxPainter oldDelegate) {
    return oldDelegate.results != results ||
        oldDelegate.imageWidth != imageWidth ||
        oldDelegate.imageHeight != imageHeight;
  }
}

class _AccOptionButton extends StatelessWidget {
  final String label;
  final bool isSelected;
  final VoidCallback onTap;

  const _AccOptionButton({
    required this.label,
    required this.isSelected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(vertical: 8),
        alignment: Alignment.center,
        decoration: BoxDecoration(
          color: isSelected
              ? Theme.of(context).colorScheme.secondary
              : Colors.transparent,
          borderRadius: BorderRadius.circular(8),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: isSelected ? Colors.white : Colors.grey[400],
            fontWeight: FontWeight.bold,
            fontSize: 12,
          ),
        ),
      ),
    );
  }
}

class _TabButton extends StatelessWidget {
  final String label;
  final bool isSelected;
  final VoidCallback onTap;

  const _TabButton({
    required this.label,
    required this.isSelected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: isSelected
              ? Theme.of(context).colorScheme.primary.withOpacity(0.15)
              : Colors.transparent,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: isSelected
                ? Theme.of(context).colorScheme.primary
                : Colors.transparent,
            width: 1,
          ),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: isSelected ? Colors.white : Colors.grey[400],
            fontWeight: FontWeight.bold,
            fontSize: 13,
          ),
        ),
      ),
    );
  }
}
