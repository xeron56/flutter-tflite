import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'trocr_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'TrOCR Lite',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF0F0F16),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF8B5CF6), // Purple accent
          secondary: Color(0xFF10B981), // Emerald accent
          surface: Color(0xFF181825),
          error: Color(0xFFEF4444),
        ),
      ),
      home: const TrocrHomeScreen(),
    );
  }
}

class TrocrHomeScreen extends StatefulWidget {
  const TrocrHomeScreen({super.key});

  @override
  State<TrocrHomeScreen> createState() => _TrocrHomeScreenState();
}

class _TrocrHomeScreenState extends State<TrocrHomeScreen> {
  final TrocrService _trocrService = TrocrService();
  final ImagePicker _picker = ImagePicker();
  final TextEditingController _textController = TextEditingController();

  File? _imageFile;
  img.Image? _decodedImage;
  bool _isLoading = false;
  HardwareAccelerator _selectedAccelerator = HardwareAccelerator.cpu;
  String? _statusText;

  @override
  void initState() {
    super.initState();
    _initTrocrService();
  }

  Future<void> _initTrocrService() async {
    setState(() {
      _isLoading = true;
      _statusText = 'Loading TrOCR encoder & decoder models...';
    });
    try {
      await _trocrService.loadModels(accelerator: _selectedAccelerator);
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
        'Failed to load TrOCR models. Make sure encoder.tflite, decoder.tflite, and vocab.json are in your assets. Details: $e',
      );
    }
  }

  Future<void> _changeAccelerator(HardwareAccelerator newAcc) async {
    if (newAcc == _selectedAccelerator) return;
    setState(() {
      _isLoading = true;
      _selectedAccelerator = newAcc;
      _statusText = 'Reloading models with ${newAcc.name.toUpperCase()}...';
      _textController.clear();
    });

    try {
      await _trocrService.loadModels(accelerator: newAcc);
      
      final bool encoderFellback = newAcc != HardwareAccelerator.cpu &&
          _trocrService.encoderAccelerator == HardwareAccelerator.cpu;
      final bool decoderFellback = newAcc != HardwareAccelerator.cpu &&
          _trocrService.decoderAccelerator == HardwareAccelerator.cpu;

      String statusMsg = 'Models loaded';
      if (newAcc != HardwareAccelerator.cpu) {
        if (encoderFellback && decoderFellback) {
          statusMsg = 'Accelerator failed: both models fell back to CPU';
          _selectedAccelerator = HardwareAccelerator.cpu;
          _showFallbackNotification(
            'Requested acceleration was not supported on this device. Both models fell back to CPU.',
          );
        } else if (encoderFellback) {
          statusMsg = 'Encoder fell back to CPU';
          _showFallbackNotification(
            'TrOCR Encoder model fell back to CPU. Decoder is running on ${newAcc.name.toUpperCase()}.',
          );
        } else if (decoderFellback) {
          statusMsg = 'Decoder fell back to CPU';
          _showFallbackNotification(
            'TrOCR Decoder model fell back to CPU. Encoder is running on ${newAcc.name.toUpperCase()}.',
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
      _initTrocrService();
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
        _textController.clear();
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
      _statusText = 'Running TrOCR encoder & decoder...';
    });

    try {
      final result = await _trocrService.runOcr(_decodedImage!);
      setState(() {
        _textController.text = result;
        _isLoading = false;
        _statusText = 'Processed successfully';
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _statusText = 'OCR failed';
      });
      _showErrorDialog('TrOCR Inference error: $e');
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

  @override
  void dispose() {
    _trocrService.dispose();
    _textController.dispose();
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
              'TrOCR Lite',
              style: TextStyle(
                fontWeight: FontWeight.w800,
                fontSize: 20,
                letterSpacing: -0.5,
              ),
            ),
            Text(
              'On-Device Transformer OCR Pipeline',
              style: TextStyle(fontSize: 11, color: Colors.grey),
            ),
          ],
        ),
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(
              horizontal: 16.0,
              vertical: 8.0,
            ),
            child: Container(
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
          ),
          Expanded(
            child: _imageFile == null
                ? _buildEmptyState()
                : _buildImageWorkspace(),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            decoration: BoxDecoration(
              color: const Color(0xFF13131F),
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
                Icons.text_fields_rounded,
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
              'Take a photo or pick an image of a line of text to run on-device TrOCR.',
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
                    child: Image.file(_imageFile!, fit: BoxFit.contain),
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
                                  'Backend: ${_selectedAccelerator.name.toUpperCase()}',
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
                              'Encoder: ${_trocrService.encoderTimeMs.toStringAsFixed(1)} ms (${_trocrService.encoderAccelerator.name.toUpperCase()})',
                              style: const TextStyle(
                                fontSize: 10,
                                color: Colors.white60,
                              ),
                            ),
                            Text(
                              'Decoder Step: ${_trocrService.decoderTimeMs.toStringAsFixed(1)} ms (${_trocrService.decoderAccelerator.name.toUpperCase()})',
                              style: const TextStyle(
                                fontSize: 10,
                                color: Colors.white60,
                              ),
                            ),
                            Text(
                              'Total Dec: ${(_trocrService.decoderTimeMs * _trocrService.decodedTokenCount).toStringAsFixed(1)} ms (${_trocrService.decodedTokenCount} tokens)',
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
                      const Text(
                        'Recognized Text',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
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
                  child: Padding(
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
                              controller: _textController,
                              maxLines: null,
                              style: const TextStyle(
                                fontSize: 14,
                                height: 1.5,
                              ),
                              decoration: const InputDecoration(
                                contentPadding: EdgeInsets.all(12),
                                border: InputBorder.none,
                                hintText: 'Decoded text will appear here...',
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
                                    if (_textController.text.isNotEmpty) {
                                      Clipboard.setData(
                                        ClipboardData(text: _textController.text),
                                      );
                                      ScaffoldMessenger.of(context).showSnackBar(
                                        const SnackBar(
                                          content: Text('Copied text to clipboard'),
                                          behavior: SnackBarBehavior.floating,
                                          duration: Duration(seconds: 1),
                                        ),
                                      );
                                    }
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
                  ),
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
              ? Theme.of(context).colorScheme.primary
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
