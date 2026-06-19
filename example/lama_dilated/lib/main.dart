import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'lama_dilated_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
    ),
  );
  runApp(const LamaInpaintApp());
}

class LamaInpaintApp extends StatelessWidget {
  const LamaInpaintApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'LaMa-Dilated Inpainting',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF0A0A0E),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF8B5CF6), // Violet
          secondary: Color(0xFFEC4899), // Pink
          surface: Color(0xFF13131A),
          error: Color(0xFFEF4444),
        ),
        cardTheme: const CardThemeData(
          color: Color(0xFF13131A),
          elevation: 4,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.all(Radius.circular(16)),
          ),
        ),
      ),
      home: const InpaintHomeScreen(),
    );
  }
}

class InpaintHomeScreen extends StatefulWidget {
  const InpaintHomeScreen({super.key});

  @override
  State<InpaintHomeScreen> createState() => _InpaintHomeScreenState();
}

class _InpaintHomeScreenState extends State<InpaintHomeScreen> {
  final LamaDilatedService _inpaintService = LamaDilatedService();
  final ImagePicker _picker = ImagePicker();

  Uint8List? _originalImageBytes;
  int _nativeWidth = 0;
  int _nativeHeight = 0;

  // Drawing paths in native coordinates
  List<List<Offset>> _nativePaths = [];
  List<Offset> _currentNativePath = [];

  Uint8List? _inpaintedImageBytes;
  bool _isLoading = false;
  bool _isProcessing = false;
  String? _statusMessage;

  // Latency metrics
  double _preprocessTimeMs = 0.0;
  double _inferenceTimeMs = 0.0;
  double _postprocessTimeMs = 0.0;

  HardwareAccelerator _selectedAccelerator = HardwareAccelerator.cpu;
  bool _highFidelityComposite = true;
  double _brushSize = 25.0;

  @override
  void initState() {
    super.initState();
    _initModel();
  }

  Future<void> _initModel() async {
    setState(() {
      _isLoading = true;
      _statusMessage = 'Initializing LaMa Model...';
    });

    try {
      await _inpaintService.loadModel(accelerator: _selectedAccelerator);
      setState(() {
        _isLoading = false;
        _statusMessage = 'Model loaded';
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _statusMessage = 'Error loading model: $e';
      });
      _showErrorDialog('Failed to load model. Details: $e');
    }
  }

  Future<void> _changeAccelerator(HardwareAccelerator? accelerator) async {
    if (accelerator == null || accelerator == _selectedAccelerator) return;
    setState(() {
      _selectedAccelerator = accelerator;
    });
    await _initModel();
  }

  // Generates a black-and-white mask PNG bytes of native size
  Future<Uint8List> _generateMaskBytes() async {
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);

    // 1. Fill background with black
    final bgPaint = Paint()..color = Colors.black;
    canvas.drawRect(
      Rect.fromLTWH(0, 0, _nativeWidth.toDouble(), _nativeHeight.toDouble()),
      bgPaint,
    );

    // 2. Draw mask paths with white color
    final pathPaint = Paint()
      ..color = Colors.white
      ..strokeWidth = _brushSize // In native space
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    for (final path in _nativePaths) {
      if (path.isEmpty) continue;
      final uiPath = ui.Path();
      uiPath.moveTo(path.first.dx, path.first.dy);
      for (int i = 1; i < path.length; i++) {
        uiPath.lineTo(path[i].dx, path[i].dy);
      }
      canvas.drawPath(uiPath, pathPaint);
    }

    final picture = recorder.endRecording();
    final image = await picture.toImage(_nativeWidth, _nativeHeight);
    final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
    return byteData!.buffer.asUint8List();
  }

  Future<void> _runInpainting() async {
    if (_originalImageBytes == null || _nativePaths.isEmpty) return;

    setState(() {
      _isProcessing = true;
      _statusMessage = 'Generating mask & running LaMa model...';
    });

    try {
      final maskBytes = await _generateMaskBytes();
      
      setState(() {
        _statusMessage = 'Running model inference...';
      });

      final inpaintedBytes = await _inpaintService.inpaint(
        _originalImageBytes!,
        maskBytes,
        highFidelityComposite: _highFidelityComposite,
        onStatsUpdated: (pre, inf, post) {
          setState(() {
            _preprocessTimeMs = pre;
            _inferenceTimeMs = inf;
            _postprocessTimeMs = post;
          });
        },
      );

      setState(() {
        _inpaintedImageBytes = inpaintedBytes;
        _isProcessing = false;
        _statusMessage = 'Inpainting complete';
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _statusMessage = 'Error during inpainting: $e';
      });
      _showErrorDialog('Inpainting failed. Details: $e');
    }
  }

  Future<void> _loadImage(Uint8List bytes) async {
    setState(() {
      _isProcessing = true;
      _statusMessage = 'Analyzing image...';
    });

    try {
      // Decode dimensions
      final codec = await ui.instantiateImageCodec(bytes);
      final frameInfo = await codec.getNextFrame();
      
      setState(() {
        _originalImageBytes = bytes;
        _nativeWidth = frameInfo.image.width;
        _nativeHeight = frameInfo.image.height;
        _nativePaths = [];
        _currentNativePath = [];
        _inpaintedImageBytes = null;
        _isProcessing = false;
        _statusMessage = 'Image loaded';
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _statusMessage = 'Failed to load image';
      });
      _showErrorDialog('Error reading image metadata: $e');
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? image = await _picker.pickImage(source: source);
      if (image != null) {
        final bytes = await image.readAsBytes();
        await _loadImage(bytes);
      }
    } catch (e) {
      _showErrorDialog('Error picking image: $e');
    }
  }

  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Row(
          children: [
            Icon(Icons.error_outline, color: Colors.redAccent),
            SizedBox(width: 8),
            Text('Error'),
          ],
        ),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final double totalTimeMs = _preprocessTimeMs + _inferenceTimeMs + _postprocessTimeMs;

    return Scaffold(
      appBar: AppBar(
        title: const Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'LaMa-Dilated AI',
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
            ),
            Text(
              'On-Device Image Inpainting',
              style: TextStyle(fontSize: 11, color: Colors.grey),
            ),
          ],
        ),
        backgroundColor: const Color(0xFF13131A),
        elevation: 0,
        actions: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            alignment: Alignment.center,
            child: DropdownButtonHideUnderline(
              child: DropdownButton<HardwareAccelerator>(
                value: _selectedAccelerator,
                onChanged: _isProcessing || _isLoading ? null : _changeAccelerator,
                dropdownColor: const Color(0xFF13131A),
                items: HardwareAccelerator.values.map((acc) {
                  return DropdownMenuItem(
                    value: acc,
                    child: Text(
                      acc.name.toUpperCase(),
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        color: _selectedAccelerator == acc ? const Color(0xFF8B5CF6) : Colors.white70,
                      ),
                    ),
                  );
                }).toList(),
              ),
            ),
          ),
        ],
      ),
      body: _isLoading
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const CircularProgressIndicator(color: Color(0xFF8B5CF6)),
                  const SizedBox(height: 24),
                  Text(_statusMessage ?? 'Loading...', style: const TextStyle(color: Colors.grey)),
                ],
              ),
            )
          : Column(
              children: [
                // Top Status Bar
                if (_statusMessage != null)
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
                    color: const Color(0xFF8B5CF6).withValues(alpha: 0.1),
                    child: Text(
                      _statusMessage!,
                      textAlign: TextAlign.center,
                      style: const TextStyle(fontSize: 12, color: Color(0xFF8B5CF6), fontWeight: FontWeight.w600),
                    ),
                  ),

                // Main Workspace
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: _originalImageBytes == null
                        ? _buildEmptyState()
                        : _buildWorkspace(),
                  ),
                ),

                // Bottom Panel
                if (_originalImageBytes != null) _buildBottomControls(totalTimeMs),
              ],
            ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            height: 120,
            width: 120,
            decoration: BoxDecoration(
              color: const Color(0xFF181824),
              borderRadius: BorderRadius.circular(30),
              border: Border.all(color: Colors.grey.withValues(alpha: 0.1)),
            ),
            child: const Icon(
              Icons.image_search_rounded,
              size: 54,
              color: Color(0xFF8B5CF6),
            ),
          ),
          const SizedBox(height: 24),
          const Text(
            'Select an Image to Edit',
            style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          const Text(
            'Import a photo and paint over the areas\nyou want to erase/inpaint on-device.',
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.grey, fontSize: 14),
          ),
          const SizedBox(height: 32),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ElevatedButton.icon(
                onPressed: () => _pickImage(ImageSource.gallery),
                icon: const Icon(Icons.photo_library),
                label: const Text('Gallery'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF8B5CF6),
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                ),
              ),
              const SizedBox(width: 16),
              OutlinedButton.icon(
                onPressed: () => _pickImage(ImageSource.camera),
                icon: const Icon(Icons.camera_alt),
                label: const Text('Camera'),
                style: OutlinedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
                  side: BorderSide(color: Colors.grey.withValues(alpha: 0.3)),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildWorkspace() {
    if (_inpaintedImageBytes != null) {
      // Comparison View
      return Column(
        children: [
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: Stack(
                children: [
                  Positioned.fill(
                    child: Image.memory(
                      _inpaintedImageBytes!,
                      fit: BoxFit.contain,
                    ),
                  ),
                  Positioned(
                    top: 12,
                    right: 12,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                      decoration: BoxDecoration(
                        color: Colors.black.withValues(alpha: 0.7),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: const Text(
                        'Inpainted Result',
                        style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Color(0xFFEC4899)),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          // View Original / Compare buttons
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              TextButton.icon(
                onPressed: () {
                  setState(() {
                    _inpaintedImageBytes = null;
                  });
                },
                icon: const Icon(Icons.edit_road),
                label: const Text('Modify Mask'),
                style: TextButton.styleFrom(foregroundColor: const Color(0xFF8B5CF6)),
              ),
              ElevatedButton.icon(
                onPressed: () {
                  setState(() {
                    _nativePaths = [];
                    _currentNativePath = [];
                    _inpaintedImageBytes = null;
                  });
                },
                icon: const Icon(Icons.refresh),
                label: const Text('Reset Image'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF181824),
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                ),
              ),
            ],
          ),
        ],
      );
    }

    // Editing mode: Draw mask on top of original image
    return Column(
      children: [
        Expanded(
          child: Container(
            decoration: BoxDecoration(
              color: const Color(0xFF13131A),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: Colors.white12),
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: InteractiveViewer(
                maxScale: 4.0,
                minScale: 1.0,
                child: LayoutBuilder(
                  builder: (context, constraints) {
                    // Calculate displays coordinates
                    final double imgAspect = _nativeWidth / _nativeHeight;
                    final double constraintAspect = constraints.maxWidth / constraints.maxHeight;

                    double displayW, displayH;
                    if (imgAspect > constraintAspect) {
                      displayW = constraints.maxWidth;
                      displayH = constraints.maxWidth / imgAspect;
                    } else {
                      displayH = constraints.maxHeight;
                      displayW = constraints.maxHeight * imgAspect;
                    }

                    final double offsetX = (constraints.maxWidth - displayW) / 2;
                    final double offsetY = (constraints.maxHeight - displayH) / 2;
                    final double scale = displayW / _nativeWidth;

                    return Stack(
                      children: [
                        // Background image
                        Positioned(
                          left: offsetX,
                          top: offsetY,
                          width: displayW,
                          height: displayH,
                          child: Image.memory(
                            _originalImageBytes!,
                            fit: BoxFit.fill,
                          ),
                        ),
                        // Transparent drawing canvas
                        Positioned.fill(
                          child: GestureDetector(
                            onPanStart: (details) {
                              if (_isProcessing) return;
                              final localPos = details.localPosition;
                              final imageX = localPos.dx - offsetX;
                              final imageY = localPos.dy - offsetY;
                              if (imageX >= 0 && imageX <= displayW && imageY >= 0 && imageY <= displayH) {
                                final nativePoint = Offset(imageX / scale, imageY / scale);
                                setState(() {
                                  _currentNativePath = [nativePoint];
                                  _nativePaths.add(_currentNativePath);
                                });
                              }
                            },
                            onPanUpdate: (details) {
                              if (_isProcessing) return;
                              final localPos = details.localPosition;
                              final imageX = localPos.dx - offsetX;
                              final imageY = localPos.dy - offsetY;
                              if (imageX >= 0 && imageX <= displayW && imageY >= 0 && imageY <= displayH) {
                                final nativePoint = Offset(imageX / scale, imageY / scale);
                                setState(() {
                                  _currentNativePath.add(nativePoint);
                                });
                              }
                            },
                            onPanEnd: (_) {
                              _currentNativePath = [];
                            },
                            child: CustomPaint(
                              painter: MaskPainter(
                                paths: _nativePaths,
                                scale: scale,
                                offsetX: offsetX,
                                offsetY: offsetY,
                                brushSize: _brushSize,
                              ),
                            ),
                          ),
                        ),
                      ],
                    );
                  },
                ),
              ),
            ),
          ),
        ),
        const SizedBox(height: 12),
        // Brush size slider and helper buttons
        Row(
          children: [
            const Icon(Icons.brush, size: 18, color: Colors.grey),
            const SizedBox(width: 8),
            Expanded(
              child: SliderTheme(
                data: SliderTheme.of(context).copyWith(
                  activeTrackColor: const Color(0xFF8B5CF6),
                  inactiveTrackColor: Colors.white12,
                  thumbColor: Colors.white,
                  overlayColor: const Color(0xFF8B5CF6).withAlpha(32),
                ),
                child: Slider(
                  value: _brushSize,
                  min: 5.0,
                  max: 100.0,
                  onChanged: _isProcessing
                      ? null
                      : (val) {
                          setState(() {
                            _brushSize = val;
                          });
                        },
                ),
              ),
            ),
            Text(
              '${_brushSize.round()} px',
              style: const TextStyle(fontSize: 12, color: Colors.grey, fontWeight: FontWeight.bold),
            ),
            const SizedBox(width: 16),
            IconButton(
              icon: const Icon(Icons.undo, color: Colors.white70),
              tooltip: 'Undo stroke',
              onPressed: _isProcessing || _nativePaths.isEmpty
                  ? null
                  : () {
                      setState(() {
                        _nativePaths.removeLast();
                      });
                    },
            ),
            IconButton(
              icon: const Icon(Icons.delete_outline, color: Colors.redAccent),
              tooltip: 'Clear mask',
              onPressed: _isProcessing || _nativePaths.isEmpty
                  ? null
                  : () {
                      setState(() {
                        _nativePaths.clear();
                      });
                    },
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildBottomControls(double totalTimeMs) {
    return Container(
      padding: const EdgeInsets.only(left: 20, right: 20, bottom: 32, top: 16),
      decoration: const BoxDecoration(
        color: Color(0xFF13131A),
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      child: Column(
        children: [
          // HF Composite Toggle & Stats toggle
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  Switch(
                    value: _highFidelityComposite,
                    activeThumbColor: const Color(0xFFEC4899),
                    onChanged: _isProcessing
                        ? null
                        : (val) {
                            setState(() {
                              _highFidelityComposite = val;
                            });
                          },
                  ),
                  const Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'High-Fidelity Composite',
                        style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
                      ),
                      Text(
                        'Keep original background resolution',
                        style: TextStyle(fontSize: 10, color: Colors.grey),
                      ),
                    ],
                  ),
                ],
              ),
              if (totalTimeMs > 0.0)
                _buildMetricsMiniWidget(totalTimeMs)
            ],
          ),
          const SizedBox(height: 16),
          // Action Buttons
          Row(
            children: [
              IconButton(
                icon: const Icon(Icons.add_photo_alternate_outlined),
                onPressed: _isProcessing
                    ? null
                    : () {
                        setState(() {
                          _originalImageBytes = null;
                          _inpaintedImageBytes = null;
                          _nativePaths = [];
                        });
                      },
                tooltip: 'Select new photo',
                style: IconButton.styleFrom(
                  backgroundColor: Colors.white10,
                  padding: const EdgeInsets.all(16),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _isProcessing
                    ? Container(
                        height: 52,
                        alignment: Alignment.center,
                        decoration: BoxDecoration(
                          color: const Color(0xFF8B5CF6).withValues(alpha: 0.3),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: const Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            SizedBox(
                              width: 18,
                              height: 18,
                              child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                            ),
                            SizedBox(width: 12),
                            Text('Processing Inpainting...', style: TextStyle(fontWeight: FontWeight.bold)),
                          ],
                        ),
                      )
                    : ElevatedButton.icon(
                        onPressed: _nativePaths.isEmpty ? null : _runInpainting,
                        icon: const Icon(Icons.auto_awesome),
                        label: const Text(
                          'Run AI Inpainting',
                          style: TextStyle(fontWeight: FontWeight.bold),
                        ),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF8B5CF6),
                          disabledBackgroundColor: Colors.white10,
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                        ),
                      ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildMetricsMiniWidget(double totalTimeMs) {
    return InkWell(
      onTap: () {
        showModalBottomSheet(
          context: context,
          backgroundColor: const Color(0xFF13131A),
          shape: const RoundedRectangleBorder(
            borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
          ),
          builder: (context) => Padding(
            padding: const EdgeInsets.all(24.0),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Latency Diagnostics',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white),
                ),
                const SizedBox(height: 4),
                const Text(
                  'On-device performance breakdown (ms)',
                  style: TextStyle(fontSize: 12, color: Colors.grey),
                ),
                const Divider(height: 32, color: Colors.white12),
                _buildMetricRow('Preprocessing', _preprocessTimeMs, const Color(0xFF8B5CF6)),
                const SizedBox(height: 12),
                _buildMetricRow('TFLite Inference', _inferenceTimeMs, const Color(0xFFEC4899)),
                const SizedBox(height: 12),
                _buildMetricRow('Postprocessing', _postprocessTimeMs, Colors.cyan),
                const Divider(height: 32, color: Colors.white12),
                _buildMetricRow('Total Pipeline Time', totalTimeMs, Colors.greenAccent, isTotal: true),
                const SizedBox(height: 24),
              ],
            ),
          ),
        );
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: Colors.white10,
          borderRadius: BorderRadius.circular(8),
        ),
        child: Row(
          children: [
            const Icon(Icons.speed, size: 14, color: Colors.greenAccent),
            const SizedBox(width: 6),
            Text(
              '${totalTimeMs.toStringAsFixed(0)} ms',
              style: const TextStyle(fontSize: 12, color: Colors.greenAccent, fontWeight: FontWeight.bold),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMetricRow(String name, double val, Color dotColor, {bool isTotal = false}) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Row(
          children: [
            Container(
              width: 8,
              height: 8,
              decoration: BoxDecoration(color: dotColor, shape: BoxShape.circle),
            ),
            const SizedBox(width: 8),
            Text(
              name,
              style: TextStyle(
                fontWeight: isTotal ? FontWeight.bold : FontWeight.normal,
                fontSize: isTotal ? 15 : 14,
                color: isTotal ? Colors.white : Colors.white70,
              ),
            ),
          ],
        ),
        Text(
          '${val.toStringAsFixed(1)} ms',
          style: TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: isTotal ? 15 : 14,
            color: isTotal ? Colors.greenAccent : Colors.white,
          ),
        ),
      ],
    );
  }
}

class MaskPainter extends CustomPainter {
  final List<List<Offset>> paths;
  final double scale;
  final double offsetX;
  final double offsetY;
  final double brushSize;

  MaskPainter({
    required this.paths,
    required this.scale,
    required this.offsetX,
    required this.offsetY,
    required this.brushSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    canvas.save();
    
    // Scale and offset canvas so that paths draw correctly on screen
    final paint = Paint()
      ..color = Colors.redAccent.withValues(alpha: 0.5)
      ..style = PaintingStyle.stroke
      ..strokeWidth = brushSize * scale
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    for (final path in paths) {
      if (path.isEmpty) continue;
      final uiPath = Path();
      // First point
      final displayStart = Offset(path.first.dx * scale + offsetX, path.first.dy * scale + offsetY);
      uiPath.moveTo(displayStart.dx, displayStart.dy);

      for (int i = 1; i < path.length; i++) {
        final displayPoint = Offset(path[i].dx * scale + offsetX, path[i].dy * scale + offsetY);
        uiPath.lineTo(displayPoint.dx, displayPoint.dy);
      }
      canvas.drawPath(uiPath, paint);
    }
    
    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant MaskPainter oldDelegate) {
    return oldDelegate.paths != paths ||
        oldDelegate.scale != scale ||
        oldDelegate.brushSize != brushSize ||
        oldDelegate.offsetX != offsetX ||
        oldDelegate.offsetY != offsetY;
  }
}
