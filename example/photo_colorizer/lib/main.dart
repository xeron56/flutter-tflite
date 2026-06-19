import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'ddcolor_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
    ),
  );
  runApp(const DdColorApp());
}

class DdColorApp extends StatelessWidget {
  const DdColorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'DDColor AI Colorizer',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF0F0F16),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF6366F1), // Indigo
          secondary: Color(0xFF10B981), // Emerald
          surface: Color(0xFF181824),
          error: Color(0xFFEF4444),
        ),
        cardTheme: const CardThemeData(
          color: Color(0xFF181824),
          elevation: 4,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.all(Radius.circular(16)),
          ),
        ),
      ),
      home: const ColorizerHomeScreen(),
    );
  }
}

class ColorizerHomeScreen extends StatefulWidget {
  const ColorizerHomeScreen({super.key});

  @override
  State<ColorizerHomeScreen> createState() => _ColorizerHomeScreenState();
}

class _ColorizerHomeScreenState extends State<ColorizerHomeScreen> {
  final DdColorService _colorService = DdColorService();
  final ImagePicker _picker = ImagePicker();

  Uint8List? _originalImageBytes;
  Uint8List? _grayscaleImageBytes; // Local grayscale conversion of original image
  Uint8List? _colorizedImageBytes;

  bool _isLoading = false;
  bool _isProcessing = false;
  String? _statusMessage;

  // Latency metrics
  double _preprocessTimeMs = 0.0;
  double _inferenceTimeMs = 0.0;
  double _postprocessTimeMs = 0.0;

  HardwareAccelerator _selectedAccelerator = HardwareAccelerator.cpu;

  @override
  void initState() {
    super.initState();
    _initModel();
  }

  Future<void> _initModel() async {
    setState(() {
      _isLoading = true;
      _statusMessage = 'Initializing DDColor Model...';
    });

    try {
      await _colorService.loadModel(accelerator: _selectedAccelerator);
      setState(() {
        _isLoading = false;
        _statusMessage = 'Model loaded successfully';
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
    if (_originalImageBytes != null) {
      _runColorization();
    }
  }

  Future<void> _runColorization() async {
    if (_originalImageBytes == null) return;

    setState(() {
      _isProcessing = true;
      _statusMessage = 'Processing image...';
    });

    try {
      final colorizedBytes = await _colorService.colorize(
        _originalImageBytes!,
        onStatsUpdated: (pre, inf, post) {
          setState(() {
            _preprocessTimeMs = pre;
            _inferenceTimeMs = inf;
            _postprocessTimeMs = post;
          });
        },
      );

      setState(() {
        _colorizedImageBytes = colorizedBytes;
        _isProcessing = false;
        _statusMessage = 'Colorization complete';
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _statusMessage = 'Error colorizing: $e';
      });
      _showErrorDialog('Colorization failed. Details: $e');
    }
  }

  Future<void> _loadImage(Uint8List bytes) async {
    setState(() {
      _originalImageBytes = bytes;
      _colorizedImageBytes = null;
      _statusMessage = 'Creating grayscale preview...';
    });

    // Generate local grayscale preview (so user sees grayscale on the left side of compare slider)
    final grayBytes = await _convertToGrayscale(bytes);

    setState(() {
      _grayscaleImageBytes = grayBytes;
    });

    _runColorization();
  }

  Future<Uint8List> _convertToGrayscale(Uint8List bytes) async {
    return await Stream.fromFuture(Future(() {
      final image = img.decodeImage(bytes);
      if (image == null) return bytes;
      final grayscaleImage = img.grayscale(image);
      return Uint8List.fromList(img.encodeJpg(grayscaleImage));
    })).first;
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? file = await _picker.pickImage(source: source);
      if (file == null) return;
      final bytes = await file.readAsBytes();
      await _loadImage(bytes);
    } catch (e) {
      _showErrorDialog('Failed to pick image: $e');
    }
  }

  Future<void> _loadSampleImage() async {
    setState(() {
      _isProcessing = true;
      _statusMessage = 'Generating sample grayscale image...';
    });

    // Programmatically generate a sphere/gradient pattern grayscale image for testing
    final sampleBytes = await Stream.fromFuture(Future(() {
      final imgImage = img.Image(width: 512, height: 512);
      // Draw background gray gradient
      for (int y = 0; y < 512; y++) {
        for (int x = 0; x < 512; x++) {
          final double dx = (x - 256) / 256.0;
          final double dy = (y - 256) / 256.0;
          final double dist = math.sqrt(dx * dx + dy * dy);
          final int val = ((1.0 - dist.clamp(0.0, 1.0)) * 180 + 40).round();
          imgImage.setPixelRgb(x, y, val, val, val);
        }
      }
      // Add two spheres with different grey shades
      img.fillCircle(imgImage, x: 180, y: 220, radius: 80, color: img.ColorRgb8(190, 190, 190));
      img.fillCircle(imgImage, x: 340, y: 320, radius: 100, color: img.ColorRgb8(80, 80, 80));
      img.fillCircle(imgImage, x: 360, y: 150, radius: 50, color: img.ColorRgb8(230, 230, 230));

      return Uint8List.fromList(img.encodeJpg(imgImage));
    })).first;

    await _loadImage(sampleBytes);
  }

  Future<void> _saveImage() async {
    if (_colorizedImageBytes == null) return;

    try {
      String savePath = '';
      if (Platform.isMacOS) {
        final dir = Directory('/Users/mdshahidulislam/Downloads');
        if (dir.existsSync()) {
          savePath = '${dir.path}/colorized_${DateTime.now().millisecondsSinceEpoch}.jpg';
        }
      }

      if (savePath.isEmpty) {
        final tempDir = Directory.systemTemp;
        savePath = '${tempDir.path}/colorized_${DateTime.now().millisecondsSinceEpoch}.jpg';
      }

      final file = File(savePath);
      await file.writeAsBytes(_colorizedImageBytes!);

      if (!mounted) return;

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Row(
            children: [
              const Icon(Icons.check_circle_outline, color: Colors.greenAccent),
              const SizedBox(width: 8),
              Expanded(child: Text('Image saved successfully to:\n$savePath')),
            ],
          ),
          behavior: SnackBarBehavior.floating,
          duration: const Duration(seconds: 5),
        ),
      );
    } catch (e) {
      _showErrorDialog('Failed to save image: $e');
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
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final double totalTimeMs = _preprocessTimeMs + _inferenceTimeMs + _postprocessTimeMs;
    final bool hasResult = _colorizedImageBytes != null && _grayscaleImageBytes != null;

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'DDColor AI Colorizer',
          style: TextStyle(fontWeight: FontWeight.bold, letterSpacing: 0.5),
        ),
        centerTitle: true,
        backgroundColor: const Color(0xFF181824),
        elevation: 2,
        actions: [
          if (hasResult)
            IconButton(
              icon: const Icon(Icons.save_alt),
              tooltip: 'Save Colorized Image',
              onPressed: _isProcessing ? null : _saveImage,
            ),
        ],
      ),
      body: _isLoading
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const CircularProgressIndicator(),
                  const SizedBox(height: 16),
                  Text(
                    _statusMessage ?? 'Loading...',
                    style: const TextStyle(color: Colors.grey),
                  ),
                ],
              ),
            )
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Hardware Accelerator Picker Card
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Row(
                            children: [
                              Icon(Icons.bolt, color: Color(0xFF6366F1)),
                              SizedBox(width: 8),
                              Text(
                                'Acceleration:',
                                style: TextStyle(fontWeight: FontWeight.w600),
                              ),
                            ],
                          ),
                          DropdownButton<HardwareAccelerator>(
                            value: _selectedAccelerator,
                            underline: const SizedBox(),
                            items: HardwareAccelerator.values.map((acc) {
                              return DropdownMenuItem(
                                value: acc,
                                child: Text(acc.name.toUpperCase()),
                              );
                            }).toList(),
                            onChanged: _isProcessing ? null : _changeAccelerator,
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),

                  // Image Display Area
                  AspectRatio(
                    aspectRatio: 1,
                    child: Container(
                      decoration: BoxDecoration(
                        color: const Color(0xFF181824),
                        borderRadius: BorderRadius.circular(24),
                        border: Border.all(color: Colors.white10),
                      ),
                      clipBehavior: Clip.antiAlias,
                      child: _isProcessing && !hasResult
                          ? Center(
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  const CircularProgressIndicator(),
                                  const SizedBox(height: 16),
                                  Text(
                                    _statusMessage ?? 'Processing...',
                                    style: const TextStyle(color: Colors.grey),
                                  ),
                                ],
                              ),
                            )
                          : hasResult
                              ? Stack(
                                  children: [
                                    ImageCompareSlider(
                                      beforeImage: Image.memory(
                                        _grayscaleImageBytes!,
                                        fit: BoxFit.cover,
                                      ),
                                      afterImage: Image.memory(
                                        _colorizedImageBytes!,
                                        fit: BoxFit.cover,
                                      ),
                                    ),
                                    if (_isProcessing)
                                      Container(
                                        color: Colors.black38,
                                        child: const Center(
                                          child: CircularProgressIndicator(),
                                        ),
                                      ),
                                  ],
                                )
                              : Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Icon(
                                      Icons.photo_library_outlined,
                                      size: 64,
                                      color: Colors.grey.shade700,
                                    ),
                                    const SizedBox(height: 16),
                                    const Text(
                                      'No photo colorized yet',
                                      style: TextStyle(
                                        color: Colors.grey,
                                        fontWeight: FontWeight.w500,
                                      ),
                                    ),
                                    const SizedBox(height: 8),
                                    const Text(
                                      'Pick a photo or load a sample to start',
                                      style: TextStyle(
                                        color: Colors.grey,
                                        fontSize: 12,
                                      ),
                                    ),
                                  ],
                                ),
                    ),
                  ),
                  const SizedBox(height: 16),

                  // Info Panel / Model Status
                  if (_statusMessage != null && !_isProcessing)
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 4.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            _colorService.isLocalFileLoaded
                                ? Icons.folder_open
                                : Icons.cloud_done_outlined,
                            size: 16,
                            color: Colors.grey,
                          ),
                          const SizedBox(width: 4),
                          Text(
                            _colorService.isLocalFileLoaded
                                ? 'Loaded from local path (Optimized)'
                                : 'Loaded from standard assets',
                            style: const TextStyle(color: Colors.grey, fontSize: 12),
                          ),
                        ],
                      ),
                    ),
                  const SizedBox(height: 8),

                  // Execution Stats Card
                  if (hasResult)
                    Card(
                      child: Padding(
                        padding: const EdgeInsets.all(16.0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Row(
                              children: [
                                Icon(Icons.analytics_outlined, color: Color(0xFF10B981)),
                                SizedBox(width: 8),
                                Text(
                                  'Performance Latency Metrics',
                                  style: TextStyle(
                                    fontWeight: FontWeight.bold,
                                    fontSize: 16,
                                  ),
                                ),
                              ],
                            ),
                            const Divider(height: 24),
                            _buildStatRow('Preprocessing', '${_preprocessTimeMs.toStringAsFixed(1)} ms'),
                            const SizedBox(height: 8),
                            _buildStatRow('Model Inference', '${_inferenceTimeMs.toStringAsFixed(1)} ms'),
                            const SizedBox(height: 8),
                            _buildStatRow('Postprocessing', '${_postprocessTimeMs.toStringAsFixed(1)} ms'),
                            const Divider(height: 24),
                            _buildStatRow(
                              'Total Processing Time',
                              '${totalTimeMs.toStringAsFixed(1)} ms',
                              isBold: true,
                              valueColor: const Color(0xFF10B981),
                            ),
                          ],
                        ),
                      ),
                    ),
                  const SizedBox(height: 16),

                  // Action Buttons
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: _isProcessing ? null : () => _pickImage(ImageSource.gallery),
                          icon: const Icon(Icons.photo_library),
                          label: const Text('Gallery'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFF181824),
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(vertical: 16),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                              side: const BorderSide(color: Colors.white10),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: _isProcessing ? null : () => _pickImage(ImageSource.camera),
                          icon: const Icon(Icons.camera_alt),
                          label: const Text('Camera'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFF181824),
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(vertical: 16),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                              side: const BorderSide(color: Colors.white10),
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  ElevatedButton.icon(
                    onPressed: _isProcessing ? null : _loadSampleImage,
                    icon: const Icon(Icons.grain),
                    label: const Text('Load Programmatic Sample'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF6366F1),
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                  ),
                ],
              ),
            ),
    );
  }

  Widget _buildStatRow(String label, String value, {bool isBold = false, Color? valueColor}) {
    final style = TextStyle(
      fontWeight: isBold ? FontWeight.bold : FontWeight.normal,
      color: isBold ? Colors.white : Colors.grey.shade300,
    );
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(label, style: style),
        Text(
          value,
          style: TextStyle(
            fontWeight: isBold ? FontWeight.bold : FontWeight.normal,
            color: valueColor ?? (isBold ? Colors.white : Colors.grey.shade400),
          ),
        ),
      ],
    );
  }
}

class ImageCompareSlider extends StatefulWidget {
  final Widget beforeImage;
  final Widget afterImage;

  const ImageCompareSlider({
    super.key,
    required this.beforeImage,
    required this.afterImage,
  });

  @override
  State<ImageCompareSlider> createState() => _ImageCompareSliderState();
}

class _ImageCompareSliderState extends State<ImageCompareSlider> {
  double _sliderPosition = 0.5;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final double width = constraints.maxWidth;
        final double height = constraints.maxHeight;

        return GestureDetector(
          onPanUpdate: (details) {
            setState(() {
              _sliderPosition = (details.localPosition.dx / width).clamp(0.0, 1.0);
            });
          },
          child: Stack(
            children: [
              // Grayscale / Before image at bottom
              Positioned.fill(child: widget.beforeImage),

              // Colorized / After image clipped at top
              Positioned.fill(
                child: ClipRect(
                  clipper: SliderClipper(_sliderPosition),
                  child: widget.afterImage,
                ),
              ),

              // Divider Line
              Positioned(
                left: width * _sliderPosition - 1,
                top: 0,
                bottom: 0,
                child: Container(
                  width: 2,
                  color: Colors.white,
                ),
              ),

              // Slider Handle (Thumb)
              Positioned(
                left: width * _sliderPosition - 20,
                top: height / 2 - 20,
                child: Container(
                  width: 40,
                  height: 40,
                  decoration: BoxDecoration(
                    color: const Color(0xFF6366F1),
                    shape: BoxShape.circle,
                    border: Border.all(color: Colors.white, width: 2),
                    boxShadow: const [
                      BoxShadow(
                        color: Colors.black38,
                        blurRadius: 6,
                        spreadRadius: 2,
                      )
                    ],
                  ),
                  child: const Center(
                    child: Icon(
                      Icons.swap_horiz,
                      color: Colors.white,
                      size: 24,
                    ),
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

class SliderClipper extends CustomClipper<Rect> {
  final double fraction;

  SliderClipper(this.fraction);

  @override
  Rect getClip(Size size) {
    // Show afterImage on the right side of the slider, or left?
    // In our case, standard compare sliders show before on left, after on right.
    // Wait, if fraction goes from 0.0 to 1.0:
    // If we return Rect.fromLTRB(size.width * fraction, 0, size.width, size.height),
    // then the afterImage will be visible on the right side (from width*fraction to width).
    // Let's do that! That's the standard compare style!
    return Rect.fromLTRB(size.width * fraction, 0, size.width, size.height);
  }

  @override
  bool shouldReclip(SliderClipper oldClipper) => oldClipper.fraction != fraction;
}
