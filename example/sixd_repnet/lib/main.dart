import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'sixd_repnet_service.dart';
import 'pose_painter.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
    ),
  );
  runApp(const SixDRepNetApp());
}

class SixDRepNetApp extends StatelessWidget {
  const SixDRepNetApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SixDRepNet Head Pose',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF0F0F15),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF00E5FF),     // Vibrant Cyan
          secondary: Color(0xFF00E676),   // Vibrant Green
          surface: Color(0xFF161622),
          error: Color(0xFFFF1744),
        ),
        cardTheme: const CardThemeData(
          color: Color(0xFF161622),
          elevation: 6,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.all(Radius.circular(16)),
          ),
        ),
      ),
      home: const PoseHomeScreen(),
    );
  }
}

class PoseHomeScreen extends StatefulWidget {
  const PoseHomeScreen({super.key});

  @override
  State<PoseHomeScreen> createState() => _PoseHomeScreenState();
}

class _PoseHomeScreenState extends State<PoseHomeScreen> {
  final SixDRepNetService _poseService = SixDRepNetService();
  final ImagePicker _picker = ImagePicker();

  Uint8List? _imageBytes;
  Size _imageSize = Size.zero;
  List<FacePose> _detectedPoses = [];

  bool _isModelLoading = false;
  bool _isProcessing = false;
  String? _statusMessage;

  // Parameters
  double _scoreThreshold = 0.85;
  double _nmsThreshold = 0.4;
  HardwareAccelerator _selectedAccelerator = HardwareAccelerator.cpu;

  // Execution Times
  double _preTimeMs = 0.0;
  double _detTimeMs = 0.0;
  double _postTimeMs = 0.0;
  double _poseTimeMs = 0.0;

  @override
  void initState() {
    super.initState();
    _initModels();
  }

  Future<void> _initModels() async {
    setState(() {
      _isModelLoading = true;
      _statusMessage = 'Loading TFLite models...';
    });

    try {
      await _poseService.loadModels(accelerator: _selectedAccelerator);
      setState(() {
        _isModelLoading = false;
        _statusMessage = _poseService.isLocalFileLoaded
            ? 'Loaded (Local System)'
            : 'Loaded (Assets)';
      });
    } catch (e) {
      setState(() {
        _isModelLoading = false;
        _statusMessage = 'Load error';
      });
      _showErrorDialog('Failed to load TFLite models: $e');
    }
  }

  Future<void> _changeAccelerator(HardwareAccelerator? accelerator) async {
    if (accelerator == null || accelerator == _selectedAccelerator) return;
    setState(() {
      _selectedAccelerator = accelerator;
    });
    await _initModels();
    if (_imageBytes != null) {
      _processImage(_imageBytes!);
    }
  }

  Future<Size> _getImageSize(Uint8List bytes) async {
    final ui.Codec codec = await ui.instantiateImageCodec(bytes);
    final ui.FrameInfo frameInfo = await codec.getNextFrame();
    return Size(
      frameInfo.image.width.toDouble(),
      frameInfo.image.height.toDouble(),
    );
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? file = await _picker.pickImage(source: source);
      if (file == null) return;

      final Uint8List bytes = await file.readAsBytes();
      final Size size = await _getImageSize(bytes);

      setState(() {
        _imageBytes = bytes;
        _imageSize = size;
        _detectedPoses = [];
      });

      await _processImage(bytes);
    } catch (e) {
      _showErrorDialog('Failed to pick/load image: $e');
    }
  }

  Future<void> _processImage(Uint8List bytes) async {
    if (!_poseService.isLoaded) {
      _showErrorDialog('Model is not ready.');
      return;
    }

    setState(() {
      _isProcessing = true;
      _statusMessage = 'Running head pose pipeline...';
    });

    try {
      final poses = await _poseService.estimatePose(
        bytes,
        scoreThreshold: _scoreThreshold,
        nmsThreshold: _nmsThreshold,
        onStatsUpdated: (pre, det, post, pose) {
          setState(() {
            _preTimeMs = pre;
            _detTimeMs = det;
            _postTimeMs = post;
            _poseTimeMs = pose;
          });
        },
      );

      setState(() {
        _detectedPoses = poses;
        _isProcessing = false;
        _statusMessage = 'Inference complete. Faces: ${poses.length}';
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _statusMessage = 'Error during inference';
      });
      _showErrorDialog('Inference error: $e');
    }
  }

  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Row(
          children: [
            Icon(Icons.error_outline, color: Color(0xFFFF1744)),
            SizedBox(width: 8),
            Text('Error'),
          ],
        ),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK', style: TextStyle(color: Color(0xFF00E5FF))),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final double totalPipelineTime = _preTimeMs + _detTimeMs + _postTimeMs + _poseTimeMs;

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'SixDRepNet Head Pose',
          style: TextStyle(fontWeight: FontWeight.bold, letterSpacing: 0.8),
        ),
        backgroundColor: const Color(0xFF161622),
        elevation: 2,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh, color: Color(0xFF00E5FF)),
            onPressed: _imageBytes == null ? null : () => _processImage(_imageBytes!),
            tooltip: 'Re-run inference',
          ),
        ],
      ),
      body: Row(
        children: [
          // Left Sidebar for settings & stats on desktop, or top of scroll on mobile
          Expanded(
            flex: 3,
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: ListView(
                children: [
                  _buildStatusCard(),
                  const SizedBox(height: 16),
                  _buildControlsCard(),
                  const SizedBox(height: 16),
                  if (_imageBytes != null) _buildStatsCard(totalPipelineTime),
                ],
              ),
            ),
          ),
          
          // Right panel for image visualization & detailed output
          Expanded(
            flex: 5,
            child: Container(
              color: const Color(0xFF0A0A0F),
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    Expanded(
                      flex: 3,
                      child: _buildImageDisplayArea(),
                    ),
                    const SizedBox(height: 16),
                    Expanded(
                      flex: 2,
                      child: _buildAnglesListArea(),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatusCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Model Status',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white70),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Container(
                  width: 12,
                  height: 12,
                  decoration: BoxDecoration(
                    color: _poseService.isLoaded
                        ? const Color(0xFF00E676)
                        : (_isModelLoading ? Colors.orange : const Color(0xFFFF1744)),
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(
                        color: (_poseService.isLoaded
                            ? const Color(0xFF00E676)
                            : (_isModelLoading ? Colors.orange : const Color(0xFFFF1744))).withOpacity(0.5),
                        blurRadius: 6,
                        spreadRadius: 2,
                      )
                    ],
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    _statusMessage ?? 'Unloaded',
                    style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w500),
                  ),
                ),
              ],
            ),
            if (_isModelLoading || _isProcessing) ...[
              const SizedBox(height: 16),
              const LinearProgressIndicator(
                backgroundColor: Colors.white10,
                color: Color(0xFF00E5FF),
              ),
            ]
          ],
        ),
      ),
    );
  }

  Widget _buildControlsCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Configuration',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white70),
            ),
            const SizedBox(height: 16),
            const Text('Hardware Accelerator', style: TextStyle(fontSize: 12, color: Colors.white38)),
            DropdownButton<HardwareAccelerator>(
              value: _selectedAccelerator,
              isExpanded: true,
              dropdownColor: const Color(0xFF161622),
              items: HardwareAccelerator.values.map((HardwareAccelerator val) {
                return DropdownMenuItem<HardwareAccelerator>(
                  value: val,
                  child: Text(val.name.toUpperCase()),
                );
              }).toList(),
              onChanged: _isModelLoading || _isProcessing ? null : _changeAccelerator,
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Expanded(
                  child: Text('Face Confidence', style: TextStyle(fontSize: 12, color: Colors.white38)),
                ),
                Text('${(_scoreThreshold * 100).toStringAsFixed(0)}%',
                    style: const TextStyle(fontSize: 12, color: Color(0xFF00E5FF), fontWeight: FontWeight.bold)),
              ],
            ),
            Slider(
              value: _scoreThreshold,
              min: 0.50,
              max: 0.99,
              divisions: 49,
              activeColor: const Color(0xFF00E5FF),
              inactiveColor: Colors.white10,
              onChanged: _isProcessing
                  ? null
                  : (val) {
                      setState(() => _scoreThreshold = val);
                    },
              onChangeEnd: (val) {
                if (_imageBytes != null) _processImage(_imageBytes!);
              },
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Expanded(
                  child: Text('NMS IoU Threshold', style: TextStyle(fontSize: 12, color: Colors.white38)),
                ),
                Text(_nmsThreshold.toStringAsFixed(2),
                    style: const TextStyle(fontSize: 12, color: Color(0xFF00E676), fontWeight: FontWeight.bold)),
              ],
            ),
            Slider(
              value: _nmsThreshold,
              min: 0.10,
              max: 0.80,
              divisions: 70,
              activeColor: const Color(0xFF00E676),
              inactiveColor: Colors.white10,
              onChanged: _isProcessing
                  ? null
                  : (val) {
                      setState(() => _nmsThreshold = val);
                    },
              onChangeEnd: (val) {
                if (_imageBytes != null) _processImage(_imageBytes!);
              },
            ),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              icon: const Icon(Icons.photo_library),
              label: const Text('Pick Image'),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF00E5FF),
                foregroundColor: Colors.black,
                minimumSize: const Size.fromHeight(45),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
              ),
              onPressed: _isModelLoading || _isProcessing ? null : () => _pickImage(ImageSource.gallery),
            ),
            const SizedBox(height: 10),
            OutlinedButton.icon(
              icon: const Icon(Icons.camera_alt),
              label: const Text('Take Photo'),
              style: OutlinedButton.styleFrom(
                foregroundColor: const Color(0xFF00E5FF),
                side: const BorderSide(color: Color(0xFF00E5FF)),
                minimumSize: const Size.fromHeight(45),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
              ),
              onPressed: _isModelLoading || _isProcessing ? null : () => _pickImage(ImageSource.camera),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatsCard(double total) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Latency Metrics',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white70),
            ),
            const SizedBox(height: 16),
            _buildStatRow('Pre-processing', _preTimeMs),
            _buildStatRow('Face Detector', _detTimeMs),
            _buildStatRow('Post-processing', _postTimeMs),
            _buildStatRow('Pose Estimator', _poseTimeMs),
            const Divider(color: Colors.white24, height: 24),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Expanded(
                  child: Text(
                    'Total Pipeline',
                    style: TextStyle(fontWeight: FontWeight.bold, color: Color(0xFF00E5FF)),
                  ),
                ),
                Text(
                  '${total.toStringAsFixed(1)} ms',
                  style: const TextStyle(fontWeight: FontWeight.bold, color: Color(0xFF00E5FF)),
                ),
              ],
            )
          ],
        ),
      ),
    );
  }

  Widget _buildStatRow(String title, double time) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Expanded(
            child: Text(title, style: const TextStyle(fontSize: 13, color: Colors.white60)),
          ),
          Text('${time.toStringAsFixed(1)} ms', style: const TextStyle(fontSize: 13, fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }

  Widget _buildImageDisplayArea() {
    if (_imageBytes == null) {
      return Card(
        color: const Color(0xFF161622).withOpacity(0.5),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: const BorderSide(color: Colors.white10),
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.face_retouching_natural, size: 80, color: const Color(0xFF00E5FF).withOpacity(0.2)),
              const SizedBox(height: 16),
              const Text(
                'No Image Selected',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white38),
              ),
              const SizedBox(height: 8),
              const Text(
                'Pick an image or take a photo to start head pose estimation',
                style: TextStyle(fontSize: 12, color: Colors.white24),
              ),
            ],
          ),
        ),
      );
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(8.0),
        child: LayoutBuilder(
          builder: (context, constraints) {
            final double maxWidth = constraints.maxWidth;
            final double maxHeight = constraints.maxHeight;

            final double imageAspectRatio = _imageSize.width / _imageSize.height;
            final double containerAspectRatio = maxWidth / maxHeight;

            double dispW;
            double dispH;

            if (imageAspectRatio > containerAspectRatio) {
              dispW = maxWidth;
              dispH = maxWidth / imageAspectRatio;
            } else {
              dispH = maxHeight;
              dispW = maxHeight * imageAspectRatio;
            }

            return Center(
              child: SizedBox(
                width: dispW,
                height: dispH,
                child: Stack(
                  children: [
                    Image.memory(
                      _imageBytes!,
                      width: dispW,
                      height: dispH,
                      fit: BoxFit.fill,
                    ),
                    if (!_isProcessing)
                      CustomPaint(
                        size: Size(dispW, dispH),
                        painter: PosePainter(
                          poses: _detectedPoses,
                          imageSize: _imageSize,
                        ),
                      ),
                    if (_isProcessing)
                      Container(
                        color: Colors.black45,
                        child: const Center(
                          child: CircularProgressIndicator(
                            color: Color(0xFF00E5FF),
                          ),
                        ),
                      ),
                  ],
                ),
              ),
            );
          },
        ),
      ),
    );
  }

  Widget _buildAnglesListArea() {
    if (_imageBytes == null || _detectedPoses.isEmpty) {
      return const Card(
        child: Center(
          child: Text(
            'Estimation results will appear here',
            style: TextStyle(color: Colors.white24),
          ),
        ),
      );
    }

    return ListView.builder(
      itemCount: _detectedPoses.length,
      scrollDirection: Axis.horizontal,
      itemBuilder: (context, index) {
        final pose = _detectedPoses[index];
        return Container(
          width: 320,
          margin: const EdgeInsets.only(right: 16),
          child: Card(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        'Face #${index + 1}',
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: Color(0xFF00E5FF),
                        ),
                      ),
                      if (pose.score < 1.0)
                        Text(
                          'Score: ${(pose.score * 100).toStringAsFixed(0)}%',
                          style: const TextStyle(fontSize: 12, color: Colors.white38),
                        ),
                    ],
                  ),
                  const Divider(color: Colors.white12, height: 16),
                  Expanded(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        AngleGauge(
                          label: 'Yaw (Rotation L/R)',
                          value: pose.yaw,
                          activeColor: const Color(0xFF00E5FF),
                          leftLabel: 'Left (Turn)',
                          rightLabel: 'Right (Turn)',
                        ),
                        AngleGauge(
                          label: 'Pitch (Tilt Up/Down)',
                          value: pose.pitch,
                          activeColor: const Color(0xFF00E676),
                          leftLabel: 'Down (Tilt)',
                          rightLabel: 'Up (Tilt)',
                        ),
                        AngleGauge(
                          label: 'Roll (Tilt L/R)',
                          value: pose.roll,
                          activeColor: const Color(0xFFFFEB3B),
                          leftLabel: 'Left (Roll)',
                          rightLabel: 'Right (Roll)',
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

class AngleGauge extends StatelessWidget {
  final String label;
  final double value;
  final double minVal;
  final double maxVal;
  final Color activeColor;
  final String leftLabel;
  final String rightLabel;

  const AngleGauge({
    super.key,
    required this.label,
    required this.value,
    this.minVal = -90.0,
    this.maxVal = 90.0,
    required this.activeColor,
    required this.leftLabel,
    required this.rightLabel,
  });

  @override
  Widget build(BuildContext context) {
    final double clampedVal = value.clamp(minVal, maxVal);
    // Convert -90..90 range to 0..1 percentage
    final double percent = (clampedVal - minVal) / (maxVal - minVal);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              label,
              style: const TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: Colors.white70,
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
              decoration: BoxDecoration(
                color: activeColor.withOpacity(0.15),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: activeColor.withOpacity(0.3), width: 0.5),
              ),
              child: Text(
                '${value.toStringAsFixed(1)}°',
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.bold,
                  color: activeColor,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 6),
        Stack(
          children: [
            Container(
              height: 6,
              decoration: BoxDecoration(
                color: Colors.white12,
                borderRadius: BorderRadius.circular(3),
              ),
            ),
            // Center line
            Positioned(
              left: 0,
              right: 0,
              child: Center(
                child: Container(
                  width: 1.5,
                  height: 6,
                  color: Colors.white30,
                ),
              ),
            ),
            FractionallySizedBox(
              widthFactor: percent,
              child: Container(
                height: 6,
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      activeColor.withOpacity(0.4),
                      activeColor,
                    ],
                  ),
                  borderRadius: BorderRadius.circular(3),
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 3),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              leftLabel,
              style: const TextStyle(fontSize: 8, color: Colors.white30),
            ),
            const Text(
              '0°',
              style: TextStyle(fontSize: 8, color: Colors.white30),
            ),
            Text(
              rightLabel,
              style: const TextStyle(fontSize: 8, color: Colors.white30),
            ),
          ],
        ),
      ],
    );
  }
}
