import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:camera/camera.dart';
import 'sixd_repnet_service.dart';
import 'pose_painter.dart';
import 'image_utils.dart';
import 'package:image/image.dart' as img;

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

class _PoseHomeScreenState extends State<PoseHomeScreen>
    with WidgetsBindingObserver {
  final SixDRepNetService _poseService = SixDRepNetService();
  final ImagePicker _picker = ImagePicker();

  int _currentTabIndex = 0;

  // Cameras list & controller
  List<CameraDescription> _cameras = [];
  CameraController? _cameraController;
  int _selectedCameraIndex = -1; // -1 means none

  // Image mode variables
  Uint8List? _staticImageBytes;
  Size _staticImageSize = Size.zero;
  List<FacePose> _staticDetectedPoses = [];

  // Live Camera mode variables
  Size _liveImageSize = Size.zero;
  List<FacePose> _liveDetectedPoses = [];
  bool _isCameraProcessingFrame = false;

  bool _isModelLoading = false;
  bool _isProcessingStatic = false;
  String? _statusMessage;

  // Hyperparameters
  double _scoreThreshold = 0.85;
  double _nmsThreshold = 0.4;
  HardwareAccelerator _selectedAccelerator = HardwareAccelerator.cpu;

  // Latency Metrics (Static mode)
  double _preTimeMs = 0.0;
  double _detTimeMs = 0.0;
  double _postTimeMs = 0.0;
  double _poseTimeMs = 0.0;

  // Latency Metrics (Live camera mode)
  double _livePreTimeMs = 0.0;
  double _liveDetTimeMs = 0.0;
  double _livePostTimeMs = 0.0;
  double _livePoseTimeMs = 0.0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initModels().then((_) {
      _initializeCameraSystem();
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraController?.dispose();
    _poseService.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final CameraController? controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) return;

    if (state == AppLifecycleState.inactive || state == AppLifecycleState.paused) {
      controller.stopImageStream();
    } else if (state == AppLifecycleState.resumed) {
      if (_currentTabIndex == 0) {
        _startCameraStream();
      }
    }
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
            ? 'Models Loaded (Local)'
            : 'Models Loaded (Assets)';
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
    
    // Stop live stream before switching models
    final bool wasStreaming = _cameraController?.value.isStreamingImages ?? false;
    if (wasStreaming) {
      await _cameraController?.stopImageStream();
    }

    setState(() {
      _selectedAccelerator = accelerator;
    });

    await _initModels();

    if (_currentTabIndex == 0 && _cameraController != null) {
      _startCameraStream();
    } else if (_currentTabIndex == 1 && _staticImageBytes != null) {
      _processStaticImage(_staticImageBytes!);
    }
  }

  // Find and setup available cameras
  Future<void> _initializeCameraSystem() async {
    try {
      _cameras = await availableCameras();
      if (_cameras.isEmpty) {
        debugPrint("No cameras found on device.");
        return;
      }

      // Default to front camera for selfie-style pose estimation
      int frontCamIdx = _cameras.indexWhere(
          (cam) => cam.lensDirection == CameraLensDirection.front);
      
      setState(() {
        _selectedCameraIndex = frontCamIdx != -1 ? frontCamIdx : 0;
      });

      if (_currentTabIndex == 0) {
        _startCameraController();
      }
    } catch (e) {
      debugPrint("Camera initialization error: $e");
    }
  }

  Future<void> _startCameraController() async {
    if (_selectedCameraIndex == -1 || _cameras.isEmpty) return;
    
    await _cameraController?.dispose();
    
    final controller = CameraController(
      _cameras[_selectedCameraIndex],
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: Platform.isAndroid
          ? ImageFormatGroup.yuv420
          : ImageFormatGroup.bgra8888,
    );

    _cameraController = controller;

    try {
      await controller.initialize();
      if (mounted) {
        setState(() {});
        _startCameraStream();
      }
    } catch (e) {
      _showErrorDialog('Failed to initialize camera preview: $e');
    }
  }

  void _startCameraStream() {
    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) return;

    if (controller.value.isStreamingImages) return;

    controller.startImageStream((CameraImage image) {
      _processCameraFrame(image);
    });
  }

  Future<void> _toggleCameraLens() async {
    if (_cameras.length < 2) return;
    setState(() {
      _selectedCameraIndex = (_selectedCameraIndex + 1) % _cameras.length;
      _liveDetectedPoses = [];
    });
    await _startCameraController();
  }

  // Real-time camera frame processing
  void _processCameraFrame(CameraImage cameraImage) async {
    if (_isCameraProcessingFrame || !_poseService.isLoaded || _isModelLoading) return;
    
    _isCameraProcessingFrame = true;

    try {
      // 1. Convert frame to img.Image
      final img.Image? decoded = ImageUtils.convertCameraImage(cameraImage);
      if (decoded != null) {
        // 2. Perform rotation/mirroring for upright portrait orientation
        final bool isFront = _cameras[_selectedCameraIndex].lensDirection == CameraLensDirection.front;
        
        img.Image oriented = decoded;
        if (isFront) {
          oriented = img.flip(oriented, direction: img.FlipDirection.horizontal);
        }
        // Rotate 90 degrees to offset the standard landscape camera mount
        oriented = img.copyRotate(oriented, angle: 90);

        // 3. Execute SixDRepNet pipeline
        final poses = await _poseService.estimatePoseFromImage(
          oriented,
          scoreThreshold: _scoreThreshold,
          nmsThreshold: _nmsThreshold,
          onStatsUpdated: (pre, det, post, pose) {
            if (mounted) {
              setState(() {
                _livePreTimeMs = pre;
                _liveDetTimeMs = det;
                _livePostTimeMs = post;
                _livePoseTimeMs = pose;
              });
            }
          },
        );

        if (mounted) {
          setState(() {
            _liveDetectedPoses = poses;
            _liveImageSize = Size(oriented.width.toDouble(), oriented.height.toDouble());
          });
        }
      }
    } catch (e) {
      debugPrint("Live camera frame estimation error: $e");
    } finally {
      _isCameraProcessingFrame = false;
    }
  }

  // Static Image Pick & Process
  Future<Size> _getImageSize(Uint8List bytes) async {
    final ui.Codec codec = await ui.instantiateImageCodec(bytes);
    final ui.FrameInfo frameInfo = await codec.getNextFrame();
    return Size(
      frameInfo.image.width.toDouble(),
      frameInfo.image.height.toDouble(),
    );
  }

  Future<void> _pickStaticImage(ImageSource source) async {
    try {
      final XFile? file = await _picker.pickImage(source: source);
      if (file == null) return;

      final Uint8List bytes = await file.readAsBytes();
      final Size size = await _getImageSize(bytes);

      setState(() {
        _staticImageBytes = bytes;
        _staticImageSize = size;
        _staticDetectedPoses = [];
      });

      await _processStaticImage(bytes);
    } catch (e) {
      _showErrorDialog('Failed to load/decode selected image: $e');
    }
  }

  Future<void> _processStaticImage(Uint8List bytes) async {
    if (!_poseService.isLoaded) {
      _showErrorDialog('Model is not loaded. Ensure models are active.');
      return;
    }

    setState(() {
      _isProcessingStatic = true;
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
        _staticDetectedPoses = poses;
        _isProcessingStatic = false;
      });
    } catch (e) {
      setState(() {
        _isProcessingStatic = false;
      });
      _showErrorDialog('Static analysis error: $e');
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
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'SixDRepNet Head Pose',
          style: TextStyle(fontWeight: FontWeight.bold, letterSpacing: 0.8),
        ),
        backgroundColor: const Color(0xFF161622),
        elevation: 2,
        actions: [
          if (_currentTabIndex == 1 && _staticImageBytes != null)
            IconButton(
              icon: const Icon(Icons.refresh, color: Color(0xFF00E5FF)),
              onPressed: _isProcessingStatic ? null : () => _processStaticImage(_staticImageBytes!),
              tooltip: 'Re-run static pose estimation',
            ),
        ],
      ),
      body: _buildBody(),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentTabIndex,
        backgroundColor: const Color(0xFF161622),
        selectedItemColor: const Color(0xFF00E5FF),
        unselectedItemColor: Colors.white38,
        showUnselectedLabels: true,
        type: BottomNavigationBarType.fixed,
        onTap: (int index) {
          if (index == _currentTabIndex) return;

          // Stop or start camera stream depending on tab active states
          if (index == 0) {
            _startCameraController();
          } else {
            _cameraController?.stopImageStream();
            _cameraController?.dispose();
            _cameraController = null;
          }

          setState(() {
            _currentTabIndex = index;
          });
        },
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.videocam),
            label: 'Live Video',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.photo_library),
            label: 'Photo Analyzer',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.settings),
            label: 'Settings',
          ),
        ],
      ),
    );
  }

  Widget _buildBody() {
    switch (_currentTabIndex) {
      case 0:
        return _buildLiveCameraTab();
      case 1:
        return _buildPhotoAnalyzerTab();
      case 2:
        return _buildSettingsTab();
      default:
        return const SizedBox.shrink();
    }
  }

  // ==================== TAB 1: LIVE VIDEO ====================
  Widget _buildLiveCameraTab() {
    final double totalLiveTime = _livePreTimeMs + _liveDetTimeMs + _livePostTimeMs + _livePoseTimeMs;

    return Column(
      children: [
        // Camera Preview Area
        Expanded(
          flex: 5,
          child: Container(
            color: const Color(0xFF0A0A0F),
            child: _buildCameraDisplayArea(),
          ),
        ),
        // Live angles and summary card at bottom
        Expanded(
          flex: 3,
          child: Container(
            color: const Color(0xFF0F0F15),
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            child: _liveDetectedPoses.isEmpty
                ? const Center(
                    child: Text(
                      'No faces detected. Position your face in front of the camera.',
                      textAlign: TextAlign.center,
                      style: TextStyle(color: Colors.white30, fontSize: 13),
                    ),
                  )
                : ListView.builder(
                    itemCount: _liveDetectedPoses.length,
                    scrollDirection: Axis.horizontal,
                    itemBuilder: (context, index) {
                      final pose = _liveDetectedPoses[index];
                      return Container(
                        width: MediaQuery.of(context).size.width - 32,
                        margin: const EdgeInsets.only(right: 16),
                        child: Card(
                          child: Padding(
                            padding: const EdgeInsets.all(12.0),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                  children: [
                                    Text(
                                      'Face #${index + 1}',
                                      style: const TextStyle(
                                        fontSize: 14,
                                        fontWeight: FontWeight.bold,
                                        color: Color(0xFF00E5FF),
                                      ),
                                    ),
                                    Text(
                                      'Inference: ${totalLiveTime.toStringAsFixed(1)} ms',
                                      style: const TextStyle(fontSize: 11, color: Colors.white38),
                                    ),
                                  ],
                                ),
                                const Divider(color: Colors.white12, height: 12),
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
                  ),
          ),
        ),
      ],
    );
  }

  Widget _buildCameraDisplayArea() {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(color: Color(0xFF00E5FF)),
            SizedBox(height: 16),
            Text('Starting camera preview...', style: TextStyle(color: Colors.white30, fontSize: 14)),
          ],
        ),
      );
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        final double maxWidth = constraints.maxWidth;
        final double maxHeight = constraints.maxHeight;

        // Camera aspect ratio is height / width (rotated)
        final double previewAspectRatio = _cameraController!.value.aspectRatio;
        final double displayAspectRatio = 1.0 / previewAspectRatio;

        final double containerAspectRatio = maxWidth / maxHeight;

        double dispW;
        double dispH;

        if (displayAspectRatio > containerAspectRatio) {
          dispW = maxWidth;
          dispH = maxWidth / displayAspectRatio;
        } else {
          dispH = maxHeight;
          dispW = maxHeight * displayAspectRatio;
        }

        return Center(
          child: SizedBox(
            width: dispW,
            height: dispH,
            child: Stack(
              children: [
                CameraPreview(_cameraController!),
                if (_liveDetectedPoses.isNotEmpty)
                  CustomPaint(
                    size: Size(dispW, dispH),
                    painter: PosePainter(
                      poses: _liveDetectedPoses,
                      imageSize: _liveImageSize,
                    ),
                  ),
                // Camera Lens Toggle button
                Positioned(
                  top: 16,
                  right: 16,
                  child: FloatingActionButton.small(
                    backgroundColor: Colors.black54,
                    foregroundColor: const Color(0xFF00E5FF),
                    onPressed: _toggleCameraLens,
                    child: const Icon(Icons.flip_camera_ios),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  // ==================== TAB 2: PHOTO ANALYZER ====================
  Widget _buildPhotoAnalyzerTab() {
    final double totalStaticTime = _preTimeMs + _detTimeMs + _postTimeMs + _poseTimeMs;

    return Column(
      children: [
        // Image Display Area
        Expanded(
          flex: 5,
          child: Container(
            color: const Color(0xFF0A0A0F),
            child: _buildStaticImageDisplayArea(),
          ),
        ),
        // Results & Metrics at Bottom
        Expanded(
          flex: 4,
          child: Container(
            color: const Color(0xFF0F0F15),
            padding: const EdgeInsets.all(16.0),
            child: _staticImageBytes == null
                ? Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        ElevatedButton.icon(
                          icon: const Icon(Icons.photo_library),
                          label: const Text('Pick Gallery Image'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFF00E5FF),
                            foregroundColor: Colors.black,
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                          ),
                          onPressed: () => _pickStaticImage(ImageSource.gallery),
                        ),
                        const SizedBox(height: 12),
                        OutlinedButton.icon(
                          icon: const Icon(Icons.camera_alt),
                          label: const Text('Capture with Camera'),
                          style: OutlinedButton.styleFrom(
                            foregroundColor: const Color(0xFF00E5FF),
                            side: const BorderSide(color: Color(0xFF00E5FF)),
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                          ),
                          onPressed: () => _pickStaticImage(ImageSource.camera),
                        ),
                      ],
                    ),
                  )
                : Column(
                    children: [
                      // Toggle list / metrics
                      Expanded(
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            // Left list: Detected faces
                            Expanded(
                              flex: 3,
                              child: _staticDetectedPoses.isEmpty
                                  ? const Center(child: Text('No face detected', style: TextStyle(color: Colors.white30)))
                                  : ListView.builder(
                                      itemCount: _staticDetectedPoses.length,
                                      scrollDirection: Axis.horizontal,
                                      itemBuilder: (context, index) {
                                        final pose = _staticDetectedPoses[index];
                                        return Container(
                                          width: MediaQuery.of(context).size.width - 150,
                                          margin: const EdgeInsets.only(right: 12),
                                          child: Card(
                                            color: const Color(0xFF1E1E2E),
                                            child: Padding(
                                              padding: const EdgeInsets.all(10.0),
                                              child: Column(
                                                crossAxisAlignment: CrossAxisAlignment.start,
                                                children: [
                                                  Text(
                                                    'Face #${index + 1} (${(pose.score * 100).toStringAsFixed(0)}%)',
                                                    style: const TextStyle(
                                                      fontSize: 12,
                                                      fontWeight: FontWeight.bold,
                                                      color: Color(0xFF00E5FF),
                                                    ),
                                                  ),
                                                  const Divider(color: Colors.white10, height: 8),
                                                  Expanded(
                                                    child: Column(
                                                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                                                      children: [
                                                        AngleGauge(
                                                          label: 'Yaw',
                                                          value: pose.yaw,
                                                          activeColor: const Color(0xFF00E5FF),
                                                          leftLabel: 'L',
                                                          rightLabel: 'R',
                                                        ),
                                                        AngleGauge(
                                                          label: 'Pitch',
                                                          value: pose.pitch,
                                                          activeColor: const Color(0xFF00E676),
                                                          leftLabel: 'D',
                                                          rightLabel: 'U',
                                                        ),
                                                        AngleGauge(
                                                          label: 'Roll',
                                                          value: pose.roll,
                                                          activeColor: const Color(0xFFFFEB3B),
                                                          leftLabel: 'L',
                                                          rightLabel: 'R',
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
                                    ),
                            ),
                            const SizedBox(width: 8),
                            // Right card: Latency stats
                            Expanded(
                              flex: 2,
                              child: Card(
                                child: Padding(
                                  padding: const EdgeInsets.all(10.0),
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                                    children: [
                                      const Text(
                                        'Stats (ms)',
                                        style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.white70),
                                      ),
                                      const Divider(color: Colors.white10, height: 6),
                                      _buildCompactStatRow('Pre', _preTimeMs),
                                      _buildCompactStatRow('Det', _detTimeMs),
                                      _buildCompactStatRow('Post', _postTimeMs),
                                      _buildCompactStatRow('Pose', _poseTimeMs),
                                      const Divider(color: Colors.white10, height: 6),
                                      Row(
                                         mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                         children: [
                                           const Text('Total', style: TextStyle(fontSize: 10, fontWeight: FontWeight.bold, color: Color(0xFF00E5FF))),
                                           Text(totalStaticTime.toStringAsFixed(1), style: const TextStyle(fontSize: 10, fontWeight: FontWeight.bold, color: Color(0xFF00E5FF))),
                                         ],
                                       )
                                    ],
                                  ),
                                ),
                              ),
                            )
                          ],
                        ),
                      ),
                      const SizedBox(height: 8),
                      // Action buttons to pick another
                      Row(
                        children: [
                          Expanded(
                            child: ElevatedButton.icon(
                              icon: const Icon(Icons.photo_library, size: 16),
                              label: const Text('Pick Gallery', style: TextStyle(fontSize: 12)),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: const Color(0xFF00E5FF),
                                foregroundColor: Colors.black,
                                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                                padding: const EdgeInsets.symmetric(vertical: 10),
                              ),
                              onPressed: () => _pickStaticImage(ImageSource.gallery),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: OutlinedButton.icon(
                              icon: const Icon(Icons.camera_alt, size: 16),
                              label: const Text('Take Photo', style: TextStyle(fontSize: 12)),
                              style: OutlinedButton.styleFrom(
                                foregroundColor: const Color(0xFF00E5FF),
                                side: const BorderSide(color: Color(0xFF00E5FF)),
                                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                                padding: const EdgeInsets.symmetric(vertical: 10),
                              ),
                              onPressed: () => _pickStaticImage(ImageSource.camera),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
          ),
        ),
      ],
    );
  }

  Widget _buildCompactStatRow(String label, double val) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(label, style: const TextStyle(fontSize: 10, color: Colors.white38)),
        Text(val.toStringAsFixed(1), style: const TextStyle(fontSize: 10, fontWeight: FontWeight.bold)),
      ],
    );
  }

  Widget _buildStaticImageDisplayArea() {
    if (_staticImageBytes == null) {
      return Card(
        color: const Color(0xFF161622).withOpacity(0.5),
        margin: const EdgeInsets.all(16),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: const BorderSide(color: Colors.white10),
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.face_retouching_natural, size: 70, color: const Color(0xFF00E5FF).withOpacity(0.15)),
              const SizedBox(height: 12),
              const Text(
                'Static Photo Mode',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white38),
              ),
              const SizedBox(height: 6),
              const Text(
                'Select a photo to detect faces and analyze 3D head poses.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 11, color: Colors.white24),
              ),
            ],
          ),
        ),
      );
    }

    return Card(
      margin: const EdgeInsets.all(16),
      child: Padding(
        padding: const EdgeInsets.all(6.0),
        child: LayoutBuilder(
          builder: (context, constraints) {
            final double maxWidth = constraints.maxWidth;
            final double maxHeight = constraints.maxHeight;

            final double imageAspectRatio = _staticImageSize.width / _staticImageSize.height;
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
                      _staticImageBytes!,
                      width: dispW,
                      height: dispH,
                      fit: BoxFit.fill,
                    ),
                    if (!_isProcessingStatic)
                      CustomPaint(
                        size: Size(dispW, dispH),
                        painter: PosePainter(
                          poses: _staticDetectedPoses,
                          imageSize: _staticImageSize,
                        ),
                      ),
                    if (_isProcessingStatic)
                      Container(
                        color: Colors.black45,
                        child: const Center(
                          child: CircularProgressIndicator(color: Color(0xFF00E5FF)),
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

  // ==================== TAB 3: SETTINGS ====================
  Widget _buildSettingsTab() {
    return ListView(
      padding: const EdgeInsets.all(16.0),
      children: [
        _buildStatusCard(),
        const SizedBox(height: 16),
        _buildSettingsConfigCard(),
        const SizedBox(height: 16),
        _buildLegendCard(),
      ],
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
              'Model Backend Status',
              style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: Colors.white70),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Container(
                  width: 10,
                  height: 10,
                  decoration: BoxDecoration(
                    color: _poseService.isLoaded
                        ? const Color(0xFF00E676)
                        : (_isModelLoading ? Colors.orange : const Color(0xFFFF1744)),
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(
                        color: (_poseService.isLoaded
                            ? const Color(0xFF00E676)
                            : (_isModelLoading ? Colors.orange : const Color(0xFFFF1744))).withOpacity(0.4),
                        blurRadius: 4,
                        spreadRadius: 1,
                      )
                    ],
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    _statusMessage ?? 'Unloaded',
                    style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w500),
                  ),
                ),
              ],
            ),
            if (_isModelLoading) ...[
              const SizedBox(height: 12),
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

  Widget _buildSettingsConfigCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Parameters & Accelerator',
              style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: Colors.white70),
            ),
            const SizedBox(height: 16),
            const Text('Hardware Accelerator', style: TextStyle(fontSize: 11, color: Colors.white30)),
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
              onChanged: _isModelLoading ? null : _changeAccelerator,
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Face Match Confidence', style: TextStyle(fontSize: 11, color: Colors.white30)),
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
              onChanged: (val) {
                setState(() => _scoreThreshold = val);
              },
              onChangeEnd: (val) {
                if (_staticImageBytes != null) _processStaticImage(_staticImageBytes!);
              },
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('NMS Overlap Threshold (IoU)', style: TextStyle(fontSize: 11, color: Colors.white30)),
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
              onChanged: (val) {
                setState(() => _nmsThreshold = val);
              },
              onChangeEnd: (val) {
                if (_staticImageBytes != null) _processStaticImage(_staticImageBytes!);
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLegendCard() {
    return const Card(
      child: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              '3D Axis Legend',
              style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: Colors.white70),
            ),
            SizedBox(height: 12),
            Row(
              children: [
                Icon(Icons.horizontal_rule, color: Color(0xFFFF1744), size: 20),
                SizedBox(width: 8),
                Text('Pitch Axis (Red): Tilting Head Up / Down', style: TextStyle(fontSize: 12, color: Colors.white60)),
              ],
            ),
            SizedBox(height: 8),
            Row(
              children: [
                Icon(Icons.horizontal_rule, color: Color(0xFF00E676), size: 20),
                SizedBox(width: 8),
                Text('Yaw Axis (Green): Turning Head Left / Right', style: TextStyle(fontSize: 12, color: Colors.white60)),
              ],
            ),
            SizedBox(height: 8),
            Row(
              children: [
                Icon(Icons.horizontal_rule, color: Color(0xFF2979FF), size: 20),
                SizedBox(width: 8),
                Text('Roll Axis (Blue): Tilting Head Left / Right Side', style: TextStyle(fontSize: 12, color: Colors.white60)),
              ],
            ),
          ],
        ),
      ),
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
