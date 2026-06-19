import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:camera/camera.dart';
import 'eyegaze_service.dart';
import 'gaze_painter.dart';
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
  runApp(const EyeGazeApp());
}

class EyeGazeApp extends StatelessWidget {
  const EyeGazeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EyeGaze Gaze Direction',
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
      home: const GazeHomeScreen(),
    );
  }
}

class GazeHomeScreen extends StatefulWidget {
  const GazeHomeScreen({super.key});

  @override
  State<GazeHomeScreen> createState() => _GazeHomeScreenState();
}

class _GazeHomeScreenState extends State<GazeHomeScreen>
    with WidgetsBindingObserver {
  final EyeGazeService _gazeService = EyeGazeService();
  final ImagePicker _picker = ImagePicker();

  int _currentTabIndex = 0;

  // Cameras list & controller
  List<CameraDescription> _cameras = [];
  CameraController? _cameraController;
  int _selectedCameraIndex = -1; // -1 means none

  // Image mode variables
  Uint8List? _staticImageBytes;
  Size _staticImageSize = Size.zero;
  List<EyeGazeResult> _staticGazeResults = [];

  // Live Camera mode variables
  Size _liveImageSize = Size.zero;
  List<EyeGazeResult> _liveGazeResults = [];
  bool _isCameraProcessingFrame = false;

  bool _isModelLoading = false;
  bool _isProcessingStatic = false;
  String? _statusMessage;

  // Hyperparameters
  double _scoreThreshold = 0.85;
  double _nmsThreshold = 0.4;
  double _eyeCropScale = 0.35;
  bool _rawEyeCropMode = false;
  String _rawEyeSide = 'left'; // 'left' or 'right'
  
  bool _drawFaces = true;
  bool _drawEyes = true;
  HardwareAccelerator _selectedAccelerator = HardwareAccelerator.cpu;

  // Latency Metrics (Static mode)
  double _preTimeMs = 0.0;
  double _detTimeMs = 0.0;
  double _postTimeMs = 0.0;
  double _gazeTimeMs = 0.0;

  // Latency Metrics (Live camera mode)
  double _livePreTimeMs = 0.0;
  double _liveDetTimeMs = 0.0;
  double _livePostTimeMs = 0.0;
  double _liveGazeTimeMs = 0.0;

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
    _gazeService.dispose();
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
      await _gazeService.loadModels(accelerator: _selectedAccelerator);
      setState(() {
        _isModelLoading = false;
        _statusMessage = _gazeService.isLocalFileLoaded
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

      // Default to front camera for selfie-style gaze estimation
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
    if (_selectedCameraIndex < 0 || _selectedCameraIndex >= _cameras.length) return;

    await _cameraController?.dispose();
    _cameraController = null;

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
      _liveGazeResults = [];
    });
    await _startCameraController();
  }

  // Real-time camera frame processing
  void _processCameraFrame(CameraImage cameraImage) async {
    if (_isCameraProcessingFrame || !_gazeService.isLoaded || _isModelLoading) return;
    
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

        // 3. Execute EyeGaze pipeline
        // In live camera feed, we always run in Full Face Mode (auto eyes crop)
        final results = await _gazeService.estimateGazeFromImage(
          oriented,
          scoreThreshold: _scoreThreshold,
          nmsThreshold: _nmsThreshold,
          eyeCropScale: _eyeCropScale,
          rawEyeCropMode: false, // Force Full Face Mode on video for live gaze tracking
          onStatsUpdated: (pre, det, post, gaze) {
            if (mounted) {
              setState(() {
                _livePreTimeMs = pre;
                _liveDetTimeMs = det;
                _livePostTimeMs = post;
                _liveGazeTimeMs = gaze;
              });
            }
          },
        );

        if (mounted) {
          setState(() {
            _liveGazeResults = results;
            _liveImageSize = Size(oriented.width.toDouble(), oriented.height.toDouble());
          });
        }
      }
    } catch (e) {
      debugPrint("Error processing camera frame: $e");
    } finally {
      _isCameraProcessingFrame = false;
    }
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
        _staticGazeResults = [];
      });

      await _processStaticImage(bytes);
    } catch (e) {
      _showErrorDialog('Failed to load/decode selected image: $e');
    }
  }

  Future<void> _processStaticImage(Uint8List bytes) async {
    if (!_gazeService.isLoaded) {
      _showErrorDialog('Model is not loaded. Ensure models are active.');
      return;
    }

    setState(() {
      _isProcessingStatic = true;
    });

    try {
      final results = await _gazeService.estimateGaze(
        bytes,
        scoreThreshold: _scoreThreshold,
        nmsThreshold: _nmsThreshold,
        eyeCropScale: _eyeCropScale,
        rawEyeCropMode: _rawEyeCropMode,
        rawEyeSide: _rawEyeSide,
        onStatsUpdated: (pre, det, post, gaze) {
          setState(() {
            _preTimeMs = pre;
            _detTimeMs = det;
            _postTimeMs = post;
            _gazeTimeMs = gaze;
          });
        },
      );

      setState(() {
        _staticGazeResults = results;
        _isProcessingStatic = false;
      });
    } catch (e) {
      setState(() {
        _isProcessingStatic = false;
      });
      _showErrorDialog('Error during image gaze estimation: $e');
    }
  }

  // Decodes image size helper
  Future<Size> _getImageSize(Uint8List bytes) async {
    final ui.Codec codec = await ui.instantiateImageCodec(bytes);
    final ui.FrameInfo frameInfo = await codec.getNextFrame();
    return Size(frameInfo.image.width.toDouble(), frameInfo.image.height.toDouble());
  }

  void _showErrorDialog(String message) {
    if (!mounted) return;
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
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
            onPressed: () => Navigator.of(ctx).pop(),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final isLandscape = MediaQuery.of(context).orientation == Orientation.landscape;

    return DefaultTabController(
      length: 2,
      child: Scaffold(
        appBar: AppBar(
          title: const Text(
            '👁️ Qualcomm EyeGaze',
            style: TextStyle(fontWeight: FontWeight.bold, letterSpacing: 0.5),
          ),
          centerTitle: false,
          elevation: 2,
          backgroundColor: const Color(0xFF161622),
          bottom: TabBar(
            onTap: (index) {
              setState(() {
                _currentTabIndex = index;
              });
              if (index == 0) {
                // Resume camera stream
                _startCameraStream();
              } else {
                // Pause camera stream to save CPU
                _cameraController?.stopImageStream();
              }
            },
            tabs: const [
              Tab(icon: Icon(Icons.videocam), text: 'Live Video Feed'),
              Tab(icon: Icon(Icons.photo_library), text: 'Static Image'),
            ],
            indicatorColor: Theme.of(context).colorScheme.primary,
            labelStyle: const TextStyle(fontWeight: FontWeight.bold),
          ),
          actions: [
            // Status Indicator Icon
            Container(
              margin: const EdgeInsets.only(right: 16),
              child: Tooltip(
                message: _statusMessage ?? 'Checking status...',
                child: CircleAvatar(
                  radius: 6,
                  backgroundColor: _gazeService.isLoaded
                      ? const Color(0xFF00E676)
                      : const Color(0xFFFF1744),
                ),
              ),
            ),
          ],
        ),
        body: Column(
          children: [
            Expanded(
              child: TabBarView(
                physics: const NeverScrollableScrollPhysics(),
                children: [
                  // TAB 1: Live Video Feed
                  _buildLiveCameraTab(),
                  // TAB 2: Static Image
                  _buildStaticImageTab(),
                ],
              ),
            ),
            // Bottom configuration / details panel
            _buildConfigurationPanel(isLandscape),
          ],
        ),
      ),
    );
  }

  Widget _buildLiveCameraTab() {
    final controller = _cameraController;

    if (controller == null || !controller.value.isInitialized) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Initializing Camera Feed...', style: TextStyle(color: Colors.white70)),
          ],
        ),
      );
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        final previewSize = controller.value.previewSize!;
        // Camera previewSize is landscape-based (width is larger than height),
        // but our UI preview is portrait, so we swap them.
        final double previewRatio = previewSize.height / previewSize.width;

        return Stack(
          fit: StackFit.expand,
          children: [
            // Camera Preview (cropped/scaled to fit the layout)
            ClipRect(
              child: OverflowBox(
                alignment: Alignment.center,
                child: FittedBox(
                  fit: BoxFit.cover,
                  child: SizedBox(
                    width: constraints.maxWidth,
                    height: constraints.maxWidth / previewRatio,
                    child: CameraPreview(controller),
                  ),
                ),
              ),
            ),
            // Painter Overlay
            if (_liveGazeResults.isNotEmpty && _liveImageSize != Size.zero)
              Positioned.fill(
                child: CustomPaint(
                  painter: GazePainter(
                    gazeResults: _liveGazeResults,
                    imageSize: _liveImageSize,
                    drawFaces: _drawFaces,
                    drawEyes: _drawEyes,
                  ),
                ),
              ),
            // Camera controls inside preview
            Positioned(
              top: 16,
              right: 16,
              child: FloatingActionButton.small(
                onPressed: _toggleCameraLens,
                backgroundColor: const Color(0xFF161622).withOpacity(0.8),
                foregroundColor: Colors.white,
                child: const Icon(Icons.flip_camera_ios),
              ),
            ),
            // Stats Overlay inside Preview
            Positioned(
              bottom: 16,
              left: 16,
              right: 16,
              child: _buildLatencyStatsCard(
                pre: _livePreTimeMs,
                det: _liveDetTimeMs,
                post: _livePostTimeMs,
                gaze: _liveGazeTimeMs,
                isLive: true,
              ),
            ),
          ],
        );
      },
    );
  }

  Widget _buildStaticImageTab() {
    return Column(
      children: [
        // Tab sub-bar control (Raw eye crop mode toggle)
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          color: const Color(0xFF161622).withOpacity(0.5),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Static Processing Mode:',
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 13),
              ),
              Row(
                children: [
                  ChoiceChip(
                    label: const Text('Full Face (Auto)'),
                    selected: !_rawEyeCropMode,
                    onSelected: (selected) {
                      if (selected) {
                        setState(() {
                          _rawEyeCropMode = false;
                        });
                        if (_staticImageBytes != null) {
                          _processStaticImage(_staticImageBytes!);
                        }
                      }
                    },
                    selectedColor: Theme.of(context).colorScheme.primary.withOpacity(0.3),
                    labelStyle: TextStyle(
                      color: !_rawEyeCropMode ? Theme.of(context).colorScheme.primary : Colors.white60,
                      fontWeight: FontWeight.bold,
                      fontSize: 12,
                    ),
                  ),
                  const SizedBox(width: 8),
                  ChoiceChip(
                    label: const Text('Raw Eye Crop'),
                    selected: _rawEyeCropMode,
                    onSelected: (selected) {
                      if (selected) {
                        setState(() {
                          _rawEyeCropMode = true;
                        });
                        if (_staticImageBytes != null) {
                          _processStaticImage(_staticImageBytes!);
                        }
                      }
                    },
                    selectedColor: Theme.of(context).colorScheme.primary.withOpacity(0.3),
                    labelStyle: TextStyle(
                      color: _rawEyeCropMode ? Theme.of(context).colorScheme.primary : Colors.white60,
                      fontWeight: FontWeight.bold,
                      fontSize: 12,
                    ),
                  ),
                ],
              )
            ],
          ),
        ),
        // If in raw eye crop mode, select Left vs Right side
        if (_rawEyeCropMode)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
            color: const Color(0xFF161622).withOpacity(0.3),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'Crop Eye Side (for right eye, yaw is inverted):',
                  style: TextStyle(fontSize: 12, color: Colors.white70),
                ),
                Row(
                  children: [
                    Radio<String>(
                      value: 'left',
                      groupValue: _rawEyeSide,
                      activeColor: Theme.of(context).colorScheme.primary,
                      onChanged: (val) {
                        if (val != null) {
                          setState(() {
                            _rawEyeSide = val;
                          });
                          if (_staticImageBytes != null) {
                            _processStaticImage(_staticImageBytes!);
                          }
                        }
                      },
                    ),
                    const Text('Left Eye', style: TextStyle(fontSize: 12)),
                    const SizedBox(width: 12),
                    Radio<String>(
                      value: 'right',
                      groupValue: _rawEyeSide,
                      activeColor: Theme.of(context).colorScheme.primary,
                      onChanged: (val) {
                        if (val != null) {
                          setState(() {
                            _rawEyeSide = val;
                          });
                          if (_staticImageBytes != null) {
                            _processStaticImage(_staticImageBytes!);
                          }
                        }
                      },
                    ),
                    const Text('Right Eye', style: TextStyle(fontSize: 12)),
                  ],
                ),
              ],
            ),
          ),
        Expanded(
          child: Stack(
            fit: StackFit.expand,
            children: [
              if (_staticImageBytes != null)
                Center(
                  child: InteractiveViewer(
                    maxScale: 4.0,
                    child: Stack(
                      children: [
                        Image.memory(
                          _staticImageBytes!,
                          fit: BoxFit.contain,
                        ),
                        if (_staticGazeResults.isNotEmpty && _staticImageSize != Size.zero)
                          Positioned.fill(
                            child: CustomPaint(
                              painter: GazePainter(
                                gazeResults: _staticGazeResults,
                                imageSize: _staticImageSize,
                                drawFaces: _drawFaces,
                                drawEyes: _drawEyes,
                              ),
                            ),
                          ),
                      ],
                    ),
                  ),
                )
              else
                Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        Icons.add_photo_alternate_outlined,
                        size: 80,
                        color: Colors.white24,
                      ),
                      const SizedBox(height: 16),
                      const Text(
                        'No Image Selected',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: Colors.white54,
                        ),
                      ),
                      const SizedBox(height: 8),
                      const Text(
                        'Upload a face portrait or a cropped eye image.',
                        style: TextStyle(color: Colors.white30, fontSize: 13),
                      ),
                      const SizedBox(height: 24),
                      Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          ElevatedButton.icon(
                            onPressed: () => _pickStaticImage(ImageSource.gallery),
                            icon: const Icon(Icons.photo_library),
                            label: const Text('Gallery'),
                            style: ElevatedButton.styleFrom(
                              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                            ),
                          ),
                          const SizedBox(width: 16),
                          ElevatedButton.icon(
                            onPressed: () => _pickStaticImage(ImageSource.camera),
                            icon: const Icon(Icons.camera_alt),
                            label: const Text('Take Photo'),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Theme.of(context).colorScheme.primary,
                              foregroundColor: Colors.black,
                              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              // Floating progress indicator
              if (_isProcessingStatic)
                Container(
                  color: Colors.black45,
                  child: const Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        CircularProgressIndicator(),
                        SizedBox(height: 16),
                        Text('Estimating gaze direction...', style: TextStyle(color: Colors.white70)),
                      ],
                    ),
                  ),
                ),
              // Floating pick actions when image is loaded
              if (_staticImageBytes != null)
                Positioned(
                  top: 16,
                  right: 16,
                  child: Row(
                    children: [
                      FloatingActionButton.small(
                        heroTag: 'pick_gallery',
                        onPressed: () => _pickStaticImage(ImageSource.gallery),
                        backgroundColor: const Color(0xFF161622).withOpacity(0.8),
                        foregroundColor: Colors.white,
                        child: const Icon(Icons.photo_library),
                      ),
                      const SizedBox(width: 8),
                      FloatingActionButton.small(
                        heroTag: 'pick_camera',
                        onPressed: () => _pickStaticImage(ImageSource.camera),
                        backgroundColor: const Color(0xFF161622).withOpacity(0.8),
                        foregroundColor: Colors.white,
                        child: const Icon(Icons.camera_alt),
                      ),
                      const SizedBox(width: 8),
                      FloatingActionButton.small(
                        heroTag: 'clear_img',
                        onPressed: () {
                          setState(() {
                            _staticImageBytes = null;
                            _staticGazeResults = [];
                            _staticImageSize = Size.zero;
                          });
                        },
                        backgroundColor: const Color(0xFF161622).withOpacity(0.8),
                        foregroundColor: Colors.redAccent,
                        child: const Icon(Icons.clear),
                      ),
                    ],
                  ),
                ),
              // Stats Overlay inside Preview
              if (_staticImageBytes != null)
                Positioned(
                  bottom: 16,
                  left: 16,
                  right: 16,
                  child: _buildLatencyStatsCard(
                    pre: _preTimeMs,
                    det: _detTimeMs,
                    post: _postTimeMs,
                    gaze: _gazeTimeMs,
                    isLive: false,
                  ),
                ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildLatencyStatsCard({
    required double pre,
    required double det,
    required double post,
    required double gaze,
    required bool isLive,
  }) {
    final double total = pre + det + post + gaze;
    final double fps = total > 0 ? (1000.0 / total) : 0.0;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: const Color(0xFF161622).withOpacity(0.85),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white10),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Expanded(
            child: SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(
                children: [
                  _buildStatItem('Pre', '${pre.toStringAsFixed(1)}ms'),
                  _buildStatSeparator(),
                  if (!(_rawEyeCropMode && !isLive)) ...[
                    _buildStatItem('FaceDet', '${det.toStringAsFixed(1)}ms'),
                    _buildStatSeparator(),
                    _buildStatItem('NMS', '${post.toStringAsFixed(1)}ms'),
                    _buildStatSeparator(),
                  ],
                  _buildStatItem('Gaze', '${gaze.toStringAsFixed(1)}ms'),
                ],
              ),
            ),
          ),
          const SizedBox(width: 8),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.primary.withOpacity(0.2),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              isLive
                  ? '${fps.toStringAsFixed(1)} FPS'
                  : 'Total: ${total.toStringAsFixed(1)}ms',
              style: TextStyle(
                color: Theme.of(context).colorScheme.primary,
                fontWeight: FontWeight.bold,
                fontSize: 11,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatItem(String label, String value) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(label, style: const TextStyle(fontSize: 9, color: Colors.white38)),
        const SizedBox(height: 2),
        Text(value, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.white)),
      ],
    );
  }

  Widget _buildStatSeparator() {
    return Container(
      height: 16,
      width: 1,
      color: Colors.white12,
      margin: const EdgeInsets.symmetric(horizontal: 12),
    );
  }

  Widget _buildConfigurationPanel(bool isLandscape) {
    return Container(
      width: double.infinity,
      decoration: const BoxDecoration(
        color: Color(0xFF161622),
        border: Border(top: BorderSide(color: Colors.white10)),
      ),
      padding: const EdgeInsets.all(16),
      child: SafeArea(
        top: false,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            // Row 1: Accelerator selector & Toggle painters
            Row(
              children: [
                Expanded(
                  child: DropdownButtonFormField<HardwareAccelerator>(
                    value: _selectedAccelerator,
                    decoration: const InputDecoration(
                      labelText: 'Hardware Accelerator',
                      labelStyle: TextStyle(fontSize: 12, color: Colors.white70),
                      contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                      border: OutlineInputBorder(),
                    ),
                    items: HardwareAccelerator.values.map((acc) {
                      return DropdownMenuItem(
                        value: acc,
                        child: Text(acc.name.toUpperCase(), style: const TextStyle(fontSize: 13)),
                      );
                    }).toList(),
                    onChanged: _isModelLoading ? null : _changeAccelerator,
                  ),
                ),
                const SizedBox(width: 16),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Checkbox(
                          value: _drawFaces,
                          visualDensity: VisualDensity.compact,
                          activeColor: Theme.of(context).colorScheme.primary,
                          onChanged: (val) {
                            if (val != null) setState(() => _drawFaces = val);
                          },
                        ),
                        const Text('Draw Face Bounding Box', style: TextStyle(fontSize: 11)),
                      ],
                    ),
                    Row(
                      children: [
                        Checkbox(
                          value: _drawEyes,
                          visualDensity: VisualDensity.compact,
                          activeColor: Theme.of(context).colorScheme.primary,
                          onChanged: (val) {
                            if (val != null) setState(() => _drawEyes = val);
                          },
                        ),
                        const Text('Draw Eye Crops Bounding Box', style: TextStyle(fontSize: 11)),
                      ],
                    ),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 12),
            
            // Slider configuration (Expands if clicked or just small layout)
            ExpansionTile(
              title: const Text('Fine-tuning Options (Face/Eye thresholds)', style: TextStyle(fontSize: 12, color: Colors.white70)),
              dense: true,
              tilePadding: EdgeInsets.zero,
              childrenPadding: EdgeInsets.zero,
              children: [
                // Face Score Threshold Slider
                Row(
                  children: [
                    const SizedBox(width: 100, child: Text('Face Score:', style: TextStyle(fontSize: 11, color: Colors.white54))),
                    Expanded(
                      child: Slider(
                        value: _scoreThreshold,
                        min: 0.5,
                        max: 0.99,
                        divisions: 49,
                        label: _scoreThreshold.toStringAsFixed(2),
                        onChanged: (val) {
                          setState(() => _scoreThreshold = val);
                          if (_currentTabIndex == 1 && _staticImageBytes != null) {
                            _processStaticImage(_staticImageBytes!);
                          }
                        },
                      ),
                    ),
                    Text(_scoreThreshold.toStringAsFixed(2), style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold)),
                  ],
                ),
                // NMS Threshold Slider
                Row(
                  children: [
                    const SizedBox(width: 100, child: Text('NMS Overlap:', style: TextStyle(fontSize: 11, color: Colors.white54))),
                    Expanded(
                      child: Slider(
                        value: _nmsThreshold,
                        min: 0.1,
                        max: 0.9,
                        divisions: 80,
                        label: _nmsThreshold.toStringAsFixed(2),
                        onChanged: (val) {
                          setState(() => _nmsThreshold = val);
                          if (_currentTabIndex == 1 && _staticImageBytes != null) {
                            _processStaticImage(_staticImageBytes!);
                          }
                        },
                      ),
                    ),
                    Text(_nmsThreshold.toStringAsFixed(2), style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold)),
                  ],
                ),
                // Eye Crop Scale Slider
                Row(
                  children: [
                    const SizedBox(width: 100, child: Text('Eye Crop Scale:', style: TextStyle(fontSize: 11, color: Colors.white54))),
                    Expanded(
                      child: Slider(
                        value: _eyeCropScale,
                        min: 0.2,
                        max: 0.6,
                        divisions: 40,
                        label: _eyeCropScale.toStringAsFixed(2),
                        onChanged: (val) {
                          setState(() => _eyeCropScale = val);
                          if (_currentTabIndex == 1 && _staticImageBytes != null) {
                            _processStaticImage(_staticImageBytes!);
                          }
                        },
                      ),
                    ),
                    Text(_eyeCropScale.toStringAsFixed(2), style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold)),
                  ],
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
