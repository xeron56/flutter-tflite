import 'dart:math';
import 'package:flutter/material.dart';
import 'eyegaze_service.dart';

class GazePainter extends CustomPainter {
  final List<EyeGazeResult> gazeResults;
  final Size imageSize;
  final bool drawFaces;
  final bool drawEyes;

  GazePainter({
    required this.gazeResults,
    required this.imageSize,
    this.drawFaces = true,
    this.drawEyes = true,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (gazeResults.isEmpty || imageSize.width == 0 || imageSize.height == 0) return;

    final double scaleX = size.width / imageSize.width;
    final double scaleY = size.height / imageSize.height;

    // Paints
    final faceBoxPaint = Paint()
      ..color = const Color(0x66FFFFFF) // Translucent white for face box
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    final eyeBoxPaint = Paint()
      ..color = const Color(0xFF00E5FF) // Vibrant Cyan for eye crop boxes
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;

    final landmarkPaint = Paint()
      ..color = const Color(0xFFFFEB3B) // Yellow for eye centers
      ..style = PaintingStyle.fill;

    final arrowPaint = Paint()
      ..color = const Color(0xFFFF1744) // Bright Red for gaze arrow
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.5
      ..strokeCap = StrokeCap.round;

    final arrowHeadPaint = Paint()
      ..color = const Color(0xFFFF1744)
      ..style = PaintingStyle.fill;

    // Set of drawn faces to avoid drawing the face box twice (for left & right eyes of the same face)
    final Set<Rect> drawnFaces = {};

    for (final result in gazeResults) {
      // 1. Draw face bounding box if present and requested
      if (drawFaces && result.faceBox != null && !drawnFaces.contains(result.faceBox)) {
        final Rect scaledFaceBox = Rect.fromLTRB(
          result.faceBox!.left * scaleX,
          result.faceBox!.top * scaleY,
          result.faceBox!.right * scaleX,
          result.faceBox!.bottom * scaleY,
        );
        canvas.drawRRect(
          RRect.fromRectAndRadius(scaledFaceBox, const Radius.circular(8.0)),
          faceBoxPaint,
        );
        drawnFaces.add(result.faceBox!);
      }

      // 2. Draw eye bounding box if present and requested
      if (drawEyes && result.eyeBox != null) {
        final Rect scaledEyeBox = Rect.fromLTRB(
          result.eyeBox!.left * scaleX,
          result.eyeBox!.top * scaleY,
          result.eyeBox!.right * scaleX,
          result.eyeBox!.bottom * scaleY,
        );
        canvas.drawRRect(
          RRect.fromRectAndRadius(scaledEyeBox, const Radius.circular(4.0)),
          eyeBoxPaint,
        );
      }

      // 3. Draw eye center (landmark)
      final Offset scaledCenter = Offset(result.eyeCenter.dx * scaleX, result.eyeCenter.dy * scaleY);
      canvas.drawCircle(scaledCenter, 3.0, landmarkPaint);

      // 4. Draw gaze arrow
      // Calculate 2D direction vector from pitch and yaw
      // dx = -sin(yaw), dy = sin(pitch)
      final double pitchRad = result.pitchRad;
      final double yawRad = result.yawRad;

      double dx = -sin(yawRad);
      double dy = sin(pitchRad);

      // Normalize to unit length
      final double norm = sqrt(dx * dx + dy * dy);
      if (norm > 1e-6) {
        dx /= norm;
        dy /= norm;
      }

      // Arrow length should scale with face box width or image size.
      // If eyeBox is available, make it proportional to the scaled eyeBox width.
      final double arrowLength = result.eyeBox != null
          ? result.eyeBox!.width * 1.6 * scaleX
          : size.width * 0.15; // default fallback length

      final Offset arrowEnd = Offset(
        scaledCenter.dx + dx * arrowLength,
        scaledCenter.dy + dy * arrowLength,
      );

      // Draw the main line of the arrow
      canvas.drawLine(scaledCenter, arrowEnd, arrowPaint);

      // Draw the arrow head (a triangle at the end)
      final double angle = atan2(dy, dx);
      const double arrowHeadAngle = pi / 6; // 30 degrees
      final double arrowHeadLength = arrowLength * 0.25;

      final Path arrowHeadPath = Path();
      arrowHeadPath.moveTo(arrowEnd.dx, arrowEnd.dy);
      arrowHeadPath.lineTo(
        arrowEnd.dx - arrowHeadLength * cos(angle - arrowHeadAngle),
        arrowEnd.dy - arrowHeadLength * sin(angle - arrowHeadAngle),
      );
      arrowHeadPath.lineTo(
        arrowEnd.dx - arrowHeadLength * cos(angle + arrowHeadAngle),
        arrowEnd.dy - arrowHeadLength * sin(angle + arrowHeadAngle),
      );
      arrowHeadPath.close();

      canvas.drawPath(arrowHeadPath, arrowHeadPaint);
    }
  }

  @override
  bool shouldRepaint(covariant GazePainter oldDelegate) {
    return oldDelegate.gazeResults != gazeResults ||
        oldDelegate.imageSize != imageSize ||
        oldDelegate.drawFaces != drawFaces ||
        oldDelegate.drawEyes != drawEyes;
  }
}
