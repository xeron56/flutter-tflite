import 'dart:math';
import 'package:flutter/material.dart';
import 'sixd_repnet_service.dart';

class PosePainter extends CustomPainter {
  final List<FacePose> poses;
  final Size imageSize;

  PosePainter({
    required this.poses,
    required this.imageSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (poses.isEmpty || imageSize.width == 0 || imageSize.height == 0) return;

    final double scaleX = size.width / imageSize.width;
    final double scaleY = size.height / imageSize.height;

    // Paints definition
    final boxPaint = Paint()
      ..color = const Color(0xFF00E5FF) // Vibrant Cyan
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    final landmarkPaint = Paint()
      ..color = const Color(0xFFFFEB3B) // Yellow
      ..style = PaintingStyle.fill;

    final xPaint = Paint()
      ..color = const Color(0xFFFF1744) // Bright Red
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4.0
      ..strokeCap = StrokeCap.round;

    final yPaint = Paint()
      ..color = const Color(0xFF00E676) // Bright Green
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4.0
      ..strokeCap = StrokeCap.round;

    final zPaint = Paint()
      ..color = const Color(0xFF2979FF) // Bright Blue
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4.0
      ..strokeCap = StrokeCap.round;

    for (final pose in poses) {
      // Scale bounding box
      final Rect scaledBox = Rect.fromLTRB(
        pose.box.left * scaleX,
        pose.box.top * scaleY,
        pose.box.right * scaleX,
        pose.box.bottom * scaleY,
      );

      // Draw bounding box
      canvas.drawRRect(
        RRect.fromRectAndRadius(scaledBox, const Radius.circular(8.0)),
        boxPaint,
      );

      // Draw score text (if not fallback)
      if (pose.score < 1.0) {
        final textPainter = TextPainter(
          text: TextSpan(
            text: 'Face: ${(pose.score * 100).toStringAsFixed(1)}%',
            style: const TextStyle(
              color: Colors.black,
              fontSize: 10,
              fontWeight: FontWeight.bold,
            ),
          ),
          textDirection: TextDirection.ltr,
        );
        textPainter.layout();
        
        // Draw background for text
        final bgRect = Rect.fromLTWH(
          scaledBox.left,
          scaledBox.top - 16,
          textPainter.width + 8,
          16,
        );
        canvas.drawRRect(
          RRect.fromRectAndRadius(bgRect, const Radius.circular(4.0)),
          Paint()..color = const Color(0xFF00E5FF),
        );
        textPainter.paint(canvas, Offset(scaledBox.left + 4, scaledBox.top - 14));
      }

      // Draw landmarks
      for (final lm in pose.landmarks) {
        if (lm.x == 0.0 && lm.y == 0.0) continue; // Skip fallback landmarks
        canvas.drawCircle(
          Offset(lm.x * scaleX, lm.y * scaleY),
          3.0,
          landmarkPaint,
        );
      }

      // Draw 3D pose axes
      final double tdx = scaledBox.center.dx;
      final double tdy = scaledBox.center.dy;

      // Axis size is relative to bounding box width (50% of box width)
      final double axisSize = scaledBox.width * 0.5;

      final double p = pose.pitch * pi / 180.0;
      final double y = -(pose.yaw * pi / 180.0);
      final double r = pose.roll * pi / 180.0;

      // Calculate projection endpoints
      final double x1 = axisSize * (cos(y) * cos(r)) + tdx;
      final double y1 = axisSize * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + tdy;

      final double x2 = axisSize * (-cos(y) * sin(r)) + tdx;
      final double y2 = axisSize * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + tdy;

      final double x3 = axisSize * sin(y) + tdx;
      final double y3 = axisSize * (-cos(y) * sin(p)) + tdy;

      // Draw axes lines
      // X-axis (Red)
      canvas.drawLine(Offset(tdx, tdy), Offset(x1, y1), xPaint);
      // Y-axis (Green)
      canvas.drawLine(Offset(tdx, tdy), Offset(x2, y2), yPaint);
      // Z-axis (Blue)
      canvas.drawLine(Offset(tdx, tdy), Offset(x3, y3), zPaint);

      // Draw center dot
      canvas.drawCircle(
        Offset(tdx, tdy),
        5.0,
        Paint()
          ..color = Colors.white
          ..style = PaintingStyle.fill,
      );
      canvas.drawCircle(
        Offset(tdx, tdy),
        5.0,
        Paint()
          ..color = Colors.black
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.0,
      );
    }
  }

  @override
  bool shouldRepaint(covariant PosePainter oldDelegate) {
    return oldDelegate.poses != poses || oldDelegate.imageSize != imageSize;
  }
}
