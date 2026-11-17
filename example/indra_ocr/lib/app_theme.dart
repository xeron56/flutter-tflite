import 'package:flutter/material.dart';

abstract final class AppColors {
  static const seed = Color(0xFF146C58);
}

abstract final class AppSpacing {
  static const sm = 8.0;
  static const md = 16.0;
  static const lg = 24.0;
}

abstract final class AppRadius {
  static const md = 16.0;
}

abstract final class AppTypography {
  static TextTheme apply(TextTheme base) => base.copyWith(
    headlineMedium: base.headlineMedium?.copyWith(fontWeight: FontWeight.w700),
    titleLarge: base.titleLarge?.copyWith(fontWeight: FontWeight.w600),
  );
}

abstract final class DocumentStyle {
  static const paper = Color(0xFFFFFEF9);
  static const maxWidth = 760.0;
  static const fontSize = 17.0;
  static const lineHeight = 1.75;
  static const pagePadding = 28.0;
}
