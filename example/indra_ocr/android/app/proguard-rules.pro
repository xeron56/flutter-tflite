# ORT's JNI bridge constructs TensorInfo / OnnxValue classes by their Java names.
# R8 cannot infer those references; retain the entire small Java binding surface.
-keep class ai.onnxruntime.** { *; }
# OpenCV also resolves Java types and members from native code.
-keep class org.opencv.** { *; }
