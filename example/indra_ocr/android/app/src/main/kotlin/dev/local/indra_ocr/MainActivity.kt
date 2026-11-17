package dev.local.indra_ocr

import android.app.ActivityManager
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Build
import android.util.Log
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : FlutterActivity() {
    private val worker = Executors.newSingleThreadExecutor()
    private val busy = AtomicBoolean(false)
    private var engine: IndraEngine? = null
    private var exportResult: MethodChannel.Result? = null
    private var exportText: String? = null
    private val prefs by lazy { getSharedPreferences("indra", Context.MODE_PRIVATE) }
    private fun savedConfig() = EngineConfig(prefs.getString("backend", "cpu") ?: "cpu", prefs.getInt("threads", if (Build.VERSION.SDK_INT >= 31 && Build.SOC_MODEL == "SM8750") 8 else 4))
    private fun current(): IndraEngine = engine ?: IndraEngine(applicationContext, savedConfig()).also { engine = it }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "dev.local.indra/ocr").setMethodCallHandler { call, result ->
            if (call.method == "exportMarkdown") {
                if (exportResult != null) {
                    result.error("busy", "A save dialog is already open.", null)
                } else {
                    val text = call.argument<String>("markdown")
                    if (text == null) {
                        result.error("invalid", "Markdown text is required.", null)
                    } else {
                        exportResult = result; exportText = text
                        try {
                            startActivityForResult(Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                                addCategory(Intent.CATEGORY_OPENABLE)
                                type = "text/markdown"
                                putExtra(Intent.EXTRA_TITLE, "indra-document.md")
                            }, 7301)
                        } catch (e: Exception) {
                            exportResult = null; exportText = null
                            result.error("export_error", e.message, null)
                        }
                    }
                }
                return@setMethodCallHandler
            }
            if (!busy.compareAndSet(false, true)) { result.error("busy", "OCR is already running.", null); return@setMethodCallHandler }
            worker.execute {
                try {
                    val value: Any? = when (call.method) {
                        "info" -> info()
                        "sample" -> sample(call.argument<String>("kind") ?: "english")
                        "recognize" -> current().process(requireNotNull(call.argument<String>("path")), call.argument<Boolean>("quality") ?: false)
                        "configure" -> {
                            val c = EngineConfig(call.argument<String>("backend") ?: "cpu", call.argument<Int>("threads") ?: 4)
                            engine?.close(); engine = null
                            engine = IndraEngine(applicationContext, c)
                            prefs.edit().putString("backend", c.backend).putInt("threads", c.threads).apply()
                            info()
                        }
                        "benchmark" -> benchmark(requireNotNull(call.argument<String>("path")), call.argument<Boolean>("quality") ?: false)
                        else -> null
                    }
                    runOnUiThread { busy.set(false); if (value == null) result.notImplemented() else result.success(value) }
                } catch (e: Exception) {
                    Log.e("IndraOCR", "${call.method} failed", e)
                    runOnUiThread { busy.set(false); result.error("ocr_error", e.message ?: e.javaClass.simpleName, null) }
                }
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != 7301) return
        val reply = exportResult ?: return
        val text = exportText.orEmpty()
        exportResult = null; exportText = null
        if (resultCode != Activity.RESULT_OK) { reply.success(null); return }
        val uri = data?.data
        if (uri == null) { reply.error("export_error", "No file was selected.", null); return }
        worker.execute {
            try {
                val stream = contentResolver.openOutputStream(uri, "wt")
                    ?: error("Could not open the selected file.")
                stream.bufferedWriter(Charsets.UTF_8).use { it.write(text) }
                var name = "Markdown document"
                contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)?.use {
                    if (it.moveToFirst()) name = it.getString(0) ?: name
                }
                runOnUiThread { reply.success(name) }
            } catch (e: Exception) {
                Log.e("IndraOCR", "Markdown export failed", e)
                runOnUiThread { reply.error("export_error", "Could not save the file: ${e.message}", null) }
            }
        }
    }

    private fun info(): Map<String, Any> {
        val memory = ActivityManager.MemoryInfo()
        (getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(memory)
        val c = engine?.config ?: savedConfig()
        return mapOf("device" to "${Build.MANUFACTURER} ${Build.MODEL}",
            "soc" to if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE,
            "android" to Build.VERSION.RELEASE, "ramGb" to memory.totalMem / 1e9,
            "backend" to c.backend, "threads" to c.threads, "engine" to c.label)
    }

    private fun benchmark(path: String, quality: Boolean): Map<String, Any> {
        val original = savedConfig()
        engine?.close(); engine = null
        val candidates = listOf(EngineConfig("cpu", 1), EngineConfig("cpu", 2), EngineConfig("cpu", 4),
            EngineConfig("cpu", 6), EngineConfig("cpu", 8), EngineConfig("xnnpack", 4), EngineConfig("xnnpack", 6), EngineConfig("nnapi", 4))
        val rows = mutableListOf<Map<String, Any>>()
        var baseline: String? = null
        var winner = original; var best = Double.POSITIVE_INFINITY
        try {
            for (candidate in candidates) {
                var trial: IndraEngine? = null
                try {
                    trial = IndraEngine(applicationContext, candidate)
                    val cold = trial.process(path, quality)
                    val coldText = cold["text"] as String
                    if (baseline == null) baseline = coldText
                    val runs = List(3) { trial.process(path, quality) }
                    val elapsed = runs.map { (it["timings"] as Map<*, *>)["total"] as Double }.sorted()
                    val matches = coldText == baseline && runs.all { it["text"] == baseline } && !baseline.isNullOrBlank()
                    val median = elapsed[1]
                    rows.add(mapOf("backend" to candidate.backend, "threads" to candidate.threads, "label" to candidate.label,
                        "medianMs" to median, "minMs" to elapsed.first(), "maxMs" to elapsed.last(),
                        "coldMs" to ((cold["timings"] as Map<*, *>)["total"] as Double), "matchesCpuText" to matches))
                    if (matches && median < best) { best = median; winner = candidate }
                    Log.i("IndraOCR", "Benchmark ${candidate.label}: $median ms; sameText=$matches")
                } catch (e: Exception) {
                    rows.add(mapOf("backend" to candidate.backend, "threads" to candidate.threads, "label" to candidate.label,
                        "error" to (e.message ?: "Unsupported backend")))
                    Log.w("IndraOCR", "Benchmark unavailable: ${candidate.label}", e)
                } finally { trial?.close() }
            }
            require(best.isFinite()) { "No readable text found to tune. Choose a clear document." }
            prefs.edit().putString("backend", winner.backend).putInt("threads", winner.threads).apply()
            return mapOf("results" to rows, "selected" to winner.label, "backend" to winner.backend,
                "threads" to winner.threads, "medianMs" to best,
                "note" to "3 warm runs per candidate. Selected only identical text. NNAPI device assignment is not verified; CPU fallback may occur.")
        } finally {
            engine = IndraEngine(applicationContext, savedConfig())
        }
    }

    private fun sample(kind: String): String {
        val output = File(cacheDir, "indra-sample-$kind.png")
        if (kind == "scan") {
            assets.open("samples/scan.png").use { input -> output.outputStream().use { input.copyTo(it) } }
        } else {
            val bitmap = Bitmap.createBitmap(1400, 1000, Bitmap.Config.ARGB_8888)
            try {
                val canvas = Canvas(bitmap)
                canvas.drawColor(if (kind == "dark") Color.BLACK else Color.WHITE)
                val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    color = if (kind == "dark") Color.WHITE else Color.BLACK
                    textSize = 48f
                    typeface = android.graphics.Typeface.create("sans-serif", android.graphics.Typeface.NORMAL)
                }
                if (kind == "rotated") canvas.rotate(180f, 700f, 500f)
                if (kind != "blank") {
                    val lines = if (kind == "bengali") listOf("বাংলাদেশ একটি সুন্দর দেশ।", "আমি বাংলা ভাষায় কথা বলি।", "Indra reads text on your phone.")
                        else listOf("Indra OCR on Android", "Fast offline text recognition.", "Invoice number 2026. Total 125.50", "Your documents stay on your phone.")
                    lines.forEachIndexed { i, text -> canvas.drawText(text, 90f, 200f + i * 130f, paint) }
                }
                output.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
            } finally { bitmap.recycle() }
        }
        return output.absolutePath
    }

    override fun onDestroy() {
        exportResult?.error("cancelled", "The save dialog was interrupted. Please save again.", null)
        exportResult = null; exportText = null
        // Queue cleanup after any in-flight call to avoid closing tensors mid-inference.
        worker.execute { engine?.close(); engine = null }
        worker.shutdown()
        super.onDestroy()
    }
}
