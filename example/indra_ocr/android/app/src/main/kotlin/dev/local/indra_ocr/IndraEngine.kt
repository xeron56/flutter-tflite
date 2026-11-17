// Android port of Indra preprocessing and greedy decoding (Apache-2.0).
// Source: datalab-to/indra e20355319b5aa5705a876a8d2b40aca606d7df5f.
package dev.local.indra_ocr

import ai.onnxruntime.*
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.exifinterface.media.ExifInterface
import org.json.JSONObject
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.text.Normalizer
import java.util.EnumSet
import kotlin.math.*

data class EngineConfig(val backend: String = "cpu", val threads: Int = 4) {
    init { require(backend in listOf("cpu", "xnnpack", "nnapi") && threads in 1..8) }
    val label get() = when (backend) {
        "nnapi" -> "NNAPI requested + CPU fallback"
        "xnnpack" -> "XNNPACK · $threads threads + CPU fallback"
        else -> "CPU · $threads threads"
    }
}

data class TextLine(val box: FloatArray, var text: String = "", var confidence: Float = 0f)
data class Crop(val pixels: FloatArray, val width: Int, val height: Int)

/** Confined to one worker thread: no inference or pixel work on Flutter's UI thread. */
class IndraEngine(private val context: Context, val config: EngineConfig) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val sessions = mutableMapOf<String, OrtSession>()
    private val modelDir = File(context.filesDir, "indra-v1.1.0-greedy-v1")
    private val meta: JSONObject
    private val vocabs = mutableMapOf<String, List<String>>()
    private val recHeight: Int
    private val recWindow: Int
    private val groups: List<String>
    private val debug = (context.applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) != 0

    init {
        check(OpenCVLoader.initLocal()) { "OpenCV could not initialize" }
        Core.setNumThreads(config.threads)
        modelDir.mkdirs()
        meta = JSONObject(context.assets.open("models/meta.json").bufferedReader().use { it.readText() })
        recHeight = meta.getInt("rec_height")
        recWindow = meta.optInt("rec_window", 1)
        groups = meta.getJSONArray("script_groups").let { a -> List(a.length()) { a.getString(it) } }
        for (group in groups) {
            val key = if (group == "base") "vocab" else "vocab_$group"
            val chars = JSONObject(meta.getString(key)).getJSONArray("chars")
            vocabs[group] = List(chars.length()) { chars.getString(it) }
        }
    }

    private fun session(name: String): OrtSession = sessions.getOrPut(name) {
        val file = File(modelDir, name)
        // The versioned directory is immutable. Copy atomically, then let ORT read from disk.
        if (!file.exists()) {
            val temp = File(modelDir, "$name.part")
            context.assets.open("models/$name").use { input -> temp.outputStream().use { input.copyTo(it) } }
            check(temp.renameTo(file)) { "Cannot store model $name" }
        }
        OrtSession.SessionOptions().use { options ->
            options.setIntraOpNumThreads(if (config.backend == "xnnpack") 1 else config.threads)
            options.setInterOpNumThreads(1)
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            // Upstream found that the arena grows badly with variable-width line batches.
            options.setCPUArenaAllocator(false)
            options.setMemoryPatternOptimization(false)
            options.addConfigEntry("session.intra_op.allow_spinning", "0")
            options.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING)
            when (config.backend) {
                "nnapi" -> options.addNnapi(EnumSet.of(NNAPIFlags.CPU_DISABLED))
                "xnnpack" -> options.addXnnpack(mapOf("intra_op_num_threads" to config.threads.toString()))
            }
            if (debug) Log.d("IndraOCR", "Load $name: ${config.label}")
            env.createSession(file.absolutePath, options)
        }
    }

    private fun run(name: String, crop: Crop, block: (OrtSession.Result) -> Unit) {
        OnnxTensor.createTensor(env, FloatBuffer.wrap(crop.pixels), longArrayOf(1, 1, crop.height.toLong(), crop.width.toLong())).use { x ->
            session(name).run(mapOf("x" to x)).use(block)
        }
    }

    private fun ink(gray: Mat): FloatArray {
        val f = Mat()
        try {
            gray.convertTo(f, CvType.CV_32F, -1.0 / 255, 1.0)
            return FloatArray(f.rows() * f.cols()).also { f.get(0, 0, it) }
        } finally { f.release() }
    }

    private fun pageInput(gray: Mat, side: Int, square: Boolean = false): Pair<Crop, Double> {
        val scale = side.toDouble() / max(gray.rows(), gray.cols())
        val w = max(1, (gray.cols() * scale).roundToInt())
        val h = max(1, (gray.rows() * scale).roundToInt())
        val small = Mat()
        try {
            Imgproc.resize(gray, small, Size(w.toDouble(), h.toDouble()), 0.0, 0.0,
                if (scale < 1) Imgproc.INTER_AREA else Imgproc.INTER_LINEAR)
            val pw = if (square) side else (w + 31) / 32 * 32
            val ph = if (square) side else (h + 31) / 32 * 32
            val src = ink(small)
            val data = FloatArray(pw * ph)
            for (y in 0 until h) src.copyInto(data, y * pw, y * w, (y + 1) * w)
            return Crop(data, pw, ph) to scale
        } finally { small.release() }
    }

    private fun decodeDetector(data: FloatArray, width: Int, height: Int, scale: Double, gray: Mat): List<TextLine> {
        val count = width * height
        val seen = BooleanArray(count)
        val queue = IntArray(count)
        val lines = mutableListOf<TextLine>()
        for (start in 0 until count) {
            if (seen[start] || data[start] <= 0f) continue
            var head = 0; var tail = 1
            queue[0] = start; seen[start] = true
            var minX = width; var maxX = 0
            var top = 0.0; var bottom = 0.0
            while (head < tail) {
                val p = queue[head++]; val x = p % width; val y = p / width
                minX = min(minX, x); maxX = max(maxX, x)
                top += (y + 0.5) * 2 - data[count + p]
                bottom += (y + 0.5) * 2 + data[2 * count + p]
                fun visit(q: Int) {
                    if (!seen[q] && data[q] > 0f) { seen[q] = true; queue[tail++] = q }
                }
                if (x > 0) visit(p - 1)
                if (x + 1 < width) visit(p + 1)
                if (y > 0) visit(p - width)
                if (y + 1 < height) visit(p + width)
            }
            if (tail < 2) continue
            val y0 = top / tail; val y1 = bottom / tail
            val ex = 0.15 * max(1.0, y1 - y0)
            val box = floatArrayOf(((minX * 2 - ex) / scale).toFloat().coerceIn(0f, gray.cols().toFloat()),
                (y0 / scale).toFloat().coerceIn(0f, gray.rows().toFloat()),
                (((maxX + 1) * 2 + ex) / scale).toFloat().coerceIn(0f, gray.cols().toFloat()),
                (y1 / scale).toFloat().coerceIn(0f, gray.rows().toFloat()))
            if (box[2] > box[0] && box[3] > box[1]) lines.add(TextLine(box))
        }
        return lines
    }

    private fun detectAt(gray: Mat, side: Int): List<TextLine> {
        val (input, scale) = pageInput(gray, side)
        var lines: List<TextLine> = emptyList()
        run("det.onnx", input) { result ->
            val tensor = result[0] as OnnxTensor
            val shape = tensor.info.shape
            val b = tensor.floatBuffer
            val data = FloatArray(b.remaining()); b.get(data)
            lines = decodeDetector(data, shape[3].toInt(), shape[2].toInt(), scale, gray)
        }
        return lines
    }

    private fun isDark(gray: Mat, b: FloatArray): Boolean {
        val x0 = b[0].toInt().coerceIn(0, gray.cols() - 1)
        val y0 = b[1].toInt().coerceIn(0, gray.rows() - 1)
        val x1 = (b[2].toInt() + 1).coerceIn(x0 + 1, gray.cols())
        val y1 = (b[3].toInt() + 1).coerceIn(y0 + 1, gray.rows())
        val roi = gray.submat(y0, y1, x0, x1)
        val pixels = ByteArray((x1 - x0) * (y1 - y0))
        try { roi.get(0, 0, pixels) } finally { roi.release() }
        val hist = IntArray(256)
        for (p in pixels) hist[p.toInt() and 255]++
        fun percentile(q: Double): Double {
            val pos = q * (pixels.size - 1); val low = floor(pos).toInt(); val high = ceil(pos).toInt()
            var cumulative = 0; var a = -1; var bval = 255
            for (i in hist.indices) {
                cumulative += hist[i]
                if (a < 0 && cumulative > low) a = i
                if (cumulative > high) { bval = i; break }
            }
            return a + (bval - a) * (pos - low)
        }
        val lo = percentile(0.02); val mid = percentile(0.5); val hi = percentile(0.98)
        return hi - mid > 2 * (mid - lo) + 8
    }

    private fun crop(gray: Mat, b: FloatArray, window: Int = 1, dark: Boolean): Crop {
        val lineH = max(1f, b[3] - b[1])
        val padY = (window * 1.4 - 1) / 2
        val x0 = floor(b[0] - 0.3 * lineH).toInt(); val x1 = ceil(b[2] + 0.3 * lineH).toInt()
        val y0 = floor(b[1] - padY * lineH).toInt(); val y1 = ceil(b[3] + padY * lineH).toInt()
        val h = recHeight * window
        val w = ((x1 - x0).toDouble() * h / max(1, y1 - y0)).roundToInt().coerceIn(8, 4096)
        val canvas = Mat(max(1, y1 - y0), max(1, x1 - x0), CvType.CV_8UC1, Scalar(if (dark) 0.0 else 255.0))
        val small = Mat()
        try {
            val sx0 = max(0, x0); val sy0 = max(0, y0)
            val sx1 = min(gray.cols(), x1); val sy1 = min(gray.rows(), y1)
            if (sx1 > sx0 && sy1 > sy0) {
                val src = gray.submat(sy0, sy1, sx0, sx1)
                val dst = canvas.submat(sy0 - y0, sy1 - y0, sx0 - x0, sx1 - x0)
                try { src.copyTo(dst) } finally { src.release(); dst.release() }
            }
            Imgproc.resize(canvas, small, Size(w.toDouble(), h.toDouble()), 0.0, 0.0,
                if (canvas.rows() > h) Imgproc.INTER_AREA else Imgproc.INTER_LINEAR)
            val data = ink(small)
            if (dark) for (i in data.indices) data[i] = 1f - data[i]
            return Crop(data, w, h)
        } finally { canvas.release(); small.release() }
    }

    private fun batches(indices: List<Int>, crops: List<Crop>): List<List<Int>> {
        val batches = mutableListOf<MutableList<Int>>()
        for (i in indices.sortedBy { crops[it].width }) {
            val last = batches.lastOrNull()
            val w = max(64, crops[i].width)
            if (last != null && last.size < 16 && w <= 1.3 * max(64, crops[last[0]].width) && (last.size + 1) * w <= 8192) last.add(i)
            else batches.add(mutableListOf(i))
        }
        return batches
    }

    private fun runBatch(name: String, indices: List<Int>, crops: List<Crop>, outputs: Set<String>? = null, block: (OrtSession.Result) -> Unit) {
        val height = crops[indices[0]].height
        val width = (indices.maxOf { crops[it].width } + 7) / 8 * 8
        val data = FloatArray(indices.size * height * width)
        val widths = LongArray(indices.size)
        for ((k, i) in indices.withIndex()) {
            val c = crops[i]; widths[k] = c.width.toLong()
            for (y in 0 until height) c.pixels.copyInto(data, (k * height + y) * width, y * c.width, (y + 1) * c.width)
        }
        OnnxTensor.createTensor(env, FloatBuffer.wrap(data), longArrayOf(indices.size.toLong(), 1, height.toLong(), width.toLong())).use { x ->
            OnnxTensor.createTensor(env, LongBuffer.wrap(widths), longArrayOf(indices.size.toLong())).use { w ->
                val s = session(name)
                val inputs = mapOf("x" to x, "widths" to w)
                (if (outputs == null) s.run(inputs) else s.run(inputs, outputs)).use(block)
            }
        }
    }

    private fun recognize(gray: Mat, lines: List<TextLine>, vertical: Boolean) {
        if (lines.isEmpty()) return
        val dark = lines.map { isDark(gray, it.box) }
        val crops = lines.mapIndexed { i, l -> crop(gray, l.box, dark = dark[i]) }
        val route = Array(lines.size) { if (vertical) "cjk" else "base" }
        val sure = BooleanArray(lines.size) { true }
        if (!vertical) for (idx in batches(lines.indices.toList(), crops)) {
            runBatch("script.onnx", idx, crops) { result ->
                val tensor = result[0] as OnnxTensor
                val probs = tensor.floatBuffer
                for ((k, j) in idx.withIndex()) {
                    val best = groups.indices.maxBy { probs[k * groups.size + it] }
                    route[j] = groups[best]; sure[j] = probs[k * groups.size + best] > 0.95f
                }
            }
        }
        for (group in groups) {
            val selected = lines.indices.filter { route[it] == group || (group == "base" && !sure[it]) }
            if (selected.isEmpty()) continue
            // Only Latin's recognizer uses the tall context window.
            val src = if (group == "base" && recWindow > 1) lines.mapIndexed { i, l ->
                if (i in selected) crop(gray, l.box, recWindow, dark[i]) else crops[i]
            } else crops
            val name = if (group == "base") "rec.onnx" else "rec_$group.onnx"
            for (idx in batches(selected, src)) runBatch(name, idx, src, setOf("ids", "conf", "lengths")) { result ->
                val idsTensor = result.get("ids").get() as OnnxTensor
                val ids = idsTensor.longBuffer
                val probabilities = (result.get("conf").get() as OnnxTensor).floatBuffer
                val lengths = (result.get("lengths").get() as OnnxTensor).longBuffer
                val steps = idsTensor.info.shape[1].toInt()
                val vocab = vocabs.getValue(group)
                for ((k, j) in idx.withIndex()) {
                    var previous = -1L; var sum = 0f; var count = 0
                    val text = StringBuilder()
                    for (t in 0 until min(lengths[k].toInt(), steps)) {
                        val id = ids[k * steps + t]
                        if (id != 0L && id != previous && id <= vocab.size) {
                            text.append(vocab[id.toInt() - 1]); sum += probabilities[k * steps + t]; count++
                        }
                        previous = id
                    }
                    val confidence = if (count > 0) sum / count else 0f
                    if (confidence >= lines[j].confidence) {
                        lines[j].confidence = confidence
                        lines[j].text = if (confidence < 0.5f) "" else RtlText.flip(Normalizer.normalize(text.toString(), Normalizer.Form.NFC)).trim()
                    }
                }
            }
        }
    }

    private fun readGray(path: String): Mat {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, bounds)
        require(bounds.outWidth > 0 && bounds.outHeight > 0) { "This file is not a readable image." }
        val options = BitmapFactory.Options()
        options.inSampleSize = 1
        while (max(bounds.outWidth, bounds.outHeight) / options.inSampleSize > 4096) options.inSampleSize *= 2
        var bitmap = requireNotNull(BitmapFactory.decodeFile(path, options)) { "Cannot decode image" }
        val orientation = ExifInterface(path).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.setScale(-1f, 1f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.setRotate(180f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.setScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> { matrix.setRotate(90f); matrix.postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.setRotate(90f)
            ExifInterface.ORIENTATION_TRANSVERSE -> { matrix.setRotate(-90f); matrix.postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.setRotate(-90f)
        }
        if (!matrix.isIdentity) {
            val rotated = android.graphics.Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            if (rotated !== bitmap) bitmap.recycle()
            bitmap = rotated
        }
        val rgba = Mat(); val gray = Mat()
        try { Utils.bitmapToMat(bitmap, rgba); Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY) }
        finally { rgba.release(); bitmap.recycle() }
        return gray
    }

    fun process(path: String, quality: Boolean = false): Map<String, Any> {
        val start = SystemClock.elapsedRealtimeNanos()
        val gray = readGray(path)
        val times = linkedMapOf<String, Double>()
        fun elapsed(s: Long) = (SystemClock.elapsedRealtimeNanos() - s) / 1e6
        times["decode"] = elapsed(start)
        var rotation = 0; var vertical = false
        try {
            val orientStart = SystemClock.elapsedRealtimeNanos()
            run("orient.onnx", pageInput(gray, 768, true).first) { r ->
                val p = (r[0] as OnnxTensor).floatBuffer
                val k = (0 until p.remaining()).maxBy { p[it] }
                if (p[k] > 0.95f) {
                    vertical = k == 4
                    rotation = if (vertical) 270 else k * 90
                    when (rotation) {
                        90 -> Core.rotate(gray, gray, Core.ROTATE_90_CLOCKWISE)
                        180 -> Core.rotate(gray, gray, Core.ROTATE_180)
                        270 -> Core.rotate(gray, gray, Core.ROTATE_90_COUNTERCLOCKWISE)
                    }
                }
            }
            times["orientation"] = elapsed(orientStart)
            val detStart = SystemClock.elapsedRealtimeNanos()
            val side = meta.getInt("det_long_side")
            var lines = detectAt(gray, side)
            if (quality && lines.size >= 8) {
                val heights = lines.map { (it.box[3] - it.box[1]) * side / max(gray.rows(), gray.cols()) }.sorted()
                val median = heights[heights.size / 2]
                if (median < 9f) {
                    val larger = (min(2048.0, side * 12.0 / max(1f, median)) / 32).toInt() * 32
                    if (larger > side) lines = detectAt(gray, larger)
                }
            }
            times["detection"] = elapsed(detStart)
            val recStart = SystemClock.elapsedRealtimeNanos()
            // Guard pathological/noisy inputs rather than allocating unbounded batches.
            require(lines.size <= 1500) { "Too many text regions. Crop the document and try again." }
            recognize(gray, lines, vertical)
            times["recognition"] = elapsed(recStart)
            val ordered = ReadingOrder.sort(lines)
            times["total"] = elapsed(start)
            if (debug) Log.d("IndraOCR", "${config.label}: ${lines.size} lines, timings=$times")
            return mapOf("text" to ordered.filter { it.text.isNotEmpty() }.joinToString("\n") { it.text },
                "lines" to ordered.map { mapOf("box" to it.box.toList(), "text" to it.text, "confidence" to it.confidence.toDouble()) },
                "width" to gray.cols(), "height" to gray.rows(), "rotation" to rotation,
                "timings" to times, "backend" to config.label,
                "threads" to config.threads, "quality" to quality)
        } finally { gray.release() }
    }

    override fun close() { sessions.values.forEach { it.close() }; sessions.clear() }
}

/** Recursive whitespace cuts: columns before rows when a clear gutter exists. */
object ReadingOrder {
    fun sort(lines: List<TextLine>, depth: Int = 0): List<TextLine> {
        if (lines.size < 2 || depth >= 30) return lines.sortedWith(compareBy({ it.box[1] }, { it.box[0] }))
        val medianHeight = lines.map { it.box[3] - it.box[1] }.sorted()[lines.size / 2]
        fun split(axis: Int, threshold: Float): Pair<List<TextLine>, List<TextLine>>? {
            val sorted = lines.sortedBy { it.box[axis] }
            var edge = sorted[0].box[axis + 2]
            var bestGap = threshold; var cut = -1
            for (i in 1 until sorted.size) {
                val gap = sorted[i].box[axis] - edge
                if (gap > bestGap) { bestGap = gap; cut = i }
                edge = max(edge, sorted[i].box[axis + 2])
            }
            return if (cut > 0) sorted.take(cut) to sorted.drop(cut) else null
        }
        val parts = split(0, medianHeight * 0.7f) ?: split(1, medianHeight * 0.15f)
        return if (parts == null) lines.sortedWith(compareBy({ it.box[1] }, { it.box[0] }))
            else sort(parts.first, depth + 1) + sort(parts.second, depth + 1)
    }
}

/** Indra's visual-to-logical bidi subset, retaining combining marks and LTR runs. */
object RtlText {
    private fun direction(c: Int) = Character.getDirectionality(c)
    private fun rtl(c: Int) = direction(c) in listOf(Character.DIRECTIONALITY_RIGHT_TO_LEFT, Character.DIRECTIONALITY_RIGHT_TO_LEFT_ARABIC)
    private fun kind(c: Int): Char = when {
        rtl(c) -> 'R'
        direction(c) in listOf(Character.DIRECTIONALITY_LEFT_TO_RIGHT, Character.DIRECTIONALITY_EUROPEAN_NUMBER, Character.DIRECTIONALITY_ARABIC_NUMBER) -> 'L'
        else -> 'N'
    }
    private fun mirror(s: String): String {
        val i = "()[]{}<>«»‹›".indexOf(s[0])
        return if (i < 0) s else ")(][}{><»«›‹"[i] + s.substring(1)
    }
    fun flip(text: String): String {
        val points = text.codePoints().toArray()
        if (points.none { rtl(it) }) return text
        val clusters = mutableListOf<String>()
        for (p in points) {
            val s = String(Character.toChars(p))
            if (clusters.isNotEmpty() && Character.getType(p) in listOf(Character.NON_SPACING_MARK.toInt(), Character.ENCLOSING_MARK.toInt()))
                clusters[clusters.lastIndex] += s
            else clusters.add(s)
        }
        val kinds = clusters.map { kind(it.codePointAt(0)) }
        val baseRtl = points.count { rtl(it) } > points.count { direction(it) == Character.DIRECTIONALITY_LEFT_TO_RIGHT }
        val strong = if (baseRtl) 'L' else 'R'
        val stop = if (baseRtl) 'R' else 'L'
        val spans = mutableListOf<Pair<Int, Int>>()
        var start = -1; var last = -1
        for (i in 0..kinds.size) {
            val k = kinds.getOrElse(i) { '#' }
            if (k == strong) { if (start < 0) start = i; last = i }
            else if (k == stop || k == '#') { if (start >= 0) spans.add(start to last + 1); start = -1 }
        }
        if (baseRtl) {
            val units = mutableListOf<String>(); var i = 0
            for ((a, b) in spans + listOf(clusters.size to clusters.size)) {
                for (j in i until a) units.add(mirror(clusters[j]))
                if (a < b) units.add(clusters.subList(a, b).joinToString(""))
                i = b
            }
            return units.reversed().joinToString("")
        }
        val out = clusters.toMutableList()
        for ((a, b) in spans) for (j in a until b) out[j] = mirror(clusters[b - 1 - (j - a)])
        return out.joinToString("")
    }
}
