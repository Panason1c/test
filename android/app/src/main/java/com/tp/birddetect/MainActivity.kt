package com.tp.birddetect

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.nio.FloatBuffer

/** Détection de maladies de plantes (PlantDoc, YOLOv8n fine-tuné, 416 px, 29 classes).
 *  L'utilisateur choisit une photo de feuille ; l'app dessine les boîtes (vert = saine, rouge = malade)
 *  et indique la maladie détectée. Quatre versions du modèle (FP32, FP16, INT8 dynamique, INT8 statique). */
class MainActivity : Activity() {
    private lateinit var imageView: ImageView
    private lateinit var info: TextView
    private lateinit var modelLabel: TextView
    private var lastUri: Uri? = null
    private val env = OrtEnvironment.getEnvironment()

    private val size = 416
    private val maxBoxes = 10

    /** assets/classes.txt : une ligne par classe, "H|nom" (saine) ou "D|nom" (malade), dans l'ordre du modèle. */
    private val classNames: List<String> by lazy {
        assets.open("classes.txt").bufferedReader().readLines().filter { it.isNotBlank() }.map { it.substring(2) }
    }
    private val classHealthy: List<Boolean> by lazy {
        assets.open("classes.txt").bufferedReader().readLines().filter { it.isNotBlank() }.map { it.startsWith("H|") }
    }

    private val models = listOf(
        "plantdoc_fp32" to "PlantDoc FP32",
        "plantdoc_fp16" to "PlantDoc FP16",
        "plantdoc_int8_dynamic" to "PlantDoc INT8 dynamique",
        "plantdoc_int8_static" to "PlantDoc INT8 statique",
    )
    @Volatile private var selected = models.lastIndex
    private var session: OrtSession? = null
    private var loadedIndex = -1

    @Synchronized
    private fun currentSession(): OrtSession {
        val idx = selected
        if (session == null || loadedIndex != idx) {
            session?.close()
            val bytes = assets.open("models/${models[idx].first}.onnx").readBytes()
            session = env.createSession(bytes, OrtSession.SessionOptions().apply { setIntraOpNumThreads(4) })
            loadedIndex = idx
        }
        return session!!
    }

    private class Det(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val score: Float, val cls: Int)
    private class Prepared(val data: FloatArray, val ratio: Float, val left: Int, val top: Int)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24, 24, 24, 24) }
        val pick = Button(this).apply { text = "Choisir une photo de feuille"; setOnClickListener { pickImage() } }
        val bench = Button(this).apply { text = "Benchmark de latence"; setOnClickListener { runBench() } }
        modelLabel = TextView(this).apply { textSize = 15f; text = "Modèle : ${models[selected].second}" }

        // Une rangée de boutons, un par version ; le bouton de la version active est grisé.
        val modelButtons = ArrayList<Button>()
        fun selectModel(i: Int) {
            selected = i
            modelLabel.text = "Modèle : ${models[i].second}"
            modelButtons.forEachIndexed { k, b -> b.isEnabled = k != i }
            lastUri?.let { u ->
                info.text = "Analyse en cours..."
                Thread { runCatching { analyze(u) }.onFailure { e -> runOnUiThread { info.text = "Erreur: $e" } } }.start()
            }
        }
        val labels = listOf("FP32", "FP16", "INT8 dyn.", "INT8 stat.")
        val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        labels.forEachIndexed { idx, label ->
            val b = Button(this).apply {
                text = label; textSize = 12f; isAllCaps = false; isEnabled = idx != selected
                setOnClickListener { selectModel(idx) }
            }
            modelButtons.add(b)
            row.addView(b, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
        }

        imageView = ImageView(this).apply { adjustViewBounds = true; scaleType = ImageView.ScaleType.FIT_CENTER }
        info = TextView(this).apply { textSize = 16f; text = "Choisis une photo de feuille pour lancer le diagnostic." }
        root.addView(pick, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(bench, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(modelLabel, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(row, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(imageView, LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f))
        root.addView(ScrollView(this).apply { addView(info) }, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        setContentView(root)
    }

    private fun pickImage() {
        val i = Intent(Intent.ACTION_GET_CONTENT).apply { type = "image/*"; addCategory(Intent.CATEGORY_OPENABLE) }
        startActivityForResult(Intent.createChooser(i, "Photo"), 1)
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        val uri = data?.data
        if (requestCode == 1 && resultCode == RESULT_OK && uri != null) {
            info.text = "Analyse en cours..."
            Thread {
                runCatching { analyze(uri) }.onFailure { e -> runOnUiThread { info.text = "Erreur: $e" } }
            }.start()
        }
    }

    private fun loadBitmap(uri: Uri): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it, null, bounds) }
        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / sample > 2048) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        return contentResolver.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it, null, opts) }!!
    }

    private fun letterbox(src: Bitmap): Prepared {
        val r = size.toFloat() / maxOf(src.width, src.height)
        val w = (src.width * r).toInt(); val h = (src.height * r).toInt()
        val scaled = Bitmap.createScaledBitmap(src, w, h, true)
        val canvasBmp = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val left = (size - w) / 2; val top = (size - h) / 2
        Canvas(canvasBmp).apply { drawColor(Color.rgb(114, 114, 114)); drawBitmap(scaled, left.toFloat(), top.toFloat(), null) }
        val px = IntArray(size * size); canvasBmp.getPixels(px, 0, size, 0, 0, size, size)
        val out = FloatArray(3 * size * size)
        for (i in px.indices) {
            out[i] = ((px[i] shr 16) and 255) / 255f
            out[size * size + i] = ((px[i] shr 8) and 255) / 255f
            out[2 * size * size + i] = (px[i] and 255) / 255f
        }
        return Prepared(out, r, left, top)
    }

    /** Sortie YOLOv8 [4+nc][N] : cx, cy, w, h puis un score par classe. NMS par classe, les 10 meilleures boîtes. */
    private fun detect(bmp: Bitmap, threshold: Float = 0.25f): Pair<List<Det>, Double> {
        val prep = letterbox(bmp)
        val input = OnnxTensor.createTensor(env, FloatBuffer.wrap(prep.data), longArrayOf(1, 3, size.toLong(), size.toLong()))
        val t0 = System.nanoTime()
        val res = currentSession().run(mapOf("images" to input))
        val ms = (System.nanoTime() - t0) / 1e6
        @Suppress("UNCHECKED_CAST")
        val p = (res[0].value as Array<Array<FloatArray>>)[0]
        res.close(); input.close()
        val nc = p.size - 4
        val cands = ArrayList<Det>()
        for (i in p[0].indices) {
            var best = 0; var s = p[4][i]
            for (c in 1 until nc) if (p[4 + c][i] > s) { s = p[4 + c][i]; best = c }
            if (s < threshold) continue
            val cx = p[0][i]; val cy = p[1][i]; val w = p[2][i]; val h = p[3][i]
            cands.add(Det(((cx - w / 2 - prep.left) / prep.ratio).coerceIn(0f, bmp.width.toFloat()),
                ((cy - h / 2 - prep.top) / prep.ratio).coerceIn(0f, bmp.height.toFloat()),
                ((cx + w / 2 - prep.left) / prep.ratio).coerceIn(0f, bmp.width.toFloat()),
                ((cy + h / 2 - prep.top) / prep.ratio).coerceIn(0f, bmp.height.toFloat()), s, best))
        }
        val keep = ArrayList<Det>()
        for (d in cands.sortedByDescending { it.score }) {
            if (keep.size >= maxBoxes) break
            if (keep.none { it.cls == d.cls && iou(it, d) > 0.5f }) keep.add(d)
        }
        return keep to ms
    }

    private fun iou(a: Det, b: Det): Float {
        val iw = (minOf(a.x2, b.x2) - maxOf(a.x1, b.x1)).coerceAtLeast(0f)
        val ih = (minOf(a.y2, b.y2) - maxOf(a.y1, b.y1)).coerceAtLeast(0f)
        val inter = iw * ih
        return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter + 1e-6f)
    }

    private fun analyze(uri: Uri) {
        lastUri = uri
        val name = models[selected].second
        val bmp = loadBitmap(uri)
        val (dets, ms) = detect(bmp)
        val shown = bmp.copy(Bitmap.Config.ARGB_8888, true)
        val c = Canvas(shown)
        val stroke = maxOf(shown.width, shown.height) / 200f
        dets.forEach { d ->
            val color = if (classHealthy[d.cls]) Color.GREEN else Color.RED
            val box = Paint().apply { this.color = color; style = Paint.Style.STROKE; strokeWidth = stroke }
            val txt = Paint().apply { this.color = color; textSize = stroke * 7; isFakeBoldText = true }
            c.drawRect(d.x1, d.y1, d.x2, d.y2, box)
            c.drawText("%s %.0f%%".format(classNames[d.cls], d.score * 100), d.x1 + stroke, d.y1 + stroke * 7, txt)
        }
        val sick = dets.filter { !classHealthy[it.cls] }
        val verdict = when {
            sick.isNotEmpty() -> "Maladie probable : ${classNames[sick[0].cls]} (%.0f%%)".format(sick[0].score * 100)
            dets.isNotEmpty() -> "Feuille saine : ${classNames[dets[0].cls]} (%.0f%%)".format(dets[0].score * 100)
            else -> "Aucune feuille détectée au-dessus de 25 %"
        }
        val weak = if (dets.isNotEmpty() && dets[0].score < 0.5f) "\nConfiance faible : vérifie que la photo montre bien une feuille de plante." else ""
        val text = "$name\n$verdict$weak\n${dets.size} boîte(s), inférence : ${"%.0f".format(ms)} ms"
        runOnUiThread { imageView.setImageBitmap(shown); info.text = text }
    }

    private fun runBench() {
        info.text = "Benchmark en cours..."
        Thread {
            val txt = runCatching {
                val bmp = BitmapFactory.decodeStream(assets.open("test.jpg"))
                repeat(3) { detect(bmp) }
                val times = (0 until 30).map { detect(bmp).second }
                "${models[selected].second}\nBenchmark (30 passes)\nLatence moyenne : %.1f ms (%.1f FPS)\nAppareil : %s %s"
                    .format(times.average(), 1000 / times.average(), android.os.Build.MANUFACTURER, android.os.Build.MODEL)
            }.getOrElse { "Erreur: $it" }
            runOnUiThread { info.text = txt }
        }.start()
    }
}
