package com.youness.autoscan

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.youness.autoscan.vision.DocumentDetector
import com.youness.autoscan.vision.Pt
import com.youness.autoscan.vision.Quad
import org.json.JSONArray
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.imgcodecs.Imgcodecs

/**
 * Real photos of a spiral-bound A4 book (from the user's phone) and complete pages cut from them.
 * Expected outlines come from the Python prototype the algorithm was tuned with.
 */
@RunWith(AndroidJUnit4::class)
class RealPhotoTest {

    companion object {
        @BeforeClass
        @JvmStatic
        fun init() {
            check(OpenCVLoader.initLocal()) { "OpenCV not loaded" }
        }
    }

    private val ctx get() = InstrumentationRegistry.getInstrumentation().context

    private fun loadGray(name: String): Mat {
        val bytes = ctx.assets.open("real/$name").use { it.readBytes() }
        val buf = MatOfByte(*bytes)
        val m = Imgcodecs.imdecode(buf, Imgcodecs.IMREAD_GRAYSCALE)
        buf.release()
        check(!m.empty()) { "cannot decode $name" }
        return m
    }

    @Test
    fun realPhotos_andCompletePages() {
        val cases = JSONArray(ctx.assets.open("real/cases.json").bufferedReader().readText())
        val failures = ArrayList<String>()
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val file = c.getString("file")
            val img = loadGray(file)
            val w = img.cols().toDouble()
            val h = img.rows().toDouble()
            val t0 = System.nanoTime()
            val det = DocumentDetector.analyze(img, null)
            val ms = (System.nanoTime() - t0) / 1e6
            img.release()
            when (c.getString("expect")) {
                "too_close" -> {
                    val ok = det.best == null && det.tooClose
                    Log.i("AccuracyTest", "real $file: best=${det.best?.quad} tooClose=${det.tooClose} ${"%.0f".format(ms)}ms ${if (ok) "OK" else "FAIL"}")
                    if (!ok) failures += "$file: expected 'move back', got best=${det.best?.quad} tooClose=${det.tooClose}"
                }
                "page" -> {
                    val arr = c.getJSONArray("quad")
                    val pts = (0 until 4).map { Pt(arr.getJSONArray(it).getDouble(0), arr.getJSONArray(it).getDouble(1)) }
                    val expected = Quad(pts[0], pts[1], pts[2], pts[3])
                    val got = det.best?.quad?.scale(1.0 / w, 1.0 / h)
                    val err = got?.maxCornerDistance(expected)
                    Log.i("AccuracyTest", "page $file: err=${err?.let { "%.4f".format(it) } ?: "NOT FOUND"} ${"%.0f".format(ms)}ms got=$got expected=$expected n=${det.candidates.size}")
                    if (err == null || err > 0.025) failures += "$file: ${err ?: "not found"}"
                }
            }
        }
        assertTrue("Real photo failures: $failures", failures.isEmpty())
    }
}
