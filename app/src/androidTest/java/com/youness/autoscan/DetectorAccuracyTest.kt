package com.youness.autoscan

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.youness.autoscan.vision.DocumentDetector
import com.youness.autoscan.vision.ImageProcessor
import com.youness.autoscan.vision.Pt
import com.youness.autoscan.vision.Quad
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.util.Random
import kotlin.math.hypot

/**
 * Renders synthetic "photos" of a sheet of paper (known corner positions) in difficult conditions
 * and checks the detector finds the true corners. Results are printed to logcat (tag AccuracyTest).
 */
@RunWith(AndroidJUnit4::class)
class DetectorAccuracyTest {

    companion object {
        @BeforeClass
        @JvmStatic
        fun init() {
            check(OpenCVLoader.initLocal()) { "OpenCV not loaded" }
        }
    }

    enum class Bg { DARK_WOOD, LIGHT_TABLE, BUSY_PATTERN, SAME_AS_PAPER_ISH }

    data class Scene(
        val name: String,
        val bg: Bg,
        val corners: Quad,
        val shadow: Boolean = false,
        val blur: Double = 0.0,
        val noise: Double = 4.0,
        val hand: Boolean = false,
    )

    private fun render(s: Scene, w: Int, h: Int, seed: Long = 7): Mat {
        val rnd = Random(seed)
        val img = Mat(h, w, CvType.CV_8UC1)
        when (s.bg) {
            Bg.DARK_WOOD -> {
                img.setTo(Scalar(70.0))
                for (y in 0 until h step 3) {
                    val v = 55.0 + 25.0 * kotlin.math.sin(y / 17.0) + rnd.nextInt(10)
                    Imgproc.line(img, Point(0.0, y.toDouble()), Point(w.toDouble(), y + rnd.nextInt(6).toDouble()), Scalar(v), 2)
                }
            }
            Bg.LIGHT_TABLE -> img.setTo(Scalar(178.0))
            Bg.BUSY_PATTERN -> {
                img.setTo(Scalar(110.0))
                repeat(140) {
                    val c = Point(rnd.nextInt(w).toDouble(), rnd.nextInt(h).toDouble())
                    Imgproc.circle(img, c, 5 + rnd.nextInt(30), Scalar(40.0 + rnd.nextInt(150)), -1)
                }
                repeat(60) {
                    Imgproc.line(
                        img, Point(rnd.nextInt(w).toDouble(), rnd.nextInt(h).toDouble()),
                        Point(rnd.nextInt(w).toDouble(), rnd.nextInt(h).toDouble()), Scalar(rnd.nextInt(255).toDouble()), 2,
                    )
                }
            }
            Bg.SAME_AS_PAPER_ISH -> img.setTo(Scalar(200.0))
        }

        // Paper: render a flat page with text lines, then warp it into the scene.
        val pw = 600
        val ph = 848
        val page = Mat(ph, pw, CvType.CV_8UC1, Scalar(236.0))
        var y = 70
        while (y < ph - 60) {
            var x = 50
            while (x < pw - 60) {
                val len = 20 + rnd.nextInt(70)
                if (x + len < pw - 50) Imgproc.rectangle(page, Point(x.toDouble(), y.toDouble()), Point((x + len).toDouble(), (y + 9).toDouble()), Scalar(30.0), -1)
                x += len + 12
            }
            y += 26
        }
        val src = MatOfPoint2f(Point(0.0, 0.0), Point(pw.toDouble(), 0.0), Point(pw.toDouble(), ph.toDouble()), Point(0.0, ph.toDouble()))
        val dst = MatOfPoint2f(*s.corners.points.map { Point(it.x * w, it.y * h) }.toTypedArray())
        val m = Imgproc.getPerspectiveTransform(src, dst)
        val warped = Mat()
        Imgproc.warpPerspective(page, warped, m, Size(w.toDouble(), h.toDouble()), Imgproc.INTER_LINEAR)
        val mask = Mat(h, w, CvType.CV_8UC1, Scalar(0.0))
        Imgproc.fillConvexPoly(mask, MatOfPoint(*s.corners.points.map { Point(it.x * w, it.y * h) }.toTypedArray()), Scalar(255.0))
        warped.copyTo(img, mask)

        if (s.hand) {
            // a "thumb" covering part of the bottom edge
            val c = s.corners.bl.lerp(s.corners.br, 0.5)
            Imgproc.ellipse(img, Point(c.x * w, c.y * h + 10), Size(w * 0.06, h * 0.09), 0.0, 0.0, 360.0, Scalar(150.0), -1)
        }
        if (s.shadow) {
            // soft diagonal shadow across the scene
            val grad = Mat(h, w, CvType.CV_32F)
            val row = FloatArray(w)
            for (yy in 0 until h) {
                for (xx in 0 until w) row[xx] = (0.55 + 0.45 * ((xx + yy).toDouble() / (w + h))).toFloat()
                grad.put(yy, 0, row)
            }
            val f = Mat()
            img.convertTo(f, CvType.CV_32F)
            Core.multiply(f, grad, f)
            f.convertTo(img, CvType.CV_8U)
            f.release(); grad.release()
        }
        if (s.noise > 0) {
            val n = Mat(h, w, CvType.CV_16S)
            Core.randn(n, 0.0, s.noise)
            val i16 = Mat()
            img.convertTo(i16, CvType.CV_16S)
            Core.add(i16, n, i16)
            i16.convertTo(img, CvType.CV_8U)
            n.release(); i16.release()
        }
        if (s.blur > 0) Imgproc.GaussianBlur(img, img, Size(0.0, 0.0), s.blur)
        page.release(); warped.release(); mask.release(); m.release(); src.release(); dst.release()
        return img
    }

    private fun Pt.lerp(o: Pt, t: Double) = Pt(x + (o.x - x) * t, y + (o.y - y) * t)

    private val scenes = listOf(
        Scene("straight_dark_wood", Bg.DARK_WOOD, Quad(Pt(0.18, 0.14), Pt(0.82, 0.15), Pt(0.83, 0.86), Pt(0.17, 0.85))),
        Scene("perspective_dark", Bg.DARK_WOOD, Quad(Pt(0.25, 0.12), Pt(0.78, 0.18), Pt(0.88, 0.88), Pt(0.12, 0.83))),
        Scene("rotated_light_table", Bg.LIGHT_TABLE, Quad(Pt(0.30, 0.08), Pt(0.88, 0.22), Pt(0.72, 0.92), Pt(0.12, 0.78))),
        Scene("busy_background", Bg.BUSY_PATTERN, Quad(Pt(0.15, 0.12), Pt(0.85, 0.10), Pt(0.87, 0.88), Pt(0.14, 0.90))),
        Scene("shadow_light_table", Bg.LIGHT_TABLE, Quad(Pt(0.16, 0.13), Pt(0.84, 0.12), Pt(0.85, 0.87), Pt(0.15, 0.88)), shadow = true),
        Scene("low_contrast", Bg.SAME_AS_PAPER_ISH, Quad(Pt(0.17, 0.12), Pt(0.83, 0.13), Pt(0.84, 0.87), Pt(0.16, 0.86))),
        Scene("thumb_on_edge", Bg.DARK_WOOD, Quad(Pt(0.18, 0.12), Pt(0.82, 0.12), Pt(0.84, 0.86), Pt(0.16, 0.86)), hand = true),
        Scene("noisy_dim", Bg.DARK_WOOD, Quad(Pt(0.20, 0.15), Pt(0.80, 0.16), Pt(0.80, 0.85), Pt(0.20, 0.84)), noise = 12.0),
    )

    private fun cornerError(found: Quad, truth: Quad): Double = found.maxCornerDistance(truth)

    @Test
    fun previewDetection_findsSheetInAllScenes() {
        val w = 480
        val h = 640
        val failures = ArrayList<String>()
        for (s in scenes) {
            val img = render(s, w, h)
            val c = DocumentDetector.detect(img)
            img.release()
            val err = c?.let { cornerError(it.quad.scale(1.0 / w, 1.0 / h), s.corners) }
            Log.i("AccuracyTest", "preview ${s.name}: err=${err?.let { "%.4f".format(it) } ?: "NOT FOUND"} support=${c?.edgeSupport} got=${c?.quad} truth=${s.corners.scale(w.toDouble(), h.toDouble())}")
            if (err == null || err > 0.03) failures += "${s.name}: ${err ?: "not found"}"
        }
        assertTrue("Detection failures: $failures", failures.isEmpty())
    }

    @Test
    fun noSheet_noDetection() {
        val w = 480
        val h = 640
        for (bg in listOf(Bg.DARK_WOOD, Bg.LIGHT_TABLE)) {
            val img = Mat(h, w, CvType.CV_8UC1)
            val tmp = render(Scene("x", bg, Quad(Pt(-2.0, -2.0), Pt(-1.9, -2.0), Pt(-1.9, -1.9), Pt(-2.0, -1.9))), w, h)
            tmp.copyTo(img)
            tmp.release()
            val c = DocumentDetector.detect(img)
            Log.i("AccuracyTest", "empty $bg -> ${c?.score}")
            img.release()
            assertTrue("false detection on empty $bg", c == null)
        }
    }

    @Test
    fun fullResolution_refinedCornersAreSubPercentAccurate() {
        val w = 3024
        val h = 4032
        val diag = hypot(w.toDouble(), h.toDouble())
        val worst = ArrayList<String>()
        for (s in scenes.filter { !it.hand }) {
            val img = render(s, w, h)
            // Simulate an imprecise preview hint (1.5% error on each corner).
            val hint = Quad(
                s.corners.tl + Pt(0.012, -0.01), s.corners.tr + Pt(-0.01, 0.012),
                s.corners.br + Pt(0.01, 0.01), s.corners.bl + Pt(-0.012, -0.008),
            )
            val (q, detected) = ImageProcessor.locate(img, hint)
            img.release()
            assertNotNull(q)
            val errPx = q!!.scale(w.toDouble(), h.toDouble()).maxCornerDistance(s.corners.scale(w.toDouble(), h.toDouble()))
            Log.i("AccuracyTest", "fullres ${s.name}: detected=$detected err=${"%.1f".format(errPx)}px (${"%.3f".format(100 * errPx / diag)}% of diagonal)")
            if (errPx / diag > 0.004) worst += "${s.name}: ${"%.1f".format(errPx)}px"
        }
        assertTrue("Imprecise corners: $worst", worst.isEmpty())
    }

    @Test
    fun sharpness_separatesSharpFromBlurred() {
        val s = scenes[0]
        val sharp = render(s, 480, 640)
        val blurred = render(s.copy(blur = 2.5), 480, 640)
        val q = s.corners.scale(480.0, 640.0)
        val a = DocumentDetector.sharpness(sharp, q)
        val b = DocumentDetector.sharpness(blurred, q)
        Log.i("AccuracyTest", "sharpness sharp=$a blurred=$b")
        sharp.release(); blurred.release()
        assertTrue("sharp=$a blurred=$b", a > b * 3)
    }

    @Test
    fun fingerprint_samePageSimilar_differentPageNot() {
        val s = scenes[0]
        val a = render(s, 480, 640, seed = 1)
        val aShifted = render(s.copy(corners = Quad(s.corners.tl + Pt(0.01, 0.0), s.corners.tr + Pt(0.01, 0.0), s.corners.br + Pt(0.01, 0.0), s.corners.bl + Pt(0.01, 0.0))), 480, 640, seed = 1)
        val other = render(s, 480, 640, seed = 99)
        val fa = DocumentDetector.fingerprint(a, DocumentDetector.detect(a)!!.quad)
        val fb = DocumentDetector.fingerprint(aShifted, DocumentDetector.detect(aShifted)!!.quad)
        val fc = DocumentDetector.fingerprint(other, DocumentDetector.detect(other)!!.quad)
        val same = DocumentDetector.similarity(fa, fb)
        val diff = DocumentDetector.similarity(fa, fc)
        Log.i("AccuracyTest", "similarity same=$same different=$diff quads a=${DocumentDetector.detect(a)?.quad} shifted=${DocumentDetector.detect(aShifted)?.quad}")
        a.release(); aShifted.release(); other.release()
        assertTrue("same page similarity too low: $same", same > 0.9)
        assertTrue("different page similarity too high: $diff", diff < same)
    }
}
