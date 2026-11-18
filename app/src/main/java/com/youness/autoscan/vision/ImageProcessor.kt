package com.youness.autoscan.vision

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

enum class EnhanceMode { DOCUMENT, GRAYSCALE, BLACK_WHITE, ORIGINAL }

class ProcessedPage(val bitmap: Bitmap, val sharpness: Double, val usedDetectedOutline: Boolean)

/**
 * Turns a full-resolution camera photo into a clean, flat page:
 * 1. re-detects the sheet on the photo itself (more precise than the preview outline),
 * 2. refines each side to sub-pixel accuracy by fitting lines to the strongest edge,
 * 3. corrects perspective,
 * 4. removes shadows / uneven lighting and improves text contrast.
 */
object ImageProcessor {

    private const val MAX_OUTPUT_SIDE = 3000.0
    private const val DETECT_SIDE = 1000.0
    private const val REFINE_SIDE = 2000.0
    private const val A4 = 1.41421356

    fun decodeJpeg(bytes: ByteArray, rotationDegrees: Int): Mat {
        // Bound decoded size so very high-resolution sensors cannot exhaust memory.
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / sample > 4800) sample *= 2
        val opts = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts)
            ?: throw IllegalStateException("cannot decode camera image")
        val rgba = Mat()
        Utils.bitmapToMat(bmp, rgba)
        bmp.recycle()
        if (rotationDegrees == 0) return rgba
        val out = Mat()
        val code = when (rotationDegrees) {
            90 -> Core.ROTATE_90_CLOCKWISE
            180 -> Core.ROTATE_180
            else -> Core.ROTATE_90_COUNTERCLOCKWISE
        }
        Core.rotate(rgba, out, code)
        rgba.release()
        return out
    }

    /**
     * @param rgba upright photo (CV_8UC4)
     * @param hint normalized outline seen in the preview at capture time (may be null)
     */
    fun process(rgba: Mat, hint: Quad?, mode: EnhanceMode): ProcessedPage {
        val w = rgba.cols().toDouble()
        val h = rgba.rows().toDouble()
        val gray = Mat()
        Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
        try {
            val (outline, detected) = locate(gray, hint)
            val warped = if (outline != null) {
                warp(rgba, outline.scale(w, h))
            } else {
                rgba.clone() // no sheet found (manual capture): keep the whole photo
            }
            val sized = limitSize(warped)
            // Measure sharpness before enhancement: sharpening would hide real blur.
            val sharp = pageSharpness(sized)
            val enhanced = enhance(sized, mode)
            sized.release()
            val rgbaOut = Mat()
            when (enhanced.channels()) {
                1 -> Imgproc.cvtColor(enhanced, rgbaOut, Imgproc.COLOR_GRAY2RGBA)
                3 -> Imgproc.cvtColor(enhanced, rgbaOut, Imgproc.COLOR_RGB2RGBA)
                else -> enhanced.copyTo(rgbaOut)
            }
            enhanced.release()
            val bmp = Bitmap.createBitmap(rgbaOut.cols(), rgbaOut.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(rgbaOut, bmp)
            rgbaOut.release()
            return ProcessedPage(bmp, sharp, detected)
        } finally {
            gray.release()
        }
    }

    /** Returns a normalized outline for the photo and whether it was detected on the photo. */
    fun locate(gray: Mat, hint: Quad?): Pair<Quad?, Boolean> {
        val w = gray.cols().toDouble()
        val h = gray.rows().toDouble()

        // 1) detect on a reduced copy of the photo
        val s1 = min(1.0, DETECT_SIDE / max(w, h))
        val small = Mat()
        Imgproc.resize(gray, small, Size(w * s1, h * s1), 0.0, 0.0, Imgproc.INTER_AREA)
        val cands = DocumentDetector.candidates(small)
        val sw = small.cols().toDouble()
        val sh = small.rows().toDouble()
        small.release()
        val normCands = cands.map { it.copy(quad = it.quad.scale(1.0 / sw, 1.0 / sh)) }
        val chosen: Quad? = when {
            hint != null -> normCands.filter { it.quad.maxCornerDistance(hint) < 0.07 }
                .maxByOrNull { it.score }?.quad
            else -> normCands.firstOrNull()?.quad
        }
        val coarse = chosen ?: hint ?: return null to false

        // 2) refine sides on a higher-resolution copy
        val s2 = min(1.0, REFINE_SIDE / max(w, h))
        val mid = Mat()
        Imgproc.resize(gray, mid, Size(w * s2, h * s2), 0.0, 0.0, Imgproc.INTER_AREA)
        Imgproc.GaussianBlur(mid, mid, Size(5.0, 5.0), 0.0)
        val mw = mid.cols()
        val mh = mid.rows()
        val bytes = ByteArray(mw * mh)
        mid.get(0, 0, bytes)
        mid.release()
        val refined = EdgeRefiner.refine(bytes, mw, mh, coarse.scale(mw.toDouble(), mh.toDouble()))
        return refined.scale(1.0 / mw, 1.0 / mh) to (chosen != null)
    }

    fun warp(rgba: Mat, q: Quad): Mat {
        val wTop = q.tl.dist(q.tr)
        val wBottom = q.bl.dist(q.br)
        val hLeft = q.tl.dist(q.bl)
        val hRight = q.tr.dist(q.br)
        var outW = max(wTop, wBottom)
        var outH = max(hLeft, hRight)
        // Snap to A4 proportions when close: most school/office sheets are A4.
        val ratio = outH / outW
        if (abs(ratio - A4) / A4 < 0.045) outH = outW * A4
        else if (abs(ratio - 1 / A4) / (1 / A4) < 0.045) outW = outH * A4
        val scale = min(1.0, MAX_OUTPUT_SIDE / max(outW, outH))
        outW *= scale
        outH *= scale
        val ow = outW.roundToInt().coerceAtLeast(16)
        val oh = outH.roundToInt().coerceAtLeast(16)
        val src = MatOfPoint2f(*q.points.map { Point(it.x, it.y) }.toTypedArray())
        val dst = MatOfPoint2f(
            Point(0.0, 0.0), Point(ow.toDouble(), 0.0),
            Point(ow.toDouble(), oh.toDouble()), Point(0.0, oh.toDouble()),
        )
        val m = Imgproc.getPerspectiveTransform(src, dst)
        val out = Mat()
        Imgproc.warpPerspective(rgba, out, m, Size(ow.toDouble(), oh.toDouble()), Imgproc.INTER_CUBIC, Core.BORDER_REPLICATE)
        src.release(); dst.release(); m.release()
        return out
    }

    private fun limitSize(m: Mat): Mat {
        val side = max(m.cols(), m.rows()).toDouble()
        if (side <= MAX_OUTPUT_SIDE) return m
        val s = MAX_OUTPUT_SIDE / side
        val out = Mat()
        Imgproc.resize(m, out, Size(m.cols() * s, m.rows() * s), 0.0, 0.0, Imgproc.INTER_AREA)
        m.release()
        return out
    }

    /** Input RGBA; output RGB (3 channels) or GRAY (1 channel). Does not release input. */
    fun enhance(rgba: Mat, mode: EnhanceMode): Mat {
        val rgb = Mat()
        Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
        return when (mode) {
            EnhanceMode.ORIGINAL -> rgb
            EnhanceMode.DOCUMENT -> {
                val flat = flattenLighting(rgb)
                rgb.release()
                val out = finish(flat)
                flat.release()
                out
            }
            EnhanceMode.GRAYSCALE, EnhanceMode.BLACK_WHITE -> {
                val g = Mat()
                Imgproc.cvtColor(rgb, g, Imgproc.COLOR_RGB2GRAY)
                rgb.release()
                val flat = flattenLighting(g)
                g.release()
                if (mode == EnhanceMode.GRAYSCALE) {
                    val out = finish(flat)
                    flat.release()
                    out
                } else {
                    val bw = Mat()
                    Imgproc.adaptiveThreshold(
                        flat, bw, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                        Imgproc.THRESH_BINARY, 31, 12.0,
                    )
                    flat.release()
                    bw
                }
            }
        }
    }

    /**
     * Shadow / uneven light removal: estimate the paper brightness everywhere (the "background")
     * by removing text with a max filter + median blur at low resolution, then divide by it.
     * The paper becomes evenly white while ink keeps its color.
     */
    private fun flattenLighting(src: Mat): Mat {
        val channels = ArrayList<Mat>()
        Core.split(src, channels)
        val outCh = ArrayList<Mat>()
        val factor = 0.25
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(7.0, 7.0))
        for (ch in channels) {
            val small = Mat()
            Imgproc.resize(ch, small, Size(ch.cols() * factor, ch.rows() * factor), 0.0, 0.0, Imgproc.INTER_AREA)
            Imgproc.dilate(small, small, kernel)
            Imgproc.medianBlur(small, small, 21)
            val bg = Mat()
            Imgproc.resize(small, bg, ch.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
            small.release()
            val out = Mat()
            Core.divide(ch, bg, out, 255.0)
            bg.release()
            ch.release()
            outCh.add(out)
        }
        kernel.release()
        val merged = Mat()
        Core.merge(outCh, merged)
        outCh.forEach { it.release() }
        return merged
    }

    /** Tone curve (white paper, darker ink) + mild unsharp mask for crisp text. */
    private fun finish(src: Mat): Mat {
        val lut = Mat(1, 256, CvType.CV_8UC1)
        val table = ByteArray(256)
        val white = 238.0
        for (i in 0 until 256) {
            val x = (i / white).coerceAtMost(1.0)
            table[i] = (255.0 * x.pow(1.35)).roundToInt().coerceIn(0, 255).toByte()
        }
        lut.put(0, 0, table)
        val toned = Mat()
        Core.LUT(src, lut, toned)
        lut.release()
        val blur = Mat()
        Imgproc.GaussianBlur(toned, blur, Size(0.0, 0.0), 1.2)
        val out = Mat()
        Core.addWeighted(toned, 1.6, blur, -0.6, 0.0, out)
        toned.release()
        blur.release()
        return out
    }

    /** Laplacian variance of the page at a fixed width so values are comparable between pages. */
    fun pageSharpness(page: Mat): Double {
        val g = Mat()
        when (page.channels()) {
            1 -> page.copyTo(g)
            4 -> Imgproc.cvtColor(page, g, Imgproc.COLOR_RGBA2GRAY)
            else -> Imgproc.cvtColor(page, g, Imgproc.COLOR_RGB2GRAY)
        }
        val s = 1000.0 / g.cols()
        val r = Mat()
        Imgproc.resize(g, r, Size(1000.0, g.rows() * s), 0.0, 0.0, Imgproc.INTER_AREA)
        g.release()
        val v = DocumentDetector.sharpness(r, null)
        r.release()
        return v
    }
}

/**
 * Sub-pixel refinement of a coarse outline. For each side, it searches along the normal for the
 * strongest brightness step at many points, fits a straight line through those points (with
 * outlier rejection), and intersects neighbouring lines to get precise corners.
 */
object EdgeRefiner {

    private data class Line(val px: Double, val py: Double, val dx: Double, val dy: Double)

    fun refine(img: ByteArray, w: Int, h: Int, q: Quad): Quad {
        val diag = hypot(w.toDouble(), h.toDouble())
        val radius = max(6.0, diag * 0.018)
        val pts = q.points
        val lines = ArrayList<Line?>()
        for (i in 0 until 4) {
            lines.add(fitSide(img, w, h, pts[i], pts[(i + 1) % 4], radius))
        }
        if (lines.any { it == null }) return q
        val l = lines.map { it!! }
        // side i goes from corner i to corner i+1; corner i = side(i-1) ∩ side(i)
        val corners = (0 until 4).map { i -> intersect(l[(i + 3) % 4], l[i]) }
        if (corners.any { it == null }) return q
        val refined = Quad(corners[0]!!, corners[1]!!, corners[2]!!, corners[3]!!)
        // Safety: refinement must stay close to the coarse outline and stay a sane shape.
        if (refined.maxCornerDistance(q) > radius * 2.5) return q
        if (!refined.isConvex()) return q
        return refined
    }

    private fun fitSide(img: ByteArray, w: Int, h: Int, a: Pt, b: Pt, radius: Double): Line? {
        val len = a.dist(b)
        if (len < 10) return null
        val dx = (b.x - a.x) / len
        val dy = (b.y - a.y) / len
        val nx = -dy
        val ny = dx
        val samples = 48
        val xs = ArrayList<Double>()
        val ys = ArrayList<Double>()
        val steps = radius.toInt()
        for (s in 0 until samples) {
            val t = 0.08 + 0.84 * s / (samples - 1)
            val bx = a.x + (b.x - a.x) * t
            val by = a.y + (b.y - a.y) * t
            var bestG = 0.0
            var bestOff = Double.NaN
            for (o in -steps..steps) {
                val g = abs(sample(img, w, h, bx + nx * (o + 1), by + ny * (o + 1)) -
                    sample(img, w, h, bx + nx * (o - 1), by + ny * (o - 1)))
                if (g > bestG) {
                    bestG = g
                    bestOff = o.toDouble()
                }
            }
            if (bestG >= 12.0 && !bestOff.isNaN()) {
                xs.add(bx + nx * bestOff)
                ys.add(by + ny * bestOff)
            }
        }
        if (xs.size < samples / 3) return null
        var line = fitLine(xs, ys) ?: return null
        // One round of outlier rejection.
        val res = xs.indices.map { distToLine(line, xs[it], ys[it]) }
        val med = res.sorted()[res.size / 2]
        val keepX = ArrayList<Double>()
        val keepY = ArrayList<Double>()
        for (i in xs.indices) if (res[i] <= max(1.5, med * 3)) {
            keepX.add(xs[i]); keepY.add(ys[i])
        }
        if (keepX.size >= samples / 3) line = fitLine(keepX, keepY) ?: line
        return line
    }

    private fun sample(img: ByteArray, w: Int, h: Int, x: Double, y: Double): Double {
        // bilinear interpolation
        val xi = x.coerceIn(0.0, w - 1.001)
        val yi = y.coerceIn(0.0, h - 1.001)
        val x0 = xi.toInt(); val y0 = yi.toInt()
        val fx = xi - x0; val fy = yi - y0
        fun p(cx: Int, cy: Int) = (img[cy * w + cx].toInt() and 0xFF).toDouble()
        val top = p(x0, y0) * (1 - fx) + p(x0 + 1, y0) * fx
        val bot = p(x0, y0 + 1) * (1 - fx) + p(x0 + 1, y0 + 1) * fx
        return top * (1 - fy) + bot * fy
    }

    /** Total least squares line through points. */
    private fun fitLine(xs: List<Double>, ys: List<Double>): Line? {
        val n = xs.size
        if (n < 2) return null
        val mx = xs.average()
        val my = ys.average()
        var sxx = 0.0; var syy = 0.0; var sxy = 0.0
        for (i in 0 until n) {
            val ex = xs[i] - mx; val ey = ys[i] - my
            sxx += ex * ex; syy += ey * ey; sxy += ex * ey
        }
        val theta = 0.5 * kotlin.math.atan2(2 * sxy, sxx - syy)
        val dx = kotlin.math.cos(theta)
        val dy = kotlin.math.sin(theta)
        if (sqrt(dx * dx + dy * dy) < 1e-9) return null
        return Line(mx, my, dx, dy)
    }

    private fun distToLine(l: Line, x: Double, y: Double): Double =
        abs((x - l.px) * l.dy - (y - l.py) * l.dx)

    private fun intersect(a: Line, b: Line): Pt? {
        val det = a.dx * b.dy - a.dy * b.dx
        if (abs(det) < 1e-6) return null
        val t = ((b.px - a.px) * b.dy - (b.py - a.py) * b.dx) / det
        return Pt(a.px + a.dx * t, a.py + a.dy * t)
    }
}
