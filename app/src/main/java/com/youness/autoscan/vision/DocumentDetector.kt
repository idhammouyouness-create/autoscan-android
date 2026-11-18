package com.youness.autoscan.vision

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** A candidate document outline in pixel coordinates of the analyzed image. */
data class Candidate(
    val quad: Quad,
    val score: Double,
    val edgeSupport: Double,
    val areaRatio: Double,
)

/**
 * Finds the document (sheet of paper) in a grayscale image.
 *
 * Several independent strategies produce candidate quadrilaterals; each candidate is then
 * scored by how well its four sides lie on real image edges and by its size. Using several
 * strategies makes detection robust to lighting, low contrast (white sheet on light table)
 * and textured backgrounds, where a single Canny pass usually fails.
 */
object DocumentDetector {

    private const val MIN_AREA_RATIO = 0.10
    private const val MAX_AREA_RATIO = 0.985
    private const val MIN_EDGE_SUPPORT = 0.55
    private const val SAMPLES_PER_SIDE = 32

    /** Best candidate, or null if no convincing document. */
    fun detect(gray: Mat): Candidate? = candidates(gray).firstOrNull()

    /** All plausible candidates sorted by score (best first). [gray] must be CV_8UC1. */
    fun candidates(gray: Mat): List<Candidate> {
        val w = gray.cols()
        val h = gray.rows()
        if (w < 32 || h < 32) return emptyList()
        val frameArea = w.toDouble() * h

        val mats = ArrayList<Mat>()
        fun m(): Mat = Mat().also { mats.add(it) }

        try {
            val blurred = m()
            Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)

            // Strategy A: Canny with thresholds derived from Otsu (adapts to scene contrast).
            val otsuBin = m()
            val otsu = Imgproc.threshold(blurred, otsuBin, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)
            val edgesA = m()
            Imgproc.Canny(blurred, edgesA, max(10.0, otsu * 0.5), max(30.0, otsu))

            // Strategy B: local contrast enhancement then low-threshold Canny (white on white).
            val eq = m()
            Imgproc.createCLAHE(3.0, Size(8.0, 8.0)).apply(blurred, eq)
            val edgesB = m()
            Imgproc.Canny(eq, edgesB, 25.0, 75.0)

            // Strategy C: morphological gradient (robust to soft shadows) + Otsu.
            val grad = m()
            Imgproc.morphologyEx(
                blurred, grad, Imgproc.MORPH_GRADIENT,
                Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0)),
            )
            val edgesC = m()
            Imgproc.threshold(grad, edgesC, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)

            val k3 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0)).also { mats.add(it) }
            val dA = m(); Imgproc.dilate(edgesA, dA, k3)
            val dB = m(); Imgproc.dilate(edgesB, dB, k3)

            // Strategy D: the sheet as a bright blob (paper is usually lighter than the table).
            val blob = m()
            Imgproc.morphologyEx(
                otsuBin, blob, Imgproc.MORPH_CLOSE,
                Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(9.0, 9.0)).also { mats.add(it) },
            )

            // Edge-support map used to score candidates: union of edge maps, slightly widened.
            val union = m()
            Core.bitwise_or(dA, dB, union)
            Core.bitwise_or(union, edgesC, union)
            val support = m()
            Imgproc.dilate(union, support, k3)
            val supportBytes = ByteArray(w * h)
            if (support.isContinuous) support.get(0, 0, supportBytes) else {
                val c = support.clone(); c.get(0, 0, supportBytes); c.release()
            }

            val raw = ArrayList<Quad>()
            val minArea = frameArea * MIN_AREA_RATIO
            quadsFrom(dA, minArea, raw)
            quadsFrom(dB, minArea, raw)
            quadsFrom(edgesC, minArea, raw)
            quadsFrom(blob, minArea, raw)

            val scored = ArrayList<Candidate>()
            for (q in raw) {
                val areaRatio = q.area() / frameArea
                if (areaRatio < MIN_AREA_RATIO || areaRatio > MAX_AREA_RATIO) continue
                if (!q.isConvex()) continue
                if (q.interiorAnglesDeg().any { it < 45.0 || it > 135.0 }) continue
                if (q.oppositeSideRatio() < 0.45) continue
                val support01 = edgeSupport(supportBytes, w, h, q)
                if (support01 < MIN_EDGE_SUPPORT) continue
                val score = support01 * 0.65 + min(areaRatio / 0.6, 1.0) * 0.35
                scored.add(Candidate(q, score, support01, areaRatio))
            }
            return dedupe(scored.sortedByDescending { it.score }, max(w, h) * 0.02)
        } finally {
            mats.forEach { it.release() }
        }
    }

    private fun quadsFrom(binary: Mat, minArea: Double, out: MutableList<Quad>) {
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        val work = binary.clone()
        try {
            Imgproc.findContours(work, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
            val big = contours.filter {
                val r = Imgproc.boundingRect(it)
                r.width.toDouble() * r.height >= minArea
            }
            val hulls = big.map { hullOf(it) }
                .map { it to Imgproc.contourArea(it) }
                .filter { it.second >= minArea }
                .sortedByDescending { it.second }
            for ((index, pair) in hulls.withIndex()) {
                val hull = pair.first
                if (index < 6) {
                    val peri = Imgproc.arcLength(hull, true)
                    for (eps in EPSILONS) {
                        val approx = MatOfPoint2f()
                        Imgproc.approxPolyDP(hull, approx, eps * peri, true)
                        val pts = approx.toArray()
                        approx.release()
                        if (pts.size == 4) {
                            out.add(Quad.fromUnordered(pts.map { Pt(it.x, it.y) }))
                            break
                        }
                        if (pts.size < 4) break
                    }
                }
                hull.release()
            }
        } finally {
            contours.forEach { it.release() }
            hierarchy.release()
            work.release()
        }
    }

    private val EPSILONS = doubleArrayOf(0.015, 0.025, 0.035, 0.05, 0.07, 0.09)

    private fun hullOf(contour: MatOfPoint): MatOfPoint2f {
        val idx = MatOfInt()
        Imgproc.convexHull(contour, idx)
        val pts = contour.toArray()
        val hullPts = idx.toArray().map { pts[it] }
        idx.release()
        return MatOfPoint2f(*hullPts.map { Point(it.x, it.y) }.toTypedArray())
    }

    /**
     * Fraction of points sampled along the 4 sides that lie on an image edge. Points hugging the
     * image border count as unsupported: a side made by the frame border is not a real sheet edge.
     */
    fun edgeSupport(support: ByteArray, w: Int, h: Int, q: Quad): Double {
        val pts = q.points
        var hit = 0
        var total = 0
        val border = 3
        var weakestSide = 1.0
        for (i in 0 until 4) {
            val a = pts[i]
            val b = pts[(i + 1) % 4]
            var sideHit = 0
            for (s in 0 until SAMPLES_PER_SIDE) {
                val t = 0.06 + 0.88 * s / (SAMPLES_PER_SIDE - 1)
                val x = (a.x + (b.x - a.x) * t).roundToInt()
                val y = (a.y + (b.y - a.y) * t).roundToInt()
                total++
                if (x < border || y < border || x >= w - border || y >= h - border) continue
                if (support[y * w + x].toInt() != 0) {
                    hit++
                    sideHit++
                }
            }
            weakestSide = min(weakestSide, sideHit.toDouble() / SAMPLES_PER_SIDE)
        }
        // A real sheet has all four sides visible; penalize a side with almost no edge.
        val overall = hit.toDouble() / total
        return if (weakestSide < 0.3) overall * 0.5 else overall
    }

    private fun dedupe(sorted: List<Candidate>, tol: Double): List<Candidate> {
        val out = ArrayList<Candidate>()
        for (c in sorted) {
            if (out.none { it.quad.maxCornerDistance(c.quad) < tol }) out.add(c)
        }
        return out
    }

    /** Variance of the Laplacian inside the given pixel quad's bounding box: higher = sharper. */
    fun sharpness(gray: Mat, quad: Quad?): Double {
        val roi = if (quad != null) {
            val xs = quad.points.map { it.x }
            val ys = quad.points.map { it.y }
            // Shrink a bit so the sheet border/background does not dominate.
            val x0 = xs.min(); val x1 = xs.max(); val y0 = ys.min(); val y1 = ys.max()
            val mx = (x1 - x0) * 0.12; val my = (y1 - y0) * 0.12
            val rx = (x0 + mx).toInt().coerceIn(0, gray.cols() - 2)
            val ry = (y0 + my).toInt().coerceIn(0, gray.rows() - 2)
            val rw = ((x1 - mx) - rx).toInt().coerceIn(2, gray.cols() - rx)
            val rh = ((y1 - my) - ry).toInt().coerceIn(2, gray.rows() - ry)
            gray.submat(ry, ry + rh, rx, rx + rw)
        } else gray
        val lap = Mat()
        val mean = org.opencv.core.MatOfDouble()
        val std = org.opencv.core.MatOfDouble()
        try {
            Imgproc.Laplacian(roi, lap, CvType.CV_64F)
            Core.meanStdDev(lap, mean, std)
            val sd = std.toArray().firstOrNull() ?: 0.0
            return sd * sd
        } finally {
            lap.release(); mean.release(); std.release()
            if (roi !== gray) roi.release()
        }
    }

    /**
     * Small normalized grayscale "fingerprint" of the sheet content, used to tell whether the
     * sheet in view is the page we just saved.
     */
    fun fingerprint(gray: Mat, quad: Quad): FloatArray {
        val fw = 48
        val fh = 64
        val src = MatOfPoint2f(*quad.points.map { Point(it.x, it.y) }.toTypedArray())
        val dst = MatOfPoint2f(
            Point(0.0, 0.0), Point(fw.toDouble(), 0.0),
            Point(fw.toDouble(), fh.toDouble()), Point(0.0, fh.toDouble()),
        )
        val m = Imgproc.getPerspectiveTransform(src, dst)
        val warped = Mat()
        val f = Mat()
        try {
            Imgproc.warpPerspective(gray, warped, m, Size(fw.toDouble(), fh.toDouble()), Imgproc.INTER_AREA)
            Imgproc.GaussianBlur(warped, warped, Size(3.0, 3.0), 0.0)
            warped.convertTo(f, CvType.CV_32F)
            val arr = FloatArray(fw * fh)
            f.get(0, 0, arr)
            return normalize(arr)
        } finally {
            src.release(); dst.release(); m.release(); warped.release(); f.release()
        }
    }

    fun normalize(a: FloatArray): FloatArray {
        val mean = a.average().toFloat()
        var v = 0.0
        for (x in a) v += ((x - mean) * (x - mean)).toDouble()
        val sd = kotlin.math.sqrt(v / a.size).toFloat().coerceAtLeast(1e-3f)
        return FloatArray(a.size) { (a[it] - mean) / sd }
    }

    /** Normalized cross-correlation of two fingerprints, in -1..1. */
    fun similarity(a: FloatArray, b: FloatArray): Double {
        if (a.size != b.size || a.isEmpty()) return 0.0
        var s = 0.0
        for (i in a.indices) s += a[i] * b[i]
        return s / a.size
    }
}
