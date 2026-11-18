package com.youness.autoscan.vision

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin

/** A candidate page outline in pixel coordinates of the analyzed image. */
data class Candidate(
    val quad: Quad,
    val score: Double,
    /** Mean fraction (0..1) of each side showing a real paper edge. */
    val edgeSupport: Double,
    val areaRatio: Double,
)

/** Result for one image: best complete page (if any) and whether the page runs off the frame. */
data class Detection(val candidates: List<Candidate>, val tooClose: Boolean) {
    val best: Candidate? get() = candidates.firstOrNull()
}

/**
 * Finds a complete page in a grayscale image.
 *
 * 1. "Paper map": a morphological closing erases text, table lines and stamps, leaving only
 *    paper vs. background. Without it, a printed table inside the page is the most obvious
 *    rectangle and gets picked instead of the page.
 * 2. Candidate outlines come from the paper map's edges, the bright paper region, and straight
 *    lines (Hough) — lines survive the irregular edge of a spiral binding.
 * 3. Every side must show a real STEP between paper and something else (tolerant search near the
 *    expected position, for bindings and slightly curved pages). Printed frames have paper on
 *    both sides and are rejected.
 * 4. Only complete pages qualify. If paper runs off the frame, [Detection.tooClose] tells the UI to
 *    ask the user to move back instead of capturing a cut page.
 *
 * A Python mirror of this algorithm was tuned on real photos of a spiral-bound A4 book.
 */
object DocumentDetector {

    private const val MIN_AREA_RATIO = 0.15
    private const val STEP_MIN = 14.0
    private const val SIDE_GOOD_FRAC = 0.45
    private const val MEAN_GOOD = 0.72
    private const val SAMPLES = 48
    private const val BORDER_PX = 4.0
    private const val LINES_PER_FAMILY = 7

    fun detect(gray: Mat, previous: Quad? = null): Candidate? = analyze(gray, previous).best

    fun candidates(gray: Mat): List<Candidate> = analyze(gray, null).candidates

    /** [gray] must be CV_8UC1. [previous] (pixel coords) slightly favours a stable outline. */
    fun analyze(gray: Mat, previous: Quad?): Detection {
        val w = gray.cols()
        val h = gray.rows()
        if (w < 32 || h < 32) return Detection(emptyList(), false)
        val frameArea = w.toDouble() * h
        val mats = ArrayList<Mat>()
        fun m(): Mat = Mat().also { mats.add(it) }
        try {
            // 1. paper map
            var k = max(5, (min(w, h) / 50.0).roundToInt())
            if (k % 2 == 0) k++
            val closed = m()
            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(k.toDouble(), k.toDouble())).also { mats.add(it) }
            Imgproc.morphologyEx(gray, closed, Imgproc.MORPH_CLOSE, kernel)
            val paper = m()
            Imgproc.GaussianBlur(closed, paper, Size(5.0, 5.0), 0.0)

            val blob = m()
            val otsu = Imgproc.threshold(paper, blob, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)
            val edges = m()
            Imgproc.Canny(paper, edges, max(8.0, otsu * 0.25), max(20.0, otsu * 0.6))
            val edgesD = m()
            Imgproc.dilate(edges, edgesD, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0)).also { mats.add(it) })

            val paperBytes = toBytes(paper)
            val blobBytes = toBytes(blob)
            val tooClose = tooClose(blobBytes, w, h)

            // 2. candidate outlines
            val raw = ArrayList<Quad>()
            val minArea = frameArea * MIN_AREA_RATIO
            contourQuads(edgesD, minArea, raw)
            contourQuads(blob, minArea, raw)
            lineQuads(edges, w, h, raw)

            // 3. evaluate
            val seen = ArrayList<Quad>()
            val out = ArrayList<Candidate>()
            val dupTol = 4.0
            for (q in raw) {
                if (seen.any { it.maxCornerDistance(q) < dupTol }) continue
                seen.add(q)
                val areaRatio = q.area() / frameArea
                if (areaRatio < MIN_AREA_RATIO || !q.isConvex()) continue
                if (q.interiorAnglesDeg().any { it < 55.0 || it > 125.0 }) continue
                if (q.oppositeSideRatio() < 0.55) continue
                val quality = sideQuality(paperBytes, w, h, q) ?: continue
                val mean = quality.average()
                if (quality.any { it < SIDE_GOOD_FRAC } || mean < MEAN_GOOD) continue
                val inside = meanInside(paper, q)
                var score = 0.50 * min(areaRatio / 0.7, 1.0) + 0.30 * mean + 0.20 + 0.002 * (inside - 128) / 1.27
                if (previous != null && previous.maxCornerDistance(q) < min(w, h) * 0.03) score += 0.04
                out.add(Candidate(q, score, mean, areaRatio))
            }
            out.sortByDescending { it.score }
            return Detection(out, tooClose && out.isEmpty())
        } finally {
            mats.forEach { it.release() }
        }
    }

    /** Mean paper-map brightness inside the outline (a real page is bright inside). */
    private fun meanInside(paper: Mat, q: Quad): Double {
        val mask = Mat.zeros(paper.rows(), paper.cols(), CvType.CV_8UC1)
        val poly = MatOfPoint(*q.points.map { Point(Math.round(it.x).toDouble(), Math.round(it.y).toDouble()) }.toTypedArray())
        Imgproc.fillConvexPoly(mask, poly, org.opencv.core.Scalar(255.0))
        val v = Core.mean(paper, mask).`val`[0]
        poly.release()
        mask.release()
        return v
    }

    private fun toBytes(m: Mat): ByteArray {
        val b = ByteArray(m.cols() * m.rows())
        if (m.isContinuous) m.get(0, 0, b) else {
            val c = m.clone(); c.get(0, 0, b); c.release()
        }
        return b
    }

    /** Paper runs off the frame: page is cut, the user should move back. */
    private fun tooClose(blob: ByteArray, w: Int, h: Int): Boolean {
        var paper = 0
        for (v in blob) if (v.toInt() != 0) paper++
        if (paper < 0.45 * w * h) return false
        val m = 2
        fun rowFrac(y: Int): Double {
            var c = 0
            for (x in 0 until w) if (blob[y * w + x].toInt() != 0) c++
            return c.toDouble() / w
        }
        fun colFrac(x: Int): Double {
            var c = 0
            for (y in 0 until h) if (blob[y * w + x].toInt() != 0) c++
            return c.toDouble() / h
        }
        return maxOf(rowFrac(m), rowFrac(h - 1 - m), colFrac(m), colFrac(w - 1 - m)) > 0.40
    }

    private fun contourQuads(binary: Mat, minArea: Double, out: MutableList<Quad>) {
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        val work = binary.clone()
        try {
            Imgproc.findContours(work, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
            val hulls = contours.filter {
                val r = Imgproc.boundingRect(it)
                r.width.toDouble() * r.height >= minArea
            }.map { hullOf(it) }
                .map { it to Imgproc.contourArea(it) }
                .filter { it.second >= minArea }
                .sortedByDescending { it.second }
            for ((index, pair) in hulls.withIndex()) {
                val hull = pair.first
                if (index < 5) {
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

    private val EPSILONS = doubleArrayOf(0.01, 0.02, 0.03, 0.045, 0.06, 0.08)

    private fun hullOf(contour: MatOfPoint): MatOfPoint2f {
        val idx = MatOfInt()
        Imgproc.convexHull(contour, idx)
        val pts = contour.toArray()
        val hullPts = idx.toArray().map { pts[it] }
        idx.release()
        return MatOfPoint2f(*hullPts.map { Point(it.x, it.y) }.toTypedArray())
    }

    /** Line in normal form: x*nx + y*ny = rho. */
    private class Line(val nx: Double, val ny: Double, val rho: Double, var len: Double)

    private fun lineQuads(edges: Mat, w: Int, h: Int, out: MutableList<Quad>) {
        val segs = Mat()
        val minDim = min(w, h).toDouble()
        Imgproc.HoughLinesP(edges, segs, 1.0, Math.PI / 360, 30, (minDim * 0.18).toInt().toDouble(), (minDim * 0.06).toInt().toDouble())
        val lines = ArrayList<Line>()
        for (i in 0 until segs.rows()) {
            val s = segs.get(i, 0) ?: continue
            val (x1, y1, x2, y2) = listOf(s[0], s[1], s[2], s[3])
            val th = atan2(y2 - y1, x2 - x1)
            var nx = -sin(th)
            var ny = cos(th)
            var rho = x1 * nx + y1 * ny
            if (rho < 0) { nx = -nx; ny = -ny; rho = -rho }
            lines.add(Line(nx, ny, rho, hypot(x2 - x1, y2 - y1)))
        }
        segs.release()
        // merge near-identical lines
        val merged = ArrayList<Line>()
        val cos4 = cos(Math.toRadians(4.0))
        for (l in lines.sortedByDescending { it.len }) {
            val m = merged.firstOrNull { l.nx * it.nx + l.ny * it.ny > cos4 && abs(l.rho - it.rho) < 6 }
            if (m != null) m.len += l.len else merged.add(Line(l.nx, l.ny, l.rho, l.len))
        }
        val hz = merged.filter { abs(it.ny) > abs(it.nx) }.sortedByDescending { it.len }.take(LINES_PER_FAMILY)
        val vt = merged.filter { abs(it.nx) >= abs(it.ny) }.sortedByDescending { it.len }.take(LINES_PER_FAMILY)
        val tol = 0.03
        for (i in hz.indices) for (j in i + 1 until hz.size) for (a in vt.indices) for (b in a + 1 until vt.size) {
            val p = listOf(
                intersect(hz[i], vt[a]), intersect(hz[i], vt[b]),
                intersect(hz[j], vt[a]), intersect(hz[j], vt[b]),
            )
            if (p.any { it == null }) continue
            val pts = p.map { it!! }
            if (pts.any { it.x < -tol * w || it.x > (1 + tol) * w || it.y < -tol * h || it.y > (1 + tol) * h }) continue
            val clamped = pts.map { Pt(it.x.coerceIn(0.0, w - 1.0), it.y.coerceIn(0.0, h - 1.0)) }
            out.add(Quad.fromUnordered(clamped))
        }
    }

    private fun intersect(l1: Line, l2: Line): Pt? {
        val det = l1.nx * l2.ny - l1.ny * l2.nx
        if (abs(det) < 1e-6) return null
        return Pt((l1.rho * l2.ny - l2.rho * l1.ny) / det, (l1.nx * l2.rho - l2.nx * l1.rho) / det)
    }

    private fun onFrameBorder(a: Pt, b: Pt, w: Int, h: Int): Boolean {
        val t = BORDER_PX
        return (a.x <= t && b.x <= t) || (a.y <= t && b.y <= t) ||
            (a.x >= w - 1 - t && b.x >= w - 1 - t) || (a.y >= h - 1 - t && b.y >= h - 1 - t)
    }

    /**
     * For each side, the fraction of sample points where a paper/background step exists near the
     * side. Returns null as soon as a side lies on the frame border or fails (early exit = fast).
     */
    fun sideQuality(paper: ByteArray, w: Int, h: Int, q: Quad): DoubleArray? {
        val pts = q.points
        val cx = pts.sumOf { it.x } / 4
        val cy = pts.sumOf { it.y } / 4
        val d = max(3.0, min(w, h) / 120.0)
        val r = max(3.0, min(w, h) * 0.015)
        val res = DoubleArray(4)
        for (i in 0 until 4) {
            val a = pts[i]
            val b = pts[(i + 1) % 4]
            if (onFrameBorder(a, b, w, h)) return null
            val len = a.dist(b)
            if (len < 1e-6) return null
            val dx = (b.x - a.x) / len
            val dy = (b.y - a.y) / len
            var nx = -dy
            var ny = dx
            val mx = (a.x + b.x) / 2
            val my = (a.y + b.y) / 2
            if ((cx - mx) * nx + (cy - my) * ny < 0) { nx = -nx; ny = -ny }
            var good = 0
            var vis = 0
            for (s in 0 until SAMPLES) {
                val t = 0.04 + 0.92 * s / (SAMPLES - 1)
                val px = a.x + (b.x - a.x) * t
                val py = a.y + (b.y - a.y) * t
                if (bil(paper, w, h, px, py) < 0) continue
                vis++
                var best = 0.0
                var o = -r
                while (o <= r) {
                    val vin = bil(paper, w, h, px + nx * (o + d), py + ny * (o + d))
                    val vout = bil(paper, w, h, px + nx * (o - d), py + ny * (o - d))
                    if (vin >= 0 && vout >= 0) {
                        val st = abs(vin - vout)
                        if (st > best) best = st
                        if (best >= STEP_MIN) break
                    }
                    o += 1.0
                }
                if (best >= STEP_MIN) good++
            }
            if (vis < SAMPLES / 2) return null
            val frac = good.toDouble() / vis
            if (frac < SIDE_GOOD_FRAC) return null
            res[i] = frac
        }
        return res
    }

    /** Bilinear sample; -1 outside the image. */
    private fun bil(img: ByteArray, w: Int, h: Int, x: Double, y: Double): Double {
        if (x < 0 || y < 0 || x > w - 1.001 || y > h - 1.001) return -1.0
        val x0 = x.toInt()
        val y0 = y.toInt()
        val fx = x - x0
        val fy = y - y0
        val i = y0 * w + x0
        val p00 = (img[i].toInt() and 0xFF).toDouble()
        val p10 = (img[i + 1].toInt() and 0xFF).toDouble()
        val p01 = (img[i + w].toInt() and 0xFF).toDouble()
        val p11 = (img[i + w + 1].toInt() and 0xFF).toDouble()
        return (p00 * (1 - fx) + p10 * fx) * (1 - fy) + (p01 * (1 - fx) + p11 * fx) * fy
    }

    /** Variance of the Laplacian inside the given pixel quad's bounding box: higher = sharper. */
    fun sharpness(gray: Mat, quad: Quad?): Double {
        val roi = if (quad != null) {
            val xs = quad.points.map { it.x }
            val ys = quad.points.map { it.y }
            val x0 = xs.min(); val x1 = xs.max(); val y0 = ys.min(); val y1 = ys.max()
            val mx = (x1 - x0) * 0.12; val my = (y1 - y0) * 0.12
            val rx = (x0 + mx).toInt().coerceIn(0, gray.cols() - 2)
            val ry = (y0 + my).toInt().coerceIn(0, gray.rows() - 2)
            val rw = ((x1 - mx) - rx).toInt().coerceIn(2, gray.cols() - rx)
            val rh = ((y1 - my) - ry).toInt().coerceIn(2, gray.rows() - ry)
            gray.submat(ry, ry + rh, rx, rx + rw)
        } else gray
        val lap = Mat()
        val mean = MatOfDouble()
        val std = MatOfDouble()
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

    /** Small normalized grayscale "fingerprint" of the page content (detects the same page). */
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
