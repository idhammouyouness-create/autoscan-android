package com.youness.autoscan.vision

import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

data class Pt(val x: Double, val y: Double) {
    operator fun minus(o: Pt) = Pt(x - o.x, y - o.y)
    operator fun plus(o: Pt) = Pt(x + o.x, y + o.y)
    operator fun times(s: Double) = Pt(x * s, y * s)
    fun dist(o: Pt) = hypot(x - o.x, y - o.y)
    fun length() = hypot(x, y)
}

/**
 * A quadrilateral with corners ordered clockwise: top-left, top-right, bottom-right, bottom-left
 * (image coordinates, y grows downward). Units depend on context (pixels or normalized 0..1).
 */
data class Quad(val tl: Pt, val tr: Pt, val br: Pt, val bl: Pt) {

    val points: List<Pt> get() = listOf(tl, tr, br, bl)

    fun scale(sx: Double, sy: Double) = Quad(
        Pt(tl.x * sx, tl.y * sy), Pt(tr.x * sx, tr.y * sy),
        Pt(br.x * sx, br.y * sy), Pt(bl.x * sx, bl.y * sy),
    )

    fun area(): Double {
        val p = points
        var s = 0.0
        for (i in 0 until 4) {
            val a = p[i]
            val b = p[(i + 1) % 4]
            s += a.x * b.y - b.x * a.y
        }
        return abs(s) / 2.0
    }

    fun maxCornerDistance(o: Quad): Double = points.zip(o.points).maxOf { (a, b) -> a.dist(b) }

    fun lerp(o: Quad, t: Double): Quad {
        fun l(a: Pt, b: Pt) = Pt(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
        return Quad(l(tl, o.tl), l(tr, o.tr), l(br, o.br), l(bl, o.bl))
    }

    fun isConvex(): Boolean {
        val p = points
        var sign = 0
        for (i in 0 until 4) {
            val a = p[i]
            val b = p[(i + 1) % 4]
            val c = p[(i + 2) % 4]
            val cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
            if (abs(cross) < 1e-12) return false
            val s = if (cross > 0) 1 else -1
            if (sign == 0) sign = s else if (s != sign) return false
        }
        return true
    }

    /** Interior angles in degrees, one per corner in TL, TR, BR, BL order. */
    fun interiorAnglesDeg(): List<Double> {
        val p = points
        return (0 until 4).map { i ->
            val prev = p[(i + 3) % 4]
            val cur = p[i]
            val next = p[(i + 1) % 4]
            val v1 = prev - cur
            val v2 = next - cur
            val d = v1.length() * v2.length()
            if (d < 1e-12) 0.0 else {
                val c = ((v1.x * v2.x + v1.y * v2.y) / d).coerceIn(-1.0, 1.0)
                Math.toDegrees(acos(c))
            }
        }
    }

    /** min/max ratio of opposite sides (1.0 = parallelogram-like). */
    fun oppositeSideRatio(): Double {
        val top = tl.dist(tr)
        val bottom = bl.dist(br)
        val left = tl.dist(bl)
        val right = tr.dist(br)
        val r1 = min(top, bottom) / max(max(top, bottom), 1e-9)
        val r2 = min(left, right) / max(max(left, right), 1e-9)
        return min(r1, r2)
    }

    companion object {
        /** Orders 4 arbitrary points clockwise starting from the top-left-most point. */
        fun fromUnordered(pts: List<Pt>): Quad {
            require(pts.size == 4) { "need 4 points" }
            val cx = pts.sumOf { it.x } / 4.0
            val cy = pts.sumOf { it.y } / 4.0
            val sorted = pts.sortedBy { atan2(it.y - cy, it.x - cx) }
            val start = sorted.indices.minBy { sorted[it].x + sorted[it].y }
            val r = (0 until 4).map { sorted[(start + it) % 4] }
            return Quad(r[0], r[1], r[2], r[3])
        }
    }
}
