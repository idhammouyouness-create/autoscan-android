package com.youness.autoscan

import com.youness.autoscan.vision.Pt
import com.youness.autoscan.vision.Quad
import com.youness.autoscan.vision.QuadTracker
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class QuadTest {

    @Test
    fun ordersCornersClockwiseFromTopLeft() {
        val q = Quad.fromUnordered(listOf(Pt(90.0, 110.0), Pt(10.0, 12.0), Pt(12.0, 100.0), Pt(95.0, 8.0)))
        assertEquals(Pt(10.0, 12.0), q.tl)
        assertEquals(Pt(95.0, 8.0), q.tr)
        assertEquals(Pt(90.0, 110.0), q.br)
        assertEquals(Pt(12.0, 100.0), q.bl)
    }

    @Test
    fun ordersRotatedSheet() {
        // ~30 degree rotated rectangle
        val q = Quad.fromUnordered(listOf(Pt(50.0, 0.0), Pt(100.0, 30.0), Pt(70.0, 90.0), Pt(20.0, 60.0)))
        assertTrue(q.isConvex())
        assertEquals(4, q.points.toSet().size)
        assertTrue(q.area() > 0)
    }

    @Test
    fun areaAndAngles() {
        val q = Quad(Pt(0.0, 0.0), Pt(10.0, 0.0), Pt(10.0, 20.0), Pt(0.0, 20.0))
        assertEquals(200.0, q.area(), 1e-9)
        q.interiorAnglesDeg().forEach { assertEquals(90.0, it, 1e-6) }
        assertEquals(1.0, q.oppositeSideRatio(), 1e-9)
    }

    @Test
    fun nonConvexDetected() {
        val q = Quad(Pt(0.0, 0.0), Pt(10.0, 0.0), Pt(3.0, 3.0), Pt(0.0, 10.0))
        assertFalse(q.isConvex())
    }

    @Test
    fun trackerBecomesStableOnlyWhenStill() {
        val tr = QuadTracker()
        val base = Quad(Pt(0.1, 0.1), Pt(0.9, 0.1), Pt(0.9, 0.9), Pt(0.1, 0.9))
        repeat(10) { tr.update(base) }
        assertTrue(tr.isStable)
        // moving sheet
        var shift = 0.0
        repeat(10) {
            shift += 0.03
            tr.update(base.scale(1.0, 1.0).let { Quad(it.tl + Pt(shift, 0.0), it.tr + Pt(shift, 0.0), it.br + Pt(shift, 0.0), it.bl + Pt(shift, 0.0)) })
        }
        assertFalse(tr.isStable)
    }

    @Test
    fun trackerForgetsMissingDocument() {
        val tr = QuadTracker()
        val base = Quad(Pt(0.1, 0.1), Pt(0.9, 0.1), Pt(0.9, 0.9), Pt(0.1, 0.9))
        repeat(10) { tr.update(base) }
        repeat(10) { tr.update(null) }
        assertNull(tr.smoothed)
        assertFalse(tr.isStable)
    }
}
