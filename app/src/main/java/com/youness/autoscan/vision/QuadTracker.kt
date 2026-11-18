package com.youness.autoscan.vision

/**
 * Smooths the detected document outline across frames and decides when it is stable.
 * All coordinates are normalized (0..1) so thresholds do not depend on resolution.
 */
class QuadTracker(
    /** Max corner movement (fraction of frame) between frames still considered "not moving". */
    private val stillThreshold: Double = 0.015,
    /** Consecutive still frames required before the outline is considered stable. */
    private val framesForStable: Int = 5,
    /** Frames a detection may be missing before the document is considered gone. */
    private val maxMissingFrames: Int = 3,
    /** Exponential smoothing factor for the displayed outline. */
    private val smoothing: Double = 0.45,
) {
    var smoothed: Quad? = null
        private set
    var stableFrames: Int = 0
        private set
    private var missing = 0
    private var lastRaw: Quad? = null

    /** Returns the current (smoothed) document outline, or null if no document. */
    fun update(raw: Quad?): Quad? {
        if (raw == null) {
            missing++
            stableFrames = 0
            if (missing > maxMissingFrames) {
                smoothed = null
                lastRaw = null
            }
            return smoothed
        }
        missing = 0
        val prevRaw = lastRaw
        lastRaw = raw
        val prev = smoothed
        if (prev == null || prevRaw == null) {
            smoothed = raw
            stableFrames = 0
            return raw
        }
        val moved = prevRaw.maxCornerDistance(raw)
        stableFrames = if (moved < stillThreshold) stableFrames + 1 else 0
        // Big jump: follow immediately; small jitter: smooth it out.
        smoothed = if (prev.maxCornerDistance(raw) > 0.08) raw else prev.lerp(raw, smoothing)
        return smoothed
    }

    val hasDocument: Boolean get() = smoothed != null
    val isStable: Boolean get() = smoothed != null && missing == 0 && stableFrames >= framesForStable

    fun reset() {
        smoothed = null
        lastRaw = null
        stableFrames = 0
        missing = 0
    }
}
