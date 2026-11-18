package com.youness.autoscan.vision

import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max

/** Result of analyzing one preview frame. Quad is normalized (0..1) in the upright frame. */
data class FrameResult(
    val timeMs: Long,
    val quad: Quad?,
    val stable: Boolean,
    val sharpness: Double,
    val peakSharpness: Double,
    val sharp: Boolean,
    val motion: Double,
    val motionLow: Boolean,
    val similarityToLast: Double?,
    val edgeSupport: Double,
    val frameWidth: Int,
    val frameHeight: Int,
)

/**
 * CameraX analyzer: runs on a background thread for every preview frame (latest only).
 * Uses only the luminance (Y) plane, which is all the detector needs: cheap on battery.
 */
class FrameAnalyzer(
    private val isActive: () -> Boolean,
    private val lastPageFingerprint: () -> FloatArray?,
    private val onResult: (FrameResult) -> Unit,
) : ImageAnalysis.Analyzer {

    companion object {
        /** Absolute floor of Laplacian variance for a sharp frame (analysis resolution). */
        const val MIN_SHARPNESS = 18.0
        /** Frame must reach this fraction of the recent best sharpness. */
        const val RELATIVE_SHARPNESS = 0.65
        /** Mean absolute pixel change between frames (0..255) below which the scene is still. */
        const val MOTION_LOW = 2.5
        private const val MIN_FRAME_INTERVAL_MS = 55L // ~18 fps max: saves battery
    }

    private val tracker = QuadTracker()
    private var prevThumb: Mat? = null
    private var peakSharp = 0.0
    private var lastProcessed = 0L
    private var rowBuf = ByteArray(0)
    private var fullBuf = ByteArray(0)

    /** Fingerprint of the sheet currently in view (null if none). Read from any thread. */
    @Volatile
    var currentFingerprint: FloatArray? = null
        private set

    /** Last normalized quad, read by the capture code as a hint. */
    @Volatile
    var currentQuad: Quad? = null
        private set

    @Volatile
    private var resetRequested = false

    fun reset() {
        resetRequested = true
    }

    override fun analyze(image: ImageProxy) {
        try {
            val now = SystemClock.elapsedRealtime()
            if (!isActive()) {
                // Keep state fresh so resuming does not use stale data.
                resetRequested = true
                return
            }
            if (now - lastProcessed < MIN_FRAME_INTERVAL_MS) return
            lastProcessed = now
            if (resetRequested) {
                resetRequested = false
                tracker.reset()
                prevThumb?.release(); prevThumb = null
                peakSharp = 0.0
            }
            val gray = toUprightGray(image)
            try {
                process(gray, now)
            } finally {
                gray.release()
            }
        } catch (t: Throwable) {
            android.util.Log.e("FrameAnalyzer", "analysis failed", t)
        } finally {
            image.close()
        }
    }

    private fun process(gray: Mat, now: Long) {
        val w = gray.cols()
        val h = gray.rows()

        // Motion: compare tiny thumbnails of consecutive frames.
        val thumb = Mat()
        Imgproc.resize(gray, thumb, Size(48.0, 64.0), 0.0, 0.0, Imgproc.INTER_AREA)
        var motion = 0.0
        prevThumb?.let { prev ->
            val diff = Mat()
            Core.absdiff(prev, thumb, diff)
            motion = Core.mean(diff).`val`[0]
            diff.release()
        }
        prevThumb?.release()
        prevThumb = thumb

        val best = DocumentDetector.detect(gray)
        val rawNorm = best?.quad?.scale(1.0 / w, 1.0 / h)
        val smoothed = tracker.update(rawNorm)
        val pixelQuad = smoothed?.scale(w.toDouble(), h.toDouble())

        val sharpness = if (pixelQuad != null) DocumentDetector.sharpness(gray, pixelQuad) else 0.0
        if (pixelQuad != null) {
            peakSharp = max(peakSharp * 0.97, sharpness)
        } else {
            peakSharp *= 0.9
        }
        val sharp = pixelQuad != null &&
            sharpness >= MIN_SHARPNESS &&
            sharpness >= peakSharp * RELATIVE_SHARPNESS

        var similarity: Double? = null
        if (pixelQuad != null) {
            val fp = DocumentDetector.fingerprint(gray, pixelQuad)
            currentFingerprint = fp
            lastPageFingerprint()?.let { similarity = DocumentDetector.similarity(fp, it) }
        } else {
            currentFingerprint = null
        }
        currentQuad = smoothed

        onResult(
            FrameResult(
                timeMs = now,
                quad = smoothed,
                stable = tracker.isStable,
                sharpness = sharpness,
                peakSharpness = peakSharp,
                sharp = sharp,
                motion = motion,
                motionLow = motion < MOTION_LOW,
                similarityToLast = similarity,
                edgeSupport = best?.edgeSupport ?: 0.0,
                frameWidth = w,
                frameHeight = h,
            ),
        )
    }

    /** Copies the Y plane into a Mat and rotates it to match what the user sees. */
    private fun toUprightGray(image: ImageProxy): Mat {
        val plane = image.planes[0]
        val buffer = plane.buffer
        val w = image.width
        val h = image.height
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val mat = Mat(h, w, CvType.CV_8UC1)
        buffer.rewind()
        if (pixelStride == 1 && rowStride == w && buffer.remaining() >= w * h) {
            if (fullBuf.size != w * h) fullBuf = ByteArray(w * h)
            buffer.get(fullBuf, 0, w * h)
            mat.put(0, 0, fullBuf)
        } else {
            if (rowBuf.size < rowStride) rowBuf = ByteArray(rowStride)
            if (fullBuf.size != w * h) fullBuf = ByteArray(w * h)
            for (r in 0 until h) {
                buffer.position(r * rowStride)
                val len = minOf(rowStride, buffer.remaining())
                buffer.get(rowBuf, 0, len)
                if (pixelStride == 1) {
                    System.arraycopy(rowBuf, 0, fullBuf, r * w, w)
                } else {
                    for (c in 0 until w) fullBuf[r * w + c] = rowBuf[c * pixelStride]
                }
            }
            mat.put(0, 0, fullBuf)
        }
        val rotation = image.imageInfo.rotationDegrees
        if (rotation == 0) return mat
        val out = Mat()
        val code = when (rotation) {
            90 -> Core.ROTATE_90_CLOCKWISE
            180 -> Core.ROTATE_180
            else -> Core.ROTATE_90_COUNTERCLOCKWISE
        }
        Core.rotate(mat, out, code)
        mat.release()
        return out
    }
}
