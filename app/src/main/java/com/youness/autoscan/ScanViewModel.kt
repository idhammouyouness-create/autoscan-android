package com.youness.autoscan

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.Camera
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.youness.autoscan.pdf.PdfExporter
import com.youness.autoscan.pdf.PdfWriter
import com.youness.autoscan.scan.CaptureOutcome
import com.youness.autoscan.scan.Effect
import com.youness.autoscan.scan.Hint
import com.youness.autoscan.scan.MachineState
import com.youness.autoscan.scan.Observation
import com.youness.autoscan.scan.Phase
import com.youness.autoscan.scan.ScanStateMachine
import com.youness.autoscan.storage.PageStore
import com.youness.autoscan.storage.Session
import com.youness.autoscan.storage.SessionInfo
import com.youness.autoscan.storage.StorageFullException
import com.youness.autoscan.util.Haptics
import com.youness.autoscan.vision.EnhanceMode
import com.youness.autoscan.vision.FrameAnalyzer
import com.youness.autoscan.vision.FrameResult
import com.youness.autoscan.vision.ImageProcessor
import com.youness.autoscan.vision.Quad
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

enum class Screen { SETUP, SCAN, BUILDING, RESULT }

data class ScanSettings(
    val target: Int = 10,
    val countdownSec: Int = 3,
    val mode: EnhanceMode = EnhanceMode.DOCUMENT,
    val showDebug: Boolean = false,
)

data class UnfinishedSummary(val pages: Int, val target: Int)

data class ScanResult(val uri: Uri, val name: String, val pages: Int, val sizeBytes: Long)

class ScanViewModel(app: Application) : AndroidViewModel(app) {

    private val store = PageStore(app)
    private val haptics = Haptics(app)
    private val main = Handler(Looper.getMainLooper())

    val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val captureExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    var screen by mutableStateOf(Screen.SETUP); private set
    var settings by mutableStateOf(ScanSettings()); private set
    var machine by mutableStateOf<MachineState?>(null); private set
    var overlayQuad by mutableStateOf<Quad?>(null); private set
    var overlayStable by mutableStateOf(false); private set
    var lastThumb by mutableStateOf<Bitmap?>(null); private set
    var debugText by mutableStateOf(""); private set
    var unfinished by mutableStateOf<UnfinishedSummary?>(null); private set
    var result by mutableStateOf<ScanResult?>(null); private set
    var errorMessage by mutableStateOf<String?>(null); private set
    var torchOn by mutableStateOf(false); private set
    var buildMessage by mutableStateOf(""); private set

    private var sm: ScanStateMachine? = null
    private var session: Session? = null
    private var camera: Camera? = null
    private var imageCapture: ImageCapture? = null

    @Volatile private var analyzing = false
    @Volatile private var lastFingerprint: FloatArray? = null
    private var pendingFingerprint: FloatArray? = null

    val analyzer = FrameAnalyzer(
        isActive = { analyzing },
        lastPageFingerprint = { lastFingerprint },
        onResult = { r -> main.post { onFrame(r) } },
    )

    init {
        refreshUnfinished()
    }

    // ---------- setup ----------

    fun updateSettings(s: ScanSettings) {
        settings = s.copy(target = s.target.coerceIn(1, 500), countdownSec = if (s.countdownSec == 5) 5 else 3)
    }

    private fun refreshUnfinished() {
        val s = store.findUnfinished()
        store.cleanup(s)
        unfinished = s?.let { UnfinishedSummary(it.pageFiles().size, it.info.target) }
    }

    fun startScan() {
        errorMessage = null
        result = null
        val info = SessionInfo(settings.target, settings.countdownSec, settings.mode, System.currentTimeMillis())
        session = store.create(info)
        lastFingerprint = null
        lastThumb = null
        startMachine(captured = 0)
    }

    fun resumeUnfinished() {
        val s = store.findUnfinished() ?: return refreshUnfinished()
        session = s
        val info = s.info
        settings = settings.copy(target = info.target, countdownSec = info.countdownSec, mode = info.mode)
        lastFingerprint = null
        lastThumb = null
        val pages = s.pageFiles().size
        if (pages >= info.target) {
            buildPdf()
        } else {
            startMachine(captured = pages)
        }
    }

    fun buildUnfinishedNow() {
        val s = store.findUnfinished() ?: return refreshUnfinished()
        session = s
        buildPdf()
    }

    fun discardUnfinished() {
        store.findUnfinished()?.delete()
        refreshUnfinished()
    }

    private fun startMachine(captured: Int) {
        sm = ScanStateMachine(settings.target, settings.countdownSec * 1000L, initialCaptured = captured)
        machine = sm!!.state
        analyzer.reset()
        analyzing = true
        screen = Screen.SCAN
    }

    // ---------- camera ----------

    fun onCameraBound(cam: Camera, capture: ImageCapture) {
        camera = cam
        imageCapture = capture
        torchOn = false
    }

    fun onCameraUnbound() {
        camera = null
        imageCapture = null
        torchOn = false
    }

    fun onCameraError(message: String) {
        sm?.let {
            it.pause(Hint.CAMERA_ERROR)
            publish()
        }
        errorMessage = message
    }

    fun toggleTorch() {
        val cam = camera ?: return
        if (!cam.cameraInfo.hasFlashUnit()) return
        torchOn = !torchOn
        cam.cameraControl.enableTorch(torchOn)
    }

    /** The screen went to background (call, home button...): pause, never capture blindly. */
    fun onAppBackground() {
        val m = sm ?: return
        if (screen == Screen.SCAN && m.isActive) {
            m.pause()
            publish()
        }
    }

    // ---------- scanning ----------

    private fun onFrame(r: FrameResult) {
        val m = sm ?: return
        if (screen != Screen.SCAN) return
        overlayQuad = r.quad
        overlayStable = r.stable && r.sharp && r.motionLow
        if (settings.showDebug) {
            debugText = String.format(
                Locale.ROOT,
                "sharp %.0f / peak %.0f  motion %.1f  edges %.2f  sim %s  %dx%d",
                r.sharpness, r.peakSharpness, r.motion, r.edgeSupport,
                r.similarityToLast?.let { String.format(Locale.ROOT, "%.2f", it) } ?: "-",
                r.frameWidth, r.frameHeight,
            )
        }
        val effects = m.onFrame(
            Observation(
                timeMs = r.timeMs,
                hasDoc = r.quad != null,
                docStable = r.stable,
                sharp = r.sharp,
                motionLow = r.motionLow,
                similarityToLast = r.similarityToLast,
            ),
        )
        publish()
        handle(effects)
    }

    fun pause() {
        sm?.pause(); publish()
    }

    fun resume() {
        errorMessage = null
        analyzer.reset()
        sm?.resume(SystemClock.elapsedRealtime()); publish()
    }

    fun manualCapture() {
        val m = sm ?: return
        handle(m.manualCapture()).also { publish() }
    }

    /** Cancel and discard every page of this session. */
    fun cancelAndDiscard() {
        sm?.cancel()
        analyzing = false
        session?.delete()
        session = null
        sm = null
        machine = null
        overlayQuad = null
        screen = Screen.SETUP
        refreshUnfinished()
    }

    /** Stop now and make a PDF with the pages captured so far. */
    fun finishWithCapturedPages() {
        val count = session?.pageFiles()?.size ?: 0
        if (count == 0) return cancelAndDiscard()
        sm?.cancel()
        buildPdf()
    }

    private fun publish() {
        val m = sm ?: return
        machine = m.state
        analyzing = m.isActive
    }

    private fun handle(effects: List<Effect>) {
        for (e in effects) when (e) {
            Effect.VibrateOnce -> haptics.once()
            Effect.VibrateTwice -> haptics.twice()
            Effect.Capture -> doCapture()
            Effect.Completed -> buildPdf()
        }
    }

    private fun doCapture() {
        val capture = imageCapture
        val sess = session
        if (capture == null || sess == null) {
            finishCapture(CaptureOutcome.FAILED)
            return
        }
        if (!store.hasRoomForPage()) {
            finishCapture(CaptureOutcome.STORAGE_FULL)
            return
        }
        analyzing = false
        pendingFingerprint = analyzer.currentFingerprint
        val hint = analyzer.currentQuad
        val mode = settings.mode
        val index = sess.pageFiles().size + 1

        capture.takePicture(captureExecutor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                val bytes: ByteArray
                val rotation: Int
                try {
                    val buffer = image.planes[0].buffer
                    buffer.rewind()
                    bytes = ByteArray(buffer.remaining())
                    buffer.get(bytes)
                    rotation = image.imageInfo.rotationDegrees
                } catch (t: Throwable) {
                    Log.e(TAG, "read capture failed", t)
                    main.post { finishCapture(CaptureOutcome.FAILED) }
                    return
                } finally {
                    image.close()
                }
                viewModelScope.launch(Dispatchers.Default) {
                    val outcome = processAndSave(bytes, rotation, hint, mode, sess, index)
                    withContext(Dispatchers.Main) { finishCapture(outcome) }
                }
            }

            override fun onError(exception: ImageCaptureException) {
                Log.e(TAG, "capture error", exception)
                main.post { finishCapture(CaptureOutcome.FAILED) }
            }
        })
    }

    private fun processAndSave(
        bytes: ByteArray,
        rotation: Int,
        hint: Quad?,
        mode: EnhanceMode,
        sess: Session,
        index: Int,
    ): CaptureOutcome {
        var page: Bitmap? = null
        return try {
            val rgba = ImageProcessor.decodeJpeg(bytes, rotation)
            val processed = try {
                ImageProcessor.process(rgba, hint, mode)
            } finally {
                rgba.release()
            }
            page = processed.bitmap
            if (processed.sharpness < MIN_PAGE_SHARPNESS) {
                Log.w(TAG, "rejected blurry page: ${processed.sharpness}")
                return CaptureOutcome.BLURRY
            }
            sess.savePage(index, processed.bitmap)
            val thumbW = 240
            val thumbH = (processed.bitmap.height * thumbW.toFloat() / processed.bitmap.width).toInt().coerceAtLeast(1)
            val thumb = Bitmap.createScaledBitmap(processed.bitmap, thumbW, thumbH, true)
            main.post { lastThumb = thumb }
            CaptureOutcome.SAVED
        } catch (e: StorageFullException) {
            CaptureOutcome.STORAGE_FULL
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "out of memory while processing page", e)
            CaptureOutcome.FAILED
        } catch (e: Exception) {
            Log.e(TAG, "processing failed", e)
            CaptureOutcome.FAILED
        } finally {
            page?.recycle()
        }
    }

    private fun finishCapture(outcome: CaptureOutcome) {
        val m = sm ?: return
        if (outcome == CaptureOutcome.SAVED) {
            lastFingerprint = pendingFingerprint
        }
        if (outcome == CaptureOutcome.STORAGE_FULL) {
            errorMessage = "مساحة التخزين ممتلئة. حرّر مساحة ثم اضغط استئناف."
        }
        analyzer.reset()
        val effects = m.onCaptureResult(outcome, SystemClock.elapsedRealtime())
        publish()
        handle(effects)
    }

    // ---------- PDF ----------

    private fun buildPdf() {
        val sess = session ?: return
        analyzing = false
        screen = Screen.BUILDING
        buildMessage = "جارٍ إنشاء ملف PDF…"
        errorMessage = null
        viewModelScope.launch(Dispatchers.IO) {
            val app = getApplication<Application>()
            val tmp = File(app.cacheDir, "build_${System.currentTimeMillis()}.pdf")
            try {
                val pages = sess.pageFiles()
                if (pages.isEmpty()) throw IOException("لا توجد صفحات")
                PdfWriter.write(pages, tmp)
                PdfExporter.verify(tmp, pages.size)
                val name = "Scan_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.ROOT).format(Date()) + ".pdf"
                val uri = PdfExporter.exportToDocuments(app, tmp, name)
                val size = tmp.length()
                // Only now, with the PDF verified and saved, the page images can go.
                sess.delete()
                withContext(Dispatchers.Main) {
                    result = ScanResult(uri, name, pages.size, size)
                    session = null
                    sm = null
                    machine = null
                    screen = Screen.RESULT
                    refreshUnfinished()
                }
            } catch (e: Exception) {
                Log.e(TAG, "PDF build failed", e)
                withContext(Dispatchers.Main) {
                    errorMessage = "تعذّر إنشاء PDF: ${e.message}. الصور محفوظة ويمكنك إعادة المحاولة."
                }
            } finally {
                tmp.delete()
            }
        }
    }

    fun retryBuild() = buildPdf()

    fun backToSetup() {
        screen = Screen.SETUP
        errorMessage = null
        refreshUnfinished()
    }

    override fun onCleared() {
        analyzing = false
        analysisExecutor.shutdown()
        captureExecutor.shutdown()
    }

    companion object {
        private const val TAG = "AutoScan"
        /** Pages below this Laplacian variance (at 1000 px width) are rejected as blurry. */
        const val MIN_PAGE_SHARPNESS = 25.0
    }
}
