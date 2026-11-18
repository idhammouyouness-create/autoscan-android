package com.youness.autoscan.ui

import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.youness.autoscan.ScanViewModel
import com.youness.autoscan.vision.Quad

/** Aspect ratio (width / height) of the upright portrait camera frame (4:3 sensor). */
private const val FRAME_ASPECT = 3f / 4f

@Composable
fun CameraPreview(vm: ScanViewModel, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            keepScreenOn = true
        }
    }

    DisposableEffect(lifecycleOwner) {
        val future = ProcessCameraProvider.getInstance(context)
        var provider: ProcessCameraProvider? = null
        future.addListener({
            try {
                val p = future.get()
                provider = p
                val ratio43 = AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY

                val preview = Preview.Builder()
                    .setResolutionSelector(ResolutionSelector.Builder().setAspectRatioStrategy(ratio43).build())
                    .build()
                preview.setSurfaceProvider(previewView.surfaceProvider)

                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(
                        ResolutionSelector.Builder()
                            .setAspectRatioStrategy(ratio43)
                            .setResolutionStrategy(
                                ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER),
                            )
                            .build(),
                    )
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                analysis.setAnalyzer(vm.analysisExecutor, vm.analyzer)

                // ~12 MP is plenty for documents and keeps processing fast and memory-safe.
                val capture = ImageCapture.Builder()
                    .setResolutionSelector(
                        ResolutionSelector.Builder()
                            .setAspectRatioStrategy(ratio43)
                            .setResolutionStrategy(
                                ResolutionStrategy(Size(4032, 3024), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER),
                            )
                            .build(),
                    )
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .setJpegQuality(95)
                    .build()

                p.unbindAll()
                val camera = p.bindToLifecycle(
                    lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis, capture,
                )
                camera.cameraInfo.cameraState.observe(lifecycleOwner) { state ->
                    state.error?.let { vm.onCameraError("خطأ في الكاميرا (${it.code}). أغلق التطبيقات الأخرى التي تستخدم الكاميرا ثم اضغط استئناف.") }
                }
                vm.onCameraBound(camera, capture)
            } catch (e: Exception) {
                vm.onCameraError("تعذّر تشغيل الكاميرا: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(context))

        onDispose {
            provider?.unbindAll()
            vm.onCameraUnbound()
        }
    }

    Box(modifier) {
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
        QuadOverlay(vm.overlayQuad, vm.overlayStable, Modifier.fillMaxSize())
    }
}

@Composable
private fun QuadOverlay(quad: Quad?, ready: Boolean, modifier: Modifier) {
    Canvas(modifier) {
        val q = quad ?: return@Canvas
        // Same geometry as PreviewView FIT_CENTER for a 3:4 portrait frame.
        val viewAspect = size.width / size.height
        val cw: Float
        val ch: Float
        if (viewAspect > FRAME_ASPECT) {
            ch = size.height; cw = ch * FRAME_ASPECT
        } else {
            cw = size.width; ch = cw / FRAME_ASPECT
        }
        val ox = (size.width - cw) / 2f
        val oy = (size.height - ch) / 2f
        fun map(x: Double, y: Double) = Offset(ox + x.toFloat() * cw, oy + y.toFloat() * ch)
        val pts = q.points.map { map(it.x, it.y) }
        val path = Path().apply {
            moveTo(pts[0].x, pts[0].y)
            for (i in 1 until 4) lineTo(pts[i].x, pts[i].y)
            close()
        }
        val color = if (ready) Color(0xFF00E676) else Color(0xFFFFC107)
        drawPath(path, color.copy(alpha = 0.18f))
        drawPath(path, color, style = Stroke(width = 6f))
        pts.forEach { drawCircle(color, radius = 14f, center = it) }
    }
}
