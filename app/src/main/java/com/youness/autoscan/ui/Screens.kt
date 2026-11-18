package com.youness.autoscan.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.youness.autoscan.Screen
import com.youness.autoscan.ScanViewModel
import com.youness.autoscan.scan.Hint
import com.youness.autoscan.scan.MachineState
import com.youness.autoscan.scan.Phase
import com.youness.autoscan.vision.EnhanceMode
import java.util.Locale

private val Accent = Color(0xFF42A5F5)

@Composable
fun AppTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(primary = Accent, secondary = Color(0xFFFFC107)),
        content = content,
    )
}

@Composable
fun AppRoot(vm: ScanViewModel, openCvReady: Boolean) {
    Surface(Modifier.fillMaxSize(), color = Color.Black) {
        if (!openCvReady) {
            CenterMessage("تعذّر تحميل مكتبة معالجة الصور. أعد تثبيت التطبيق.")
            return@Surface
        }
        when (vm.screen) {
            Screen.SETUP -> SetupScreen(vm)
            Screen.SCAN -> ScanScreen(vm)
            Screen.BUILDING -> BuildingScreen(vm)
            Screen.RESULT -> ResultScreen(vm)
        }
    }
}

@Composable
private fun CenterMessage(text: String) {
    Box(Modifier.fillMaxSize().padding(24.dp), contentAlignment = Alignment.Center) {
        Text(text, color = Color.White, textAlign = TextAlign.Center, fontSize = 18.sp)
    }
}

// ---------------------------------------------------------------- setup

@Composable
private fun SetupScreen(vm: ScanViewModel) {
    val context = LocalContext.current
    val s = vm.settings
    var permissionDenied by remember { mutableStateOf(false) }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) vm.startScan() else permissionDenied = true
    }
    var countText by remember { mutableStateOf(s.target.toString()) }

    Column(
        Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding()
            .verticalScroll(rememberScrollState()).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp),
    ) {
        Text("المسح التلقائي المتتابع", color = Color.White, fontSize = 26.sp, fontWeight = FontWeight.Bold)

        vm.unfinished?.let { u ->
            Card {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("جلسة غير مكتملة: ${u.pages} من ${u.target} صفحات", fontWeight = FontWeight.Bold)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(onClick = {
                            if (hasCamera(context)) vm.resumeUnfinished() else launcher.launch(Manifest.permission.CAMERA)
                        }) { Text("متابعة") }
                        OutlinedButton(onClick = { vm.buildUnfinishedNow() }) { Text("PDF بما تم") }
                        TextButton(onClick = { vm.discardUnfinished() }) { Text("حذف") }
                    }
                }
            }
        }

        Label("عدد الصفحات")
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            StepButton("−") {
                val n = (s.target - 1).coerceAtLeast(1)
                vm.updateSettings(s.copy(target = n)); countText = n.toString()
            }
            androidx.compose.material3.OutlinedTextField(
                value = countText,
                onValueChange = { t ->
                    val digits = t.filter { it.isDigit() }.take(3)
                    countText = digits
                    digits.toIntOrNull()?.let { vm.updateSettings(s.copy(target = it)) }
                },
                singleLine = true,
                modifier = Modifier.width(110.dp),
                keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(
                    keyboardType = androidx.compose.ui.text.input.KeyboardType.Number,
                ),
            )
            StepButton("+") {
                val n = (s.target + 1).coerceAtMost(500)
                vm.updateSettings(s.copy(target = n)); countText = n.toString()
            }
        }

        Label("مهلة التقاط الصفحة الجديدة")
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            listOf(3, 5).forEach { sec ->
                FilterChip(
                    selected = s.countdownSec == sec,
                    onClick = { vm.updateSettings(s.copy(countdownSec = sec)) },
                    label = { Text("$sec ثوانٍ", fontSize = 16.sp) },
                )
            }
        }

        Label("نوع الصورة")
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            val modes = listOf(
                EnhanceMode.DOCUMENT to "مستند محسّن (موصى به)",
                EnhanceMode.GRAYSCALE to "رمادي",
                EnhanceMode.BLACK_WHITE to "أبيض وأسود",
                EnhanceMode.ORIGINAL to "ألوان أصلية بلا تحسين",
            )
            modes.chunked(2).forEach { row ->
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    row.forEach { (m, label) ->
                        FilterChip(selected = s.mode == m, onClick = { vm.updateSettings(s.copy(mode = m)) }, label = { Text(label) })
                    }
                }
            }
        }

        Row(verticalAlignment = Alignment.CenterVertically) {
            Switch(checked = s.showDebug, onCheckedChange = { vm.updateSettings(s.copy(showDebug = it)) })
            Spacer(Modifier.width(10.dp))
            Text("عرض معلومات تقنية أثناء المسح", color = Color.LightGray)
        }

        if (permissionDenied) {
            Text("يحتاج التطبيق إذن الكاميرا. فعّله من الإعدادات ← التطبيقات ← المسح التلقائي ← الأذونات.", color = Color(0xFFFF8A80))
        }

        Button(
            onClick = {
                if (hasCamera(context)) vm.startScan() else launcher.launch(Manifest.permission.CAMERA)
            },
            modifier = Modifier.fillMaxWidth().height(58.dp),
        ) { Text("بدء المسح التلقائي", fontSize = 19.sp) }

        Text(
            "نصيحة: يجب أن تظهر الصفحة كاملة بحوافها الأربعة داخل الإطار. السطح الداكن والإضاءة الجيدة يحسّنان الكشف.",
            color = Color.Gray, fontSize = 14.sp,
        )
    }
}

private fun hasCamera(context: android.content.Context) =
    ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

@Composable
private fun Label(t: String) = Text(t, color = Color.White, fontSize = 17.sp, fontWeight = FontWeight.SemiBold)

@Composable
private fun StepButton(t: String, onClick: () -> Unit) {
    OutlinedButton(onClick = onClick, modifier = Modifier.size(56.dp), shape = RoundedCornerShape(12.dp)) {
        Text(t, fontSize = 22.sp)
    }
}

// ---------------------------------------------------------------- scan

@Composable
private fun ScanScreen(vm: ScanViewModel) {
    val m = vm.machine ?: return
    var confirmCancel by remember { mutableStateOf(false) }
    BackHandler { confirmCancel = true }

    Box(Modifier.fillMaxSize()) {
        CameraPreview(vm, Modifier.fillMaxSize())

        Column(
            Modifier.align(Alignment.TopCenter).statusBarsPadding().padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            StatusChip(m)
            if (vm.settings.showDebug && vm.debugText.isNotEmpty()) {
                Text(
                    vm.debugText, color = Color.White, fontSize = 11.sp,
                    modifier = Modifier.padding(top = 6.dp).background(Color(0x99000000)).padding(4.dp),
                )
            }
            vm.errorMessage?.let {
                Text(
                    it, color = Color.White, fontSize = 14.sp,
                    modifier = Modifier.padding(top = 6.dp).background(Color(0xCCB71C1C), RoundedCornerShape(8.dp)).padding(8.dp),
                )
            }
        }

        if (m.phase == Phase.COUNTDOWN) {
            val secs = ((m.countdownLeftMs + 999) / 1000).toInt()
            Text(
                secs.toString(), color = Color.White, fontSize = 96.sp, fontWeight = FontWeight.Bold,
                modifier = Modifier.align(Alignment.Center).background(Color(0x66000000), RoundedCornerShape(24.dp)).padding(horizontal = 36.dp),
            )
        }

        BottomPanel(vm, m, onCancel = { confirmCancel = true }, modifier = Modifier.align(Alignment.BottomCenter))
    }

    if (confirmCancel) {
        val pages = m.captured
        AlertDialog(
            onDismissRequest = { confirmCancel = false },
            title = { Text("إيقاف المسح؟") },
            text = { Text(if (pages > 0) "تم مسح $pages صفحة. ماذا تريد أن تفعل بها؟" else "لم تُمسح أي صفحة بعد.") },
            confirmButton = {
                if (pages > 0) {
                    TextButton(onClick = { confirmCancel = false; vm.finishWithCapturedPages() }) { Text("إنشاء PDF بما تم") }
                }
            },
            dismissButton = {
                Row {
                    TextButton(onClick = { confirmCancel = false }) { Text("متابعة المسح") }
                    TextButton(onClick = { confirmCancel = false; vm.cancelAndDiscard() }) {
                        Text("إلغاء وحذف", color = Color(0xFFFF8A80))
                    }
                }
            },
        )
    }
}

@Composable
private fun StatusChip(m: MachineState) {
    val (text, color) = statusOf(m)
    Text(
        text, color = Color.Black, fontSize = 17.sp, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center,
        modifier = Modifier.background(color, RoundedCornerShape(20.dp)).padding(horizontal = 16.dp, vertical = 8.dp),
    )
}

private fun statusOf(m: MachineState): Pair<String, Color> {
    val amber = Color(0xFFFFC107)
    val green = Color(0xFF00E676)
    val blue = Color(0xFF64B5F6)
    val grey = Color(0xFFE0E0E0)
    val red = Color(0xFFFF8A80)
    return when (m.phase) {
        Phase.SEARCHING -> when (m.hint) {
            Hint.SAME_PAGE -> "هذه الصفحة السابقة — ضع صفحة جديدة" to amber
            Hint.MOVE_BACK -> "الصفحة خارج الإطار — ابتعد قليلاً" to amber
            else -> "البحث عن ورقة…" to grey
        }
        Phase.COUNTDOWN -> "ورقة مكتشفة — الالتقاط بعد ${(m.countdownLeftMs + 999) / 1000}" to amber
        Phase.VERIFYING -> when (m.hint) {
            Hint.HOLD_STILL -> "التحقق من الوضوح… ثبّت الهاتف" to amber
            Hint.NOT_SHARP -> "التحقق من الوضوح… الصورة غير واضحة" to amber
            Hint.CAPTURE_BLURRY -> "الصورة الملتقطة غير واضحة — إعادة المحاولة" to red
            Hint.CAPTURE_FAILED -> "فشل الالتقاط — إعادة المحاولة" to red
            Hint.MOVE_BACK -> "الصفحة خارج الإطار — ابتعد قليلاً" to amber
            else -> "ورقة مكتشفة — التحقق من الوضوح" to amber
        }
        Phase.READY -> "جاهز" to green
        Phase.CAPTURING -> "جارٍ الالتقاط والحفظ…" to blue
        Phase.WAIT_REMOVAL -> "تم الالتقاط ✓ — أزل الورقة" to green
        Phase.PAUSED -> when (m.hint) {
            Hint.STORAGE_FULL -> "متوقف: مساحة التخزين ممتلئة" to red
            Hint.CAMERA_ERROR -> "متوقف: خطأ في الكاميرا" to red
            Hint.CAPTURE_FAILED -> "متوقف: فشل الالتقاط عدة مرات" to red
            else -> "متوقف مؤقتاً" to grey
        }
        Phase.COMPLETED -> "اكتمل العدد — جارٍ إنشاء PDF" to blue
        Phase.CANCELLED -> "أُلغي" to grey
    }
}

@Composable
private fun BottomPanel(vm: ScanViewModel, m: MachineState, onCancel: () -> Unit, modifier: Modifier) {
    Column(
        modifier.fillMaxWidth().background(Color(0xCC000000)).navigationBarsPadding().padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(
                    String.format(Locale.ROOT, "%d / %d", m.captured, m.target),
                    color = Color.White, fontSize = 30.sp, fontWeight = FontWeight.Bold,
                )
                Text("المنجزة: ${m.captured}   •   المتبقية: ${m.remaining}", color = Color.LightGray, fontSize = 15.sp)
            }
            vm.lastThumb?.let { bmp ->
                Image(
                    bitmap = bmp.asImageBitmap(), contentDescription = "آخر صفحة",
                    modifier = Modifier.height(78.dp).background(Color.White),
                )
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
            if (m.phase == Phase.PAUSED) {
                Button(
                    onClick = { vm.resume() }, modifier = Modifier.weight(1f).height(52.dp),
                    enabled = m.pausedFrom != Phase.CAPTURING,
                ) { Text("استئناف", fontSize = 17.sp) }
            } else {
                Button(
                    onClick = { vm.pause() }, modifier = Modifier.weight(1f).height(52.dp),
                    enabled = m.phase != Phase.COMPLETED,
                ) { Text("إيقاف مؤقت", fontSize = 17.sp) }
            }
            OutlinedButton(
                onClick = { vm.manualCapture() }, modifier = Modifier.weight(1f).height(52.dp),
                enabled = m.phase in setOf(Phase.SEARCHING, Phase.COUNTDOWN, Phase.VERIFYING, Phase.READY, Phase.WAIT_REMOVAL),
            ) { Text("التقاط يدوي") }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
            OutlinedButton(onClick = { vm.toggleTorch() }, modifier = Modifier.weight(1f)) {
                Text(if (vm.torchOn) "إطفاء الفلاش" else "تشغيل الفلاش")
            }
            Button(
                onClick = onCancel, modifier = Modifier.weight(1f),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFB71C1C)),
            ) { Text("إلغاء") }
        }
    }
}

// ---------------------------------------------------------------- building / result

@Composable
private fun BuildingScreen(vm: ScanViewModel) {
    Column(
        Modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        val err = vm.errorMessage
        if (err == null) {
            CircularProgressIndicator()
            Spacer(Modifier.height(18.dp))
            Text(vm.buildMessage, color = Color.White, fontSize = 18.sp)
        } else {
            Text(err, color = Color(0xFFFF8A80), textAlign = TextAlign.Center, fontSize = 17.sp)
            Spacer(Modifier.height(18.dp))
            Button(onClick = { vm.retryBuild() }) { Text("إعادة المحاولة") }
            TextButton(onClick = { vm.backToSetup() }) { Text("رجوع (الصور تبقى محفوظة)") }
        }
    }
}

@Composable
private fun ResultScreen(vm: ScanViewModel) {
    val context = LocalContext.current
    val r = vm.result ?: return
    BackHandler { vm.backToSetup() }
    Column(
        Modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp, Alignment.CenterVertically),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("✓ تم إنشاء ملف PDF", color = Color(0xFF00E676), fontSize = 26.sp, fontWeight = FontWeight.Bold)
        Text(r.name, color = Color.White, fontSize = 16.sp)
        Text(
            "${r.pages} صفحة • " + String.format(Locale.ROOT, "%.1f MB", r.sizeBytes / 1048576.0),
            color = Color.LightGray,
        )
        Text("المكان: المستندات ← AutoScan (Documents/AutoScan)", color = Color.LightGray, textAlign = TextAlign.Center)
        Button(onClick = {
            val i = Intent(Intent.ACTION_VIEW).setDataAndType(r.uri, "application/pdf")
                .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            runCatching { context.startActivity(Intent.createChooser(i, "فتح PDF")) }
        }, modifier = Modifier.fillMaxWidth().height(52.dp)) { Text("فتح الملف") }
        OutlinedButton(onClick = {
            val i = Intent(Intent.ACTION_SEND).setType("application/pdf")
                .putExtra(Intent.EXTRA_STREAM, r.uri)
                .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            runCatching { context.startActivity(Intent.createChooser(i, "مشاركة PDF")) }
        }, modifier = Modifier.fillMaxWidth().height(52.dp)) { Text("مشاركة") }
        TextButton(onClick = { vm.backToSetup() }) { Text("مسح جديد", fontSize = 17.sp) }
    }
}
