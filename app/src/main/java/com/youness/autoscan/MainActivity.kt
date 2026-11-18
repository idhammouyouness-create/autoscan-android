package com.youness.autoscan

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.unit.LayoutDirection
import com.youness.autoscan.ui.AppRoot
import com.youness.autoscan.ui.AppTheme
import org.opencv.android.OpenCVLoader

class MainActivity : ComponentActivity() {

    private val vm: ScanViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val openCvReady = OpenCVLoader.initLocal()
        if (!openCvReady) Log.e("AutoScan", "OpenCV failed to load")
        setContent {
            AppTheme {
                CompositionLocalProvider(LocalLayoutDirection provides LayoutDirection.Rtl) {
                    AppRoot(vm, openCvReady)
                }
            }
        }
    }

    override fun onStop() {
        super.onStop()
        vm.onAppBackground()
    }
}
