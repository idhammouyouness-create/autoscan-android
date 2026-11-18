package com.youness.autoscan.util

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager

class Haptics(context: Context) {

    private val vibrator: Vibrator? =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            context.getSystemService(VibratorManager::class.java)?.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            context.getSystemService(Vibrator::class.java)
        }

    /** One short pulse: sheet is ready, capturing. */
    fun once() {
        vibrator?.vibrate(VibrationEffect.createOneShot(70, VibrationEffect.DEFAULT_AMPLITUDE))
    }

    /** Two short pulses: page saved successfully. */
    fun twice() {
        vibrator?.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 60, 120, 60), -1))
    }
}
