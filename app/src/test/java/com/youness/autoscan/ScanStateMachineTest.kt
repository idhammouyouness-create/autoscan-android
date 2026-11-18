package com.youness.autoscan

import com.youness.autoscan.scan.CaptureOutcome
import com.youness.autoscan.scan.Effect
import com.youness.autoscan.scan.Hint
import com.youness.autoscan.scan.Observation
import com.youness.autoscan.scan.Phase
import com.youness.autoscan.scan.ScanStateMachine
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ScanStateMachineTest {

    private var t = 0L

    private fun frame(
        sm: ScanStateMachine,
        doc: Boolean = true,
        stable: Boolean = true,
        sharp: Boolean = true,
        still: Boolean = true,
        sim: Double? = null,
        dt: Long = 60,
    ): List<Effect> {
        t += dt
        return sm.onFrame(Observation(t, doc, stable, sharp, still, sim))
    }

    /** Feeds frames until [phase] is reached; returns all effects produced. */
    private fun until(sm: ScanStateMachine, phase: Phase, max: Int = 500, f: () -> List<Effect>): List<Effect> {
        val all = ArrayList<Effect>()
        var n = 0
        while (sm.state.phase != phase) {
            all += f()
            n++
            check(n < max) { "never reached $phase, stuck in ${sm.state.phase}" }
        }
        return all
    }

    @Test
    fun firstPage_noCountdown_singleVibrationThenCapture() {
        val sm = ScanStateMachine(target = 3, countdownMs = 3000)
        assertEquals(Phase.SEARCHING, sm.state.phase)
        frame(sm)
        assertEquals(Phase.VERIFYING, sm.state.phase)
        val e1 = frame(sm)
        assertEquals(listOf(Effect.VibrateOnce), e1)
        assertEquals(Phase.READY, sm.state.phase)
        // capture waits for the settle time after the vibration
        val e2 = frame(sm, dt = 100)
        assertTrue(e2.isEmpty())
        val e3 = until(sm, Phase.CAPTURING) { frame(sm, dt = 100) }
        assertEquals(listOf(Effect.Capture), e3)
    }

    @Test
    fun blurryOrMoving_neverReady() {
        val sm = ScanStateMachine(target = 2, countdownMs = 3000)
        frame(sm)
        repeat(50) {
            assertTrue(frame(sm, sharp = false).isEmpty())
        }
        assertEquals(Phase.VERIFYING, sm.state.phase)
        assertEquals(Hint.NOT_SHARP, sm.state.hint)
        repeat(50) { assertTrue(frame(sm, still = false).isEmpty()) }
        assertEquals(Hint.HOLD_STILL, sm.state.hint)
    }

    private fun captureOnePage(sm: ScanStateMachine): List<Effect> {
        until(sm, Phase.CAPTURING) { frame(sm, dt = 100) }
        return sm.onCaptureResult(CaptureOutcome.SAVED, t)
    }

    @Test
    fun saved_twoVibrations_thenWaitsForRemoval_noDuplicate() {
        val sm = ScanStateMachine(target = 3, countdownMs = 3000)
        val e = captureOnePage(sm)
        assertEquals(listOf(Effect.VibrateTwice), e)
        assertEquals(1, sm.state.captured)
        assertEquals(Phase.WAIT_REMOVAL, sm.state.phase)
        // Same sheet stays in view for a long time: nothing happens.
        repeat(200) { assertTrue(frame(sm, sim = 0.99).isEmpty()) }
        assertEquals(Phase.WAIT_REMOVAL, sm.state.phase)
        assertEquals(1, sm.state.captured)
    }

    @Test
    fun removal_thenNewSheet_countdownThenCapture() {
        val sm = ScanStateMachine(target = 3, countdownMs = 3000)
        captureOnePage(sm)
        until(sm, Phase.SEARCHING) { frame(sm, doc = false, still = false, dt = 100) }
        frame(sm, sim = 0.5)
        assertEquals(Phase.COUNTDOWN, sm.state.phase)
        // Countdown lasts 3 s.
        var elapsed = 0L
        while (sm.state.phase == Phase.COUNTDOWN) {
            frame(sm, sim = 0.5, dt = 100); elapsed += 100
        }
        assertTrue("countdown too short: $elapsed", elapsed >= 2900)
        assertEquals(Phase.VERIFYING, sm.state.phase)
        until(sm, Phase.CAPTURING) { frame(sm, sim = 0.5, dt = 100) }
    }

    @Test
    fun fiveSecondCountdown() {
        val sm = ScanStateMachine(target = 3, countdownMs = 5000)
        captureOnePage(sm)
        until(sm, Phase.SEARCHING) { frame(sm, doc = false, dt = 100) }
        frame(sm, sim = 0.2)
        var elapsed = 0L
        while (sm.state.phase == Phase.COUNTDOWN) {
            frame(sm, sim = 0.2, dt = 100); elapsed += 100
        }
        assertTrue(elapsed in 4900..5100)
    }

    @Test
    fun sheetLostDuringCountdown_restartsSearch() {
        val sm = ScanStateMachine(target = 3, countdownMs = 3000)
        captureOnePage(sm)
        until(sm, Phase.SEARCHING) { frame(sm, doc = false, dt = 100) }
        frame(sm, sim = 0.3)
        assertEquals(Phase.COUNTDOWN, sm.state.phase)
        until(sm, Phase.SEARCHING) { frame(sm, doc = false, dt = 100) }
    }

    @Test
    fun samePagePutBack_isNotCapturedAgain() {
        val sm = ScanStateMachine(target = 3, countdownMs = 3000)
        captureOnePage(sm)
        until(sm, Phase.SEARCHING) { frame(sm, doc = false, dt = 100) }
        repeat(30) { frame(sm, sim = 0.99) }
        assertEquals(Phase.SEARCHING, sm.state.phase)
        assertEquals(Hint.SAME_PAGE, sm.state.hint)
    }

    @Test
    fun newSheetPlacedOnTop_detectedWithoutFullRemoval() {
        val sm = ScanStateMachine(target = 3, countdownMs = 3000)
        captureOnePage(sm)
        // Hand movement while placing the new sheet, then a different stable sheet.
        repeat(3) { frame(sm, still = false, stable = false, sim = 0.98) }
        frame(sm, sim = 0.4)
        assertEquals(Phase.COUNTDOWN, sm.state.phase)
    }

    @Test
    fun reachingTarget_completes() {
        val sm = ScanStateMachine(target = 2, countdownMs = 3000)
        captureOnePage(sm)
        until(sm, Phase.SEARCHING) { frame(sm, doc = false, dt = 100) }
        frame(sm, sim = 0.1)
        until(sm, Phase.CAPTURING) { frame(sm, sim = 0.1, dt = 100) }
        val e = sm.onCaptureResult(CaptureOutcome.SAVED, t)
        assertEquals(listOf(Effect.VibrateTwice, Effect.Completed), e)
        assertEquals(Phase.COMPLETED, sm.state.phase)
        assertEquals(2, sm.state.captured)
        // No more frames are processed.
        assertTrue(frame(sm).isEmpty())
        assertEquals(2, sm.state.captured)
    }

    @Test
    fun blurryCapture_notCounted_retries() {
        val sm = ScanStateMachine(target = 2, countdownMs = 3000)
        until(sm, Phase.CAPTURING) { frame(sm, dt = 100) }
        val e = sm.onCaptureResult(CaptureOutcome.BLURRY, t)
        assertTrue(e.isEmpty())
        assertEquals(0, sm.state.captured)
        assertEquals(Phase.VERIFYING, sm.state.phase)
        assertEquals(Hint.CAPTURE_BLURRY, sm.state.hint)
    }

    @Test
    fun repeatedFailures_pause() {
        val sm = ScanStateMachine(target = 2, countdownMs = 3000)
        repeat(3) {
            until(sm, Phase.CAPTURING) { frame(sm, dt = 100) }
            sm.onCaptureResult(CaptureOutcome.FAILED, t)
        }
        assertEquals(Phase.PAUSED, sm.state.phase)
        assertEquals(0, sm.state.captured)
    }

    @Test
    fun storageFull_pausesWithoutCounting() {
        val sm = ScanStateMachine(target = 2, countdownMs = 3000)
        until(sm, Phase.CAPTURING) { frame(sm, dt = 100) }
        sm.onCaptureResult(CaptureOutcome.STORAGE_FULL, t)
        assertEquals(Phase.PAUSED, sm.state.phase)
        assertEquals(Hint.STORAGE_FULL, sm.state.hint)
        assertEquals(0, sm.state.captured)
    }

    @Test
    fun pauseResume_ignoresFramesWhilePaused() {
        val sm = ScanStateMachine(target = 2, countdownMs = 3000)
        sm.pause()
        repeat(50) { assertTrue(frame(sm).isEmpty()) }
        assertEquals(Phase.PAUSED, sm.state.phase)
        sm.resume(t)
        assertEquals(Phase.SEARCHING, sm.state.phase)
    }

    @Test
    fun pauseDuringCapture_resultStillCounted() {
        val sm = ScanStateMachine(target = 3, countdownMs = 3000)
        until(sm, Phase.CAPTURING) { frame(sm, dt = 100) }
        sm.pause()
        sm.resume(t) // ignored while capture pending
        assertEquals(Phase.PAUSED, sm.state.phase)
        sm.onCaptureResult(CaptureOutcome.SAVED, t)
        assertEquals(1, sm.state.captured)
        assertEquals(Phase.PAUSED, sm.state.phase)
        sm.resume(t)
        assertEquals(Phase.WAIT_REMOVAL, sm.state.phase)
    }

    @Test
    fun resumedSession_startsByWaitingForRemoval() {
        val sm = ScanStateMachine(target = 5, countdownMs = 3000, initialCaptured = 2)
        assertEquals(Phase.WAIT_REMOVAL, sm.state.phase)
        assertEquals(3, sm.state.remaining)
    }

    @Test
    fun pageRunsOffFrame_asksToMoveBack_neverCaptures() {
        val sm = ScanStateMachine(target = 2, countdownMs = 3000)
        repeat(50) {
            t += 60
            assertTrue(sm.onFrame(Observation(t, false, false, false, true, null, tooClose = true)).isEmpty())
        }
        assertEquals(Phase.SEARCHING, sm.state.phase)
        assertEquals(Hint.MOVE_BACK, sm.state.hint)
    }

    @Test
    fun manualCapture_works() {
        val sm = ScanStateMachine(target = 2, countdownMs = 3000)
        assertEquals(listOf(Effect.Capture), sm.manualCapture())
        assertEquals(Phase.CAPTURING, sm.state.phase)
        assertTrue(sm.manualCapture().isEmpty()) // no double capture
    }
}
