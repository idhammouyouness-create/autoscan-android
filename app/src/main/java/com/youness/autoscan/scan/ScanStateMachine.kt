package com.youness.autoscan.scan

/**
 * Pure logic of the AUTO SEQUENTIAL SCAN mode. It receives one observation per analyzed camera
 * frame plus capture results, and returns side effects (vibrate, capture...) to perform.
 * No Android dependencies: fully unit tested.
 */
enum class Phase {
    /** Looking for a sheet. */
    SEARCHING,
    /** New sheet found (2nd page onward): waiting the user-selected delay. */
    COUNTDOWN,
    /** Checking sharpness and stability. */
    VERIFYING,
    /** Ready signal given (1 vibration); capture fires once the phone settles. */
    READY,
    /** Picture taken, processing and saving in progress. */
    CAPTURING,
    /** Page saved (2 vibrations); waiting for the previous sheet to leave the frame. */
    WAIT_REMOVAL,
    PAUSED,
    COMPLETED,
    CANCELLED,
}

enum class Hint {
    NONE,
    HOLD_STILL,
    NOT_SHARP,
    SAME_PAGE,
    MOVE_BACK,
    REMOVE_PAGE,
    CAPTURE_BLURRY,
    CAPTURE_FAILED,
    STORAGE_FULL,
    CAMERA_ERROR,
}

enum class CaptureOutcome { SAVED, BLURRY, FAILED, STORAGE_FULL }

sealed interface Effect {
    data object VibrateOnce : Effect
    data object VibrateTwice : Effect
    data object Capture : Effect
    data object Completed : Effect
}

data class Observation(
    val timeMs: Long,
    val hasDoc: Boolean,
    val docStable: Boolean,
    val sharp: Boolean,
    val motionLow: Boolean,
    /** Similarity (-1..1) between the visible sheet and the last saved page; null if unknown. */
    val similarityToLast: Double?,
    /** The page runs off the frame: ask the user to move back. */
    val tooClose: Boolean = false,
)

data class MachineState(
    val phase: Phase,
    val captured: Int,
    val target: Int,
    val countdownLeftMs: Long = 0,
    val hint: Hint = Hint.NONE,
    val pausedFrom: Phase? = null,
) {
    val remaining: Int get() = (target - captured).coerceAtLeast(0)
}

data class MachineConfig(
    /** Document must be missing this long before we consider it lost. */
    val docLostMs: Long = 700,
    /** Previous sheet must be absent this long to count as removed. */
    val removalAbsentMs: Long = 600,
    /** Wait after the "ready" vibration so the vibration itself does not blur the photo. */
    val readySettleMs: Long = 300,
    /** Give up the READY state if the phone does not settle in this time. */
    val readyTimeoutMs: Long = 1500,
    /** Similarity at/above which a sheet is considered the exact same page. */
    val samePageStrict: Double = 0.97,
    /** Below this similarity a sheet is surely a different page (used for "new page on top"). */
    val differentPageMax: Double = 0.90,
    val maxConsecutiveFailures: Int = 3,
)

class ScanStateMachine(
    target: Int,
    private val countdownMs: Long,
    initialCaptured: Int = 0,
    private val cfg: MachineConfig = MachineConfig(),
) {
    var state: MachineState = MachineState(
        phase = if (initialCaptured > 0) Phase.WAIT_REMOVAL else Phase.SEARCHING,
        captured = initialCaptured,
        target = target,
    )
        private set

    private var lastDocSeenMs = Long.MIN_VALUE / 2
    private var countdownDeadline = 0L
    private var readyAtMs = 0L
    private var sawDisturbance = false
    private var failures = 0

    val isActive: Boolean
        get() = state.phase in ACTIVE_PHASES

    fun onFrame(o: Observation): List<Effect> {
        if (o.hasDoc) lastDocSeenMs = o.timeMs
        val docLost = !o.hasDoc && o.timeMs - lastDocSeenMs >= cfg.docLostMs
        val s = state

        when (s.phase) {
            Phase.SEARCHING -> {
                if (o.hasDoc) {
                    val sim = o.similarityToLast
                    if (s.captured > 0 && sim != null && sim >= cfg.samePageStrict) {
                        set(s.copy(hint = Hint.SAME_PAGE))
                    } else if (s.captured == 0) {
                        set(s.copy(phase = Phase.VERIFYING, hint = Hint.NONE))
                    } else {
                        startCountdown(o.timeMs)
                    }
                } else {
                    set(s.copy(hint = if (o.tooClose) Hint.MOVE_BACK else Hint.NONE))
                }
            }

            Phase.COUNTDOWN -> {
                if (docLost) {
                    set(s.copy(phase = Phase.SEARCHING, hint = Hint.NONE, countdownLeftMs = 0))
                } else {
                    val left = countdownDeadline - o.timeMs
                    if (left <= 0) {
                        set(s.copy(phase = Phase.VERIFYING, countdownLeftMs = 0, hint = Hint.NONE))
                    } else {
                        set(s.copy(countdownLeftMs = left))
                    }
                }
            }

            Phase.VERIFYING -> {
                if (docLost) {
                    set(s.copy(phase = Phase.SEARCHING, hint = Hint.NONE))
                } else if (o.hasDoc && o.docStable && o.sharp && o.motionLow) {
                    readyAtMs = o.timeMs
                    set(s.copy(phase = Phase.READY, hint = Hint.NONE))
                    return listOf(Effect.VibrateOnce)
                } else {
                    val keepCaptureHint = s.hint == Hint.CAPTURE_BLURRY || s.hint == Hint.CAPTURE_FAILED
                    val hint = when {
                        !o.hasDoc && o.tooClose -> Hint.MOVE_BACK
                        !o.hasDoc -> s.hint
                        !o.motionLow || !o.docStable -> Hint.HOLD_STILL
                        !o.sharp -> Hint.NOT_SHARP
                        keepCaptureHint -> s.hint
                        else -> Hint.NONE
                    }
                    set(s.copy(hint = hint))
                }
            }

            Phase.READY -> {
                val elapsed = o.timeMs - readyAtMs
                if (docLost) {
                    set(s.copy(phase = Phase.SEARCHING, hint = Hint.NONE))
                } else if (elapsed >= cfg.readySettleMs && o.hasDoc && o.docStable && o.sharp && o.motionLow) {
                    set(s.copy(phase = Phase.CAPTURING, hint = Hint.NONE))
                    return listOf(Effect.Capture)
                } else if (elapsed > cfg.readyTimeoutMs) {
                    set(s.copy(phase = Phase.VERIFYING, hint = Hint.HOLD_STILL))
                }
            }

            Phase.WAIT_REMOVAL -> {
                if (!o.motionLow || !o.hasDoc) sawDisturbance = true
                val sim = o.similarityToLast
                if (!o.hasDoc && o.timeMs - lastDocSeenMs >= cfg.removalAbsentMs) {
                    // Previous sheet is gone.
                    set(s.copy(phase = Phase.SEARCHING, hint = Hint.NONE))
                } else if (o.hasDoc && sawDisturbance && o.docStable && sim != null && sim < cfg.differentPageMax) {
                    // A clearly different sheet was placed on top of the previous one.
                    startCountdown(o.timeMs)
                } else {
                    set(s.copy(hint = Hint.REMOVE_PAGE))
                }
            }

            Phase.CAPTURING, Phase.PAUSED, Phase.COMPLETED, Phase.CANCELLED -> Unit
        }
        return emptyList()
    }

    fun onCaptureResult(outcome: CaptureOutcome, timeMs: Long): List<Effect> {
        val s = state
        val paused = s.phase == Phase.PAUSED && s.pausedFrom == Phase.CAPTURING
        if (s.phase != Phase.CAPTURING && !paused) return emptyList()

        return when (outcome) {
            CaptureOutcome.SAVED -> {
                failures = 0
                val captured = s.captured + 1
                lastDocSeenMs = timeMs
                sawDisturbance = false
                if (captured >= s.target) {
                    set(s.copy(phase = Phase.COMPLETED, captured = captured, hint = Hint.NONE, pausedFrom = null))
                    listOf(Effect.VibrateTwice, Effect.Completed)
                } else {
                    if (paused) {
                        set(s.copy(captured = captured, pausedFrom = Phase.WAIT_REMOVAL))
                    } else {
                        set(s.copy(phase = Phase.WAIT_REMOVAL, captured = captured, hint = Hint.REMOVE_PAGE))
                    }
                    listOf(Effect.VibrateTwice)
                }
            }

            CaptureOutcome.BLURRY -> {
                if (paused) set(s.copy(pausedFrom = Phase.VERIFYING))
                else set(s.copy(phase = Phase.VERIFYING, hint = Hint.CAPTURE_BLURRY))
                emptyList()
            }

            CaptureOutcome.FAILED -> {
                failures++
                if (failures >= cfg.maxConsecutiveFailures) {
                    failures = 0
                    set(s.copy(phase = Phase.PAUSED, pausedFrom = Phase.VERIFYING, hint = Hint.CAPTURE_FAILED))
                } else if (paused) {
                    set(s.copy(pausedFrom = Phase.VERIFYING))
                } else {
                    set(s.copy(phase = Phase.VERIFYING, hint = Hint.CAPTURE_FAILED))
                }
                emptyList()
            }

            CaptureOutcome.STORAGE_FULL -> {
                set(s.copy(phase = Phase.PAUSED, pausedFrom = Phase.VERIFYING, hint = Hint.STORAGE_FULL))
                emptyList()
            }
        }
    }

    fun pause(hint: Hint = Hint.NONE) {
        val s = state
        if (s.phase in ACTIVE_PHASES || s.phase == Phase.CAPTURING) {
            set(s.copy(phase = Phase.PAUSED, pausedFrom = s.phase, hint = hint, countdownLeftMs = 0))
        } else if (s.phase == Phase.PAUSED && hint != Hint.NONE) {
            set(s.copy(hint = hint))
        }
    }

    fun resume(timeMs: Long) {
        val s = state
        if (s.phase != Phase.PAUSED) return
        if (s.pausedFrom == Phase.CAPTURING) return // result still pending; resumes when it arrives
        lastDocSeenMs = timeMs
        sawDisturbance = false
        val next = when {
            s.captured >= s.target -> Phase.COMPLETED
            s.pausedFrom == Phase.WAIT_REMOVAL -> Phase.WAIT_REMOVAL
            else -> Phase.SEARCHING
        }
        set(s.copy(phase = next, pausedFrom = null, hint = Hint.NONE))
    }

    fun cancel() {
        set(state.copy(phase = Phase.CANCELLED, pausedFrom = null, countdownLeftMs = 0))
    }

    /** User pressed the manual capture button. */
    fun manualCapture(): List<Effect> {
        val s = state
        if (s.phase !in ACTIVE_PHASES) return emptyList()
        set(s.copy(phase = Phase.CAPTURING, hint = Hint.NONE, countdownLeftMs = 0))
        return listOf(Effect.Capture)
    }

    private fun startCountdown(now: Long) {
        countdownDeadline = now + countdownMs
        set(state.copy(phase = Phase.COUNTDOWN, countdownLeftMs = countdownMs, hint = Hint.NONE))
    }

    private fun set(s: MachineState) {
        state = s
    }

    companion object {
        val ACTIVE_PHASES = setOf(
            Phase.SEARCHING, Phase.COUNTDOWN, Phase.VERIFYING, Phase.READY, Phase.WAIT_REMOVAL,
        )
    }
}
