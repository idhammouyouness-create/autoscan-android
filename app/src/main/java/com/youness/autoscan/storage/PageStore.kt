package com.youness.autoscan.storage

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.StatFs
import com.youness.autoscan.vision.EnhanceMode
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.Locale
import java.util.Properties

/** Settings a scan session was started with; saved next to its pages for crash recovery. */
data class SessionInfo(
    val target: Int,
    val countdownSec: Int,
    val mode: EnhanceMode,
    val createdAt: Long,
)

class StorageFullException(msg: String) : IOException(msg)

/**
 * Pages of a session live in app-private storage until the PDF has been created and verified.
 * Each page is written to a temporary file, synced to disk, verified, then renamed: a crash or a
 * full disk can never leave a half-written page counted as done.
 */
class Session(val dir: File) {

    val info: SessionInfo by lazy { readInfo() }

    fun pageFiles(): List<File> =
        (dir.listFiles { f -> f.isFile && PAGE_RE.matches(f.name) } ?: emptyArray())
            .sortedBy { it.name }

    @Throws(IOException::class)
    fun savePage(index: Int, bitmap: Bitmap, quality: Int = 90): File {
        val target = File(dir, String.format(Locale.ROOT, "page_%04d.jpg", index))
        val tmp = File(dir, target.name + ".tmp")
        try {
            FileOutputStream(tmp).use { out ->
                if (!bitmap.compress(Bitmap.CompressFormat.JPEG, quality, out)) {
                    throw IOException("JPEG encoding failed")
                }
                out.flush()
                out.fd.sync()
            }
            val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeFile(tmp.absolutePath, opts)
            if (opts.outWidth <= 0 || opts.outHeight <= 0) throw IOException("saved page is unreadable")
            if (target.exists()) target.delete()
            if (!tmp.renameTo(target)) throw IOException("rename failed")
            return target
        } catch (e: IOException) {
            tmp.delete()
            val msg = e.message ?: ""
            if (msg.contains("ENOSPC") || msg.contains("No space", ignoreCase = true)) {
                throw StorageFullException(msg)
            }
            throw e
        }
    }

    fun delete() {
        dir.deleteRecursively()
    }

    private fun readInfo(): SessionInfo {
        val p = Properties()
        File(dir, INFO_FILE).inputStream().use { p.load(it) }
        return SessionInfo(
            target = p.getProperty("target", "1").toInt(),
            countdownSec = p.getProperty("countdown", "3").toInt(),
            mode = runCatching { EnhanceMode.valueOf(p.getProperty("mode", "DOCUMENT")) }.getOrDefault(EnhanceMode.DOCUMENT),
            createdAt = p.getProperty("created", "0").toLong(),
        )
    }

    companion object {
        const val INFO_FILE = "session.properties"
        private val PAGE_RE = Regex("page_\\d{4}\\.jpg")
    }
}

class PageStore(context: Context) {

    private val root = File(context.filesDir, "sessions").apply { mkdirs() }
    private val statDir = context.filesDir

    fun create(info: SessionInfo): Session {
        val dir = File(root, "s_${info.createdAt}")
        dir.mkdirs()
        val p = Properties()
        p.setProperty("target", info.target.toString())
        p.setProperty("countdown", info.countdownSec.toString())
        p.setProperty("mode", info.mode.name)
        p.setProperty("created", info.createdAt.toString())
        File(dir, Session.INFO_FILE).outputStream().use { p.store(it, null) }
        return Session(dir)
    }

    /** Most recent session that was not turned into a PDF (app closed or crashed). */
    fun findUnfinished(): Session? =
        root.listFiles { f -> f.isDirectory && File(f, Session.INFO_FILE).exists() }
            ?.sortedByDescending { it.name }
            ?.map { Session(it) }
            ?.firstOrNull { it.pageFiles().isNotEmpty() }

    /** Removes empty or broken sessions (and leftover temp files). */
    fun cleanup(keep: Session?) {
        root.listFiles()?.forEach { dir ->
            if (keep != null && dir == keep.dir) return@forEach
            val s = Session(dir)
            if (!File(dir, Session.INFO_FILE).exists() || s.pageFiles().isEmpty()) dir.deleteRecursively()
        }
    }

    fun freeBytes(): Long = StatFs(statDir.absolutePath).availableBytes

    /** Enough room for one more page plus the final PDF headroom. */
    fun hasRoomForPage(): Boolean = freeBytes() > 80L * 1024 * 1024
}
