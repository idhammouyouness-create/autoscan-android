package com.youness.autoscan.pdf

import android.content.ContentValues
import android.content.Context
import android.graphics.pdf.PdfRenderer
import android.net.Uri
import android.os.Environment
import android.os.ParcelFileDescriptor
import android.provider.MediaStore
import java.io.File
import java.io.IOException

object PdfExporter {

    const val FOLDER = "AutoScan"

    /** Opens the PDF with Android's own renderer and checks every page is present and renderable. */
    @Throws(IOException::class)
    fun verify(pdf: File, expectedPages: Int) {
        ParcelFileDescriptor.open(pdf, ParcelFileDescriptor.MODE_READ_ONLY).use { fd ->
            PdfRenderer(fd).use { renderer ->
                if (renderer.pageCount != expectedPages) {
                    throw IOException("PDF has ${renderer.pageCount} pages, expected $expectedPages")
                }
                for (i in 0 until renderer.pageCount) {
                    renderer.openPage(i).use { page ->
                        if (page.width <= 0 || page.height <= 0) throw IOException("page ${i + 1} is empty")
                    }
                }
            }
        }
    }

    /** Copies the verified PDF to Documents/AutoScan (visible in the Files app). */
    @Throws(IOException::class)
    fun exportToDocuments(context: Context, pdf: File, displayName: String): Uri {
        val resolver = context.contentResolver
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, displayName)
            put(MediaStore.MediaColumns.MIME_TYPE, "application/pdf")
            put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOCUMENTS + "/" + FOLDER)
            put(MediaStore.MediaColumns.IS_PENDING, 1)
        }
        val collection = MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        val uri = resolver.insert(collection, values) ?: throw IOException("cannot create file in Documents")
        try {
            val out = resolver.openOutputStream(uri) ?: throw IOException("cannot open output")
            out.use { o -> pdf.inputStream().use { it.copyTo(o) } }
            val done = ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }
            resolver.update(uri, done, null, null)
            return uri
        } catch (e: Exception) {
            resolver.delete(uri, null, null)
            throw if (e is IOException) e else IOException(e)
        }
    }
}
