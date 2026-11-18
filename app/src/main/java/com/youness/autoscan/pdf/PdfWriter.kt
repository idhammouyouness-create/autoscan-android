package com.youness.autoscan.pdf

import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStream
import java.util.Locale

/** Dimensions and color components read from a JPEG header. */
data class JpegInfo(val width: Int, val height: Int, val components: Int) {
    companion object {
        /** Parses the SOF marker. Returns null if the data is not a valid JPEG. */
        fun read(bytes: ByteArray): JpegInfo? {
            if (bytes.size < 4 || (bytes[0].toInt() and 0xFF) != 0xFF || (bytes[1].toInt() and 0xFF) != 0xD8) return null
            var i = 2
            while (i + 3 < bytes.size) {
                if ((bytes[i].toInt() and 0xFF) != 0xFF) { i++; continue }
                var marker = bytes[i + 1].toInt() and 0xFF
                // skip fill bytes
                while (marker == 0xFF && i + 2 < bytes.size) { i++; marker = bytes[i + 1].toInt() and 0xFF }
                if (marker == 0xD8 || marker == 0x01 || marker in 0xD0..0xD7) { i += 2; continue }
                if (marker == 0xD9 || marker == 0xDA) return null
                if (i + 3 >= bytes.size) return null
                val len = ((bytes[i + 2].toInt() and 0xFF) shl 8) or (bytes[i + 3].toInt() and 0xFF)
                val isSof = marker in 0xC0..0xCF && marker != 0xC4 && marker != 0xC8 && marker != 0xCC
                if (isSof) {
                    if (i + 9 >= bytes.size) return null
                    val h = ((bytes[i + 5].toInt() and 0xFF) shl 8) or (bytes[i + 6].toInt() and 0xFF)
                    val w = ((bytes[i + 7].toInt() and 0xFF) shl 8) or (bytes[i + 8].toInt() and 0xFF)
                    val c = bytes[i + 9].toInt() and 0xFF
                    if (w <= 0 || h <= 0 || c !in setOf(1, 3, 4)) return null
                    return JpegInfo(w, h, c)
                }
                i += 2 + len
            }
            return null
        }
    }
}

/**
 * Minimal, standards-compliant PDF writer: one page per JPEG, embedding the JPEG data as-is
 * (DCTDecode). No re-compression = no quality loss, small files, very low memory use.
 * Pure Kotlin so it can be unit tested on the JVM.
 */
object PdfWriter {

    private const val A4_SHORT_PT = 595.28
    private const val A4_LONG_PT = 841.89

    @Throws(IOException::class)
    fun write(pages: List<File>, out: File) {
        require(pages.isNotEmpty()) { "no pages" }
        FileOutputStream(out).use { fos ->
            val counting = CountingStream(BufferedOutputStream(fos, 1 shl 16))
            writeTo(pages, counting)
            counting.flush()
            fos.fd.sync()
        }
    }

    private fun writeTo(pages: List<File>, os: CountingStream) {
        val n = pages.size
        val totalObjects = 2 + 3 * n
        val offsets = LongArray(totalObjects + 1)

        fun ascii(s: String) = os.write(s.toByteArray(Charsets.US_ASCII))
        fun num(d: Double) = String.format(Locale.ROOT, "%.2f", d)

        ascii("%PDF-1.4\n")
        os.write(byteArrayOf('%'.code.toByte(), 0xE2.toByte(), 0xE3.toByte(), 0xCF.toByte(), 0xD3.toByte(), '\n'.code.toByte()))

        offsets[1] = os.count
        ascii("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")

        offsets[2] = os.count
        val kids = (0 until n).joinToString(" ") { "${3 + 3 * it} 0 R" }
        ascii("2 0 obj\n<< /Type /Pages /Kids [$kids] /Count $n >>\nendobj\n")

        for ((index, file) in pages.withIndex()) {
            val pageObj = 3 + 3 * index
            val contentObj = pageObj + 1
            val imageObj = pageObj + 2
            val jpeg = file.readBytes()
            val info = JpegInfo.read(jpeg) ?: throw IOException("invalid JPEG: ${file.name}")

            // Page width = A4 width (portrait) or A4 height (landscape); height follows the image.
            val pw: Double
            val ph: Double
            if (info.height >= info.width) {
                pw = A4_SHORT_PT
                ph = A4_SHORT_PT * info.height / info.width
            } else {
                pw = A4_LONG_PT
                ph = A4_LONG_PT * info.height / info.width
            }

            offsets[pageObj] = os.count
            ascii(
                "$pageObj 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${num(pw)} ${num(ph)}] " +
                    "/Resources << /XObject << /Im0 $imageObj 0 R >> /ProcSet [/PDF /ImageB /ImageC] >> " +
                    "/Contents $contentObj 0 R >>\nendobj\n",
            )

            val content = "q\n${num(pw)} 0 0 ${num(ph)} 0 0 cm\n/Im0 Do\nQ\n".toByteArray(Charsets.US_ASCII)
            offsets[contentObj] = os.count
            ascii("$contentObj 0 obj\n<< /Length ${content.size} >>\nstream\n")
            os.write(content)
            ascii("\nendstream\nendobj\n")

            val colorSpace = when (info.components) {
                1 -> "/DeviceGray"
                4 -> "/DeviceCMYK /Decode [1 0 1 0 1 0 1 0]"
                else -> "/DeviceRGB"
            }
            offsets[imageObj] = os.count
            ascii(
                "$imageObj 0 obj\n<< /Type /XObject /Subtype /Image /Width ${info.width} /Height ${info.height} " +
                    "/ColorSpace $colorSpace /BitsPerComponent 8 /Filter /DCTDecode /Length ${jpeg.size} >>\nstream\n",
            )
            os.write(jpeg)
            ascii("\nendstream\nendobj\n")
        }

        val xrefOffset = os.count
        ascii("xref\n0 ${totalObjects + 1}\n")
        ascii("0000000000 65535 f \n")
        for (i in 1..totalObjects) {
            ascii(String.format(Locale.ROOT, "%010d 00000 n \n", offsets[i]))
        }
        ascii("trailer\n<< /Size ${totalObjects + 1} /Root 1 0 R >>\nstartxref\n$xrefOffset\n%%EOF\n")
    }

    private class CountingStream(private val inner: OutputStream) : OutputStream() {
        var count = 0L
            private set

        override fun write(b: Int) {
            inner.write(b); count++
        }

        override fun write(b: ByteArray, off: Int, len: Int) {
            inner.write(b, off, len); count += len
        }

        override fun flush() = inner.flush()
        override fun close() = inner.close()
    }
}
