package com.youness.autoscan

import com.youness.autoscan.pdf.JpegInfo
import com.youness.autoscan.pdf.PdfWriter
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.util.Locale

class PdfWriterTest {

    /** Copies a test JPEG fixture (src/test/resources/jpeg) to [dir]. */
    private fun fixture(dir: File, resource: String, name: String): File {
        val bytes = javaClass.classLoader!!.getResourceAsStream("jpeg/$resource")!!.use { it.readBytes() }
        return File(dir, name).apply { writeBytes(bytes) }
    }

    @Test
    fun jpegHeaderParsing() {
        val dir = createTempDir()
        val info = JpegInfo.read(fixture(dir, "small.jpg", "a.jpg").readBytes())
        assertNotNull(info)
        assertEquals(321, info!!.width)
        assertEquals(456, info.height)
        assertEquals(3, info.components)
        val gray = JpegInfo.read(fixture(dir, "page3.jpg", "g.jpg").readBytes())
        assertEquals(1, gray!!.components)
        assertNull(JpegInfo.read(byteArrayOf(1, 2, 3, 4, 5)))
    }

    @Test
    fun writesValidPdfWithCorrectXref() {
        val dir = createTempDir()
        val pages = listOf(
            fixture(dir, "page1.jpg", "page_0001.jpg"),
            fixture(dir, "page2.jpg", "page_0002.jpg"),
            fixture(dir, "page3.jpg", "page_0003.jpg"),
        )
        // Write the sample where CI uploads it, so it can be checked with an external PDF reader.
        val outDir = File("build/test-pdf").apply { mkdirs() }
        val out = File(outDir, "sample.pdf")
        PdfWriter.write(pages, out)

        val bytes = out.readBytes()
        val text = String(bytes, Charsets.ISO_8859_1)
        assertTrue(text.startsWith("%PDF-1.4"))
        assertTrue(text.trimEnd().endsWith("%%EOF"))
        assertTrue(text.contains("/Count 3"))
        assertEquals(3, Regex("/Type /Page ").findAll(text).count())

        // Every xref entry must point exactly at "N 0 obj".
        val startxref = Regex("startxref\\n(\\d+)").find(text)!!.groupValues[1].toInt()
        assertTrue(text.startsWith("xref", startxref))
        val entries = Regex("(\\d{10}) 00000 n ").findAll(text.substring(startxref)).map { it.groupValues[1].toInt() }.toList()
        assertEquals(2 + 3 * 3, entries.size)
        entries.forEachIndexed { i, off ->
            val expected = "${i + 1} 0 obj"
            assertTrue("object ${i + 1} offset wrong", text.startsWith(expected, off))
        }
        // The JPEG bytes are embedded unchanged.
        val first = pages[0].readBytes()
        assertTrue(indexOf(bytes, first) > 0)
    }

    @Test
    fun numbersAreAsciiEvenWithArabicLocale() {
        val old = Locale.getDefault()
        try {
            Locale.setDefault(Locale.forLanguageTag("ar-MA-u-nu-arab"))
            val dir = createTempDir()
            val out = File(dir, "ar.pdf")
            PdfWriter.write(listOf(fixture(dir, "small.jpg", "page_0001.jpg")), out)
            val text = String(out.readBytes(), Charsets.ISO_8859_1)
            assertTrue(text.contains("/MediaBox [0 0 595.28"))
        } finally {
            Locale.setDefault(old)
        }
    }

    private fun createTempDir(): File =
        File.createTempFile("pdftest", "").apply { delete(); mkdirs(); deleteOnExit() }

    private fun indexOf(hay: ByteArray, needle: ByteArray): Int {
        outer@ for (i in 0..hay.size - needle.size) {
            for (j in needle.indices) if (hay[i + j] != needle[j]) continue@outer
            return i
        }
        return -1
    }
}
