package com.youness.autoscan

import com.youness.autoscan.pdf.JpegInfo
import com.youness.autoscan.pdf.PdfWriter
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import java.util.Locale
import javax.imageio.ImageIO

class PdfWriterTest {

    private fun makeJpeg(dir: File, name: String, w: Int, h: Int, label: String): File {
        val img = BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
        val g = img.createGraphics()
        g.color = Color.WHITE
        g.fillRect(0, 0, w, h)
        g.color = Color.BLACK
        g.font = g.font.deriveFont(48f)
        g.drawString(label, 40, 100)
        g.drawRect(5, 5, w - 10, h - 10)
        g.dispose()
        val f = File(dir, name)
        ImageIO.write(img, "jpg", f)
        return f
    }

    @Test
    fun jpegHeaderParsing() {
        val dir = createTempDir()
        val f = makeJpeg(dir, "a.jpg", 321, 456, "x")
        val info = JpegInfo.read(f.readBytes())
        assertNotNull(info)
        assertEquals(321, info!!.width)
        assertEquals(456, info.height)
        assertEquals(3, info.components)
        assertNull(JpegInfo.read(byteArrayOf(1, 2, 3, 4, 5)))
    }

    @Test
    fun writesValidPdfWithCorrectXref() {
        val dir = createTempDir()
        val pages = listOf(
            makeJpeg(dir, "page_0001.jpg", 1240, 1754, "Page 1"),
            makeJpeg(dir, "page_0002.jpg", 1754, 1240, "Page 2 landscape"),
            makeJpeg(dir, "page_0003.jpg", 1000, 1500, "Page 3"),
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
            PdfWriter.write(listOf(makeJpeg(dir, "page_0001.jpg", 600, 800, "x")), out)
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
