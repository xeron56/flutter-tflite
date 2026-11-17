package dev.local.indra_ocr

import org.junit.Assert.assertEquals
import org.junit.Test

class TextLogicTest {
    @Test fun rtlKeepsDigitsInTheirOwnOrder() {
        assertEquals("123 םולש", RtlText.flip("שלום 123"))
        assertEquals("Invoice םולש 123", RtlText.flip("Invoice שלום 123"))
    }
    @Test fun rtlRetainsCombiningMarksAcrossRoundTrip() {
        val text = "سَلَام 123 (test)"
        assertEquals(text, RtlText.flip(RtlText.flip(text)))
        assertEquals("simple Latin 123", RtlText.flip("simple Latin 123"))
    }
    @Test fun readingOrderKeepsColumnsTogether() {
        val a = TextLine(floatArrayOf(0f, 0f, 100f, 10f), "a")
        val b = TextLine(floatArrayOf(0f, 20f, 100f, 30f), "b")
        val c = TextLine(floatArrayOf(200f, 0f, 300f, 10f), "c")
        val d = TextLine(floatArrayOf(200f, 20f, 300f, 30f), "d")
        assertEquals(listOf("a", "b", "c", "d"), ReadingOrder.sort(listOf(d, b, c, a)).map { it.text })
    }
}
