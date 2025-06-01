package com.example.myyolo

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private val boxPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }

    private val labelBgPaint = Paint().apply {
        color = Color.BLACK
        alpha = 160
        style = Paint.Style.FILL
    }

    private val textPaint = Paint().apply {
        color = Color.YELLOW
        textSize = 40f
        isAntiAlias = true
    }

    private var results: List<DetectionResult> = emptyList()
    private var srcW: Float = 1f
    private var srcH: Float = 1f


    fun setResults(newResults: List<DetectionResult>, origW: Float, origH: Float) {
        results = newResults
        srcW = origW
        srcH = origH
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (results.isEmpty()) return

        // Tính scale và padding letterbox cho view
        val viewW = width.toFloat()
        val viewH = height.toFloat()
        val scale = min(viewW / srcW, viewH / srcH)
        val padX = (viewW - srcW * scale) / 2f
        val padY = (viewH - srcH * scale) / 2f

        for (r in results) {
            // Map tọa độ từ ảnh gốc sang view
            val left = r.left * scale + padX
            val top = r.top * scale + padY
            val right = r.right * scale + padX
            val bottom = r.bottom * scale + padY

            // Vẽ bounding box
            val rect = RectF(left, top, right, bottom)
            canvas.drawRect(rect, boxPaint)

            // Chuẩn bị text label
            val labelText = "%s %.2f".format(r.label, r.score)
            val textWidth = textPaint.measureText(labelText)
            val fm = textPaint.fontMetrics
            val textHeight = fm.bottom - fm.top

            // Vẽ nền cho text
            val rectLabel = RectF(
                left,
                top - textHeight - 8,
                left + textWidth + 10,
                top
            )
            canvas.drawRect(rectLabel, labelBgPaint)

            // Vẽ text
            canvas.drawText(
                labelText,
                rectLabel.left + 5,
                rectLabel.bottom - 4,
                textPaint
            )
        }
    }
}
