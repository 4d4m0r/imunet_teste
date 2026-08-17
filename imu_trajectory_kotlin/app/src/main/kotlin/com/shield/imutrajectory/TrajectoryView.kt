package com.shield.imutrajectory

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

/**
 * Blank canvas that draws the reconstructed trajectory as points come in.
 * Starts empty; call clearTrajectory() to reset and addPoint(x, y) to append.
 */
class TrajectoryView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val points = mutableListOf<Pair<Float, Float>>()

    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#E53935")
        strokeWidth = 4f
        style = Paint.Style.STROKE
    }
    private val originPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#1E88E5")
        style = Paint.Style.FILL
    }

    fun clearTrajectory() {
        points.clear()
        invalidate()
    }

    fun addPoint(x: Float, y: Float) {
        points.add(x to y)
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (points.isEmpty()) return

        // Auto-scale so the whole trajectory fits the view, with a margin.
        var minX = points[0].first
        var maxX = points[0].first
        var minY = points[0].second
        var maxY = points[0].second
        for ((px, py) in points) {
            if (px < minX) minX = px
            if (px > maxX) maxX = px
            if (py < minY) minY = py
            if (py > maxY) maxY = py
        }
        val spanX = max(maxX - minX, 0.5f)
        val spanY = max(maxY - minY, 0.5f)
        val margin = 40f
        val scale = minOf(
            (width - 2 * margin) / spanX,
            (height - 2 * margin) / spanY
        )
        val cx = (minX + maxX) / 2f
        val cy = (minY + maxY) / 2f

        fun toScreen(px: Float, py: Float): Pair<Float, Float> {
            val sx = width / 2f + (px - cx) * scale
            // Screen Y grows downward; flip so "forward" (+y) draws upward.
            val sy = height / 2f - (py - cy) * scale
            return sx to sy
        }

        val (ox, oy) = toScreen(points[0].first, points[0].second)
        canvas.drawCircle(ox, oy, 8f, originPaint)

        var prev = toScreen(points[0].first, points[0].second)
        for (i in 1 until points.size) {
            val cur = toScreen(points[i].first, points[i].second)
            canvas.drawLine(prev.first, prev.second, cur.first, cur.second, linePaint)
            prev = cur
        }
    }
}
