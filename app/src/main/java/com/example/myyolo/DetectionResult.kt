package com.example.myyolo

data class DetectionResult(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val label: String,
    val score: Float
)
