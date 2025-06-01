package com.example.myyolo

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

import android.speech.tts.TextToSpeech
import java.util.Locale

import android.content.Context
import android.media.AudioManager

import android.media.AudioAttributes
import android.media.AudioFocusRequest

import android.widget.Switch





class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    companion object {
        private const val MODEL_INPUT   = 640
        private const val FLOAT_BYTES   = 4
        private const val PERM_CODE     = 10
        private const val TAG           = "YOLO_DEBUG"

        private const val OBJ_THRESH    = 0.4f
        private const val CONF_THRESH   = 0.4f
        private const val IOU_THRESH    = 0.5f
        private const val MAX_DET       = 20
        private const val SKIP_FRAMES   = 1
    }

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var objectName: TextView
    private lateinit var cameraExecutor: ExecutorService

    private var interpreter: Interpreter? = null
    private lateinit var labels: List<String>
    private var numBoxes = 0
    private var clsPlus5 = 0
    private lateinit var outBuffer: TensorBuffer
    private var tts: TextToSpeech? = null
    private var wasSpeaking = false
    private var lastSpokenLabel: String? = null

    private var ttsReady = false

    // quyền camera thiết bị
    private val cameraPermissions = arrayOf(Manifest.permission.CAMERA)
    // biến âm thanh
    private var soundEnabled = true

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        tts = TextToSpeech(this, this)

        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        objectName  = findViewById(R.id.objectNameTextView)
        cameraExecutor = Executors.newSingleThreadExecutor()


        if (cameraPermissions.all { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }) {
            startCameraAndModel()
        } else {
            ActivityCompat.requestPermissions(this, cameraPermissions, PERM_CODE)
        }
        val switchSound = findViewById<Switch>(R.id.switchSound)

        // 2. Khởi tạo state ban đầu
        soundEnabled = switchSound.isChecked

        // 3. Khi người dùng gạt on/off
        switchSound.setOnCheckedChangeListener { _, isChecked ->
            soundEnabled = isChecked
            Log.d(TAG, "Sound enabled = $soundEnabled")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        interpreter?.close()
        tts?.shutdown()
    }

    override fun onInit(status: Int) {
        Log.d(TAG, "TTS init status: $status")
        if (status == TextToSpeech.SUCCESS) {
            ttsReady = true
            // Thiết lập thành tiếng Việt
            tts?.language = Locale("vi", "VN")
            tts?.setSpeechRate(0.9f)

            // Test
            speak("Xin chào, đây là áp Dô Lô của nhóm năm")
        }
    }

    private fun speak(text: String) {
        if (!ttsReady) {
            Log.d(TAG, "TTS chưa sẵn sàng, bỏ qua speak()")
            return
        }
        // request audio focus ở Android 8+
        val am = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val attrs = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .build()
        val focusRequest = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT)
            .setAudioAttributes(attrs)
            .build()
        am.requestAudioFocus(focusRequest)

        val res = tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, text.hashCode().toString())
        Log.d(TAG, "speak() returned $res")

    }

    private fun startCameraAndModel() {
        loadModel()
        setupCamera()
    }

    private fun loadModel() {
        // Load tflite model
        val mappedFile = FileUtil.loadMappedFile(this, "yolov5_float16.tflite")
        interpreter = Interpreter(mappedFile)

        // Load labels
        labels = FileUtil.loadLabels(this, "lables_last_v.txt")
        clsPlus5 = labels.size + 5

        // Determine output shape dynamically
        val shape = interpreter!!.getOutputTensor(0).shape() // [1,C,boxes] or [1,boxes,C]
        val channels = if (shape[1] == clsPlus5) shape[1] else shape[2]
        numBoxes = if (shape[1] == clsPlus5) shape[2] else shape[1]
        outBuffer = TensorBuffer.createFixedSize(shape, DataType.FLOAT32)

        Log.i(TAG, "Model loaded → out=${shape.contentToString()}, classes=${labels.size}, boxes=$numBoxes")
        val inputT = interpreter!!.getInputTensor(0)
        Log.i(TAG, "Input tensor shape: ${inputT.shape().contentToString()}, dtype=${inputT.dataType()}")
    }

    private fun setupCamera() {
        ProcessCameraProvider.getInstance(this).also { future ->
            future.addListener({
                val cameraProvider = future.get()
                val preview = Preview.Builder()
                    .build()
                    .also { it.setSurfaceProvider(previewView.surfaceProvider) }

                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { it.setAnalyzer(cameraExecutor, YoloAnalyzer()) }

                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis
                )
            }, ContextCompat.getMainExecutor(this))
        }
    }

    inner class YoloAnalyzer : ImageAnalysis.Analyzer {
        private var frameCount = 0
        private var lastLog = 0L

        override fun analyze(imageProxy: ImageProxy) {
            if (frameCount++ % SKIP_FRAMES != 0) {
                imageProxy.close(); return
            }
            val rawBmp = imageProxy.toBitmap() ?: run { imageProxy.close(); return }

            val rotation = imageProxy.imageInfo.rotationDegrees

            val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
            val bmp = Bitmap.createBitmap(rawBmp, 0, 0, rawBmp.width, rawBmp.height, matrix, true)
            imageProxy.close()

            val (boxed, scale, padX, padY, iw, ih) = letterbox(bmp)

            // Prepare NHWC input buffer
            val inputBuffer = ByteBuffer.allocateDirect(MODEL_INPUT*MODEL_INPUT*3*FLOAT_BYTES)
                .order(ByteOrder.nativeOrder())
            val pix = IntArray(MODEL_INPUT*MODEL_INPUT)
            boxed.getPixels(pix,0,MODEL_INPUT,0,0,MODEL_INPUT,MODEL_INPUT)
            for (p in pix) {
                inputBuffer.putFloat(((p shr 16) and 0xFF)/255f)
                inputBuffer.putFloat(((p shr 8)  and 0xFF)/255f)
                inputBuffer.putFloat(( p        and 0xFF)/255f)
            }
            inputBuffer.rewind()

            // Inference
            interpreter?.run(inputBuffer, outBuffer.buffer.rewind())
            val raw = outBuffer.floatArray
            val isCHW = outBuffer.shape[1] == clsPlus5

            // Decode + NMS
            val detections = decode(raw, isCHW, scale, padX, padY, iw, ih)
            val final = nms(detections)

            if (System.currentTimeMillis()-lastLog>1000) {
                Log.d(TAG, "Decoded=${detections.size}, NMS=${final.size}")
                lastLog = System.currentTimeMillis()
            }

            runOnUiThread {
                objectName.text = if (final.isNotEmpty())
                    "${final[0].label} ${(final[0].score*100).toInt()}%" else "Detected: None"

                // Nếu soundEnabled và có object mới → gọi TTS
                if (soundEnabled && ttsReady && final.isNotEmpty()) {
                    val label = final[0].label
                    if (!wasSpeaking || lastSpokenLabel != label) {
                        speak(label)
                        lastSpokenLabel = label
                        wasSpeaking = true
                    }
                } else if (final.isEmpty()) {
                    wasSpeaking = false
                }

                overlayView.setResults(final, iw, ih)
            }
        }

        private fun sigmoid(x: Float) = 1f/(1f + exp(-x))

        private fun decode(
            raw: FloatArray,
            isCHW: Boolean,
            scale: Float,
            padX: Float,
            padY: Float,
            iw: Float,
            ih: Float
        ): List<DetectionResult> {
            val out = mutableListOf<DetectionResult>()
            for (i in 0 until numBoxes) {
                // objectness logit → prob
                val logitObj = if (isCHW) raw[4*numBoxes + i] else raw[i*clsPlus5 + 4]
                val objProb = sigmoid(logitObj)
                if (objProb < OBJ_THRESH) continue

                // find best class
                var bestIdx = 0; var bestProb = 0f
                for (c in labels.indices) {
                    val logitCls = if (isCHW) raw[(5+c)*numBoxes + i] else raw[i*clsPlus5 + 5 + c]
                    val prob = sigmoid(logitCls)
                    if (prob > bestProb) { bestProb = prob; bestIdx = c }
                }
                val conf = objProb * bestProb
                if (conf < CONF_THRESH) continue

                // bbox raw

                var cx = if (isCHW) raw[0*numBoxes + i] else raw[i*clsPlus5 + 0]
                var cy = if (isCHW) raw[1*numBoxes + i] else raw[i*clsPlus5 + 1]
                var w  = if (isCHW) raw[2*numBoxes + i] else raw[i*clsPlus5 + 2]
                var h  = if (isCHW) raw[3*numBoxes + i] else raw[i*clsPlus5 + 3]
                // normalized → pixel

                if (cx<=1f && cy<=1f && w<=1f && h<=1f) {
                    cx*=MODEL_INPUT; cy*=MODEL_INPUT; w*=MODEL_INPUT; h*=MODEL_INPUT
                }
                // undo letterbox
                val left   = (cx - w/2 - padX)/scale
                val top    = (cy - h/2 - padY)/scale
                val right  = (cx + w/2 - padX)/scale
                val bottom = (cy + h/2 - padY)/scale
                val l = max(0f,left); val t = max(0f,top)
                val r = min(iw-1,right);  val b = min(ih-1,bottom)
                if (r>l && b>t) out += DetectionResult(l,t,r,b, labels[bestIdx], conf)
            }
            return out
        }

        private fun nms(dets: List<DetectionResult>): List<DetectionResult> {
            val out = mutableListOf<DetectionResult>()
            dets.sortedByDescending { it.score }.forEach { d ->
                if (out.size >= MAX_DET) return@forEach
                if (out.none { iou(it,d) > IOU_THRESH }) out += d
            }
            return out
        }

        private fun iou(a: DetectionResult, b: DetectionResult): Float {
            val l = max(a.left, b.left)
            val t = max(a.top, b.top)
            val r = min(a.right, b.right)
            val btm = min(a.bottom, b.bottom)
            val inter = max(0f, r-l) * max(0f, btm-t)
            val union = a.area() + b.area() - inter
            return if (union<=0f) 0f else inter/union
        }

        private fun ImageProxy.toBitmap(): Bitmap? = try {
            val y=planes[0].buffer; val u=planes[1].buffer; val v=planes[2].buffer
            val nv21=ByteArray(y.remaining()+u.remaining()+v.remaining())
            y.get(nv21,0,y.remaining()); v.get(nv21,y.remaining(),v.remaining()); u.get(nv21,y.remaining()+v.remaining(),u.remaining())
            val yuv=YuvImage(nv21,ImageFormat.NV21,width,height,null)
            val out=ByteArrayOutputStream().apply{yuv.compressToJpeg(Rect(0,0,width,height),100,this)}
            BitmapFactory.decodeByteArray(out.toByteArray(),0,out.size())
        } catch(e:Exception){null}

        private fun letterbox(src: Bitmap): LetterboxResult {
            val iw=src.width.toFloat(); val ih=src.height.toFloat()
            val scale=MODEL_INPUT/max(iw,ih)
            val nw=(iw*scale).toInt(); val nh=(ih*scale).toInt()
            val padX=(MODEL_INPUT-nw)/2f; val padY=(MODEL_INPUT-nh)/2f
            val res=Bitmap.createScaledBitmap(src,nw,nh,true)
            val boxed=Bitmap.createBitmap(MODEL_INPUT,MODEL_INPUT,Bitmap.Config.ARGB_8888)
            Canvas(boxed).apply{drawColor(Color.rgb(114,114,114));drawBitmap(res,padX,padY,null)}
            return LetterboxResult(boxed,scale,padX,padY,iw,ih)
        }
    }

    private fun DetectionResult.area() = (right-left)*(bottom-top)
    private data class LetterboxResult(
        val bmp: Bitmap,
        val scale: Float,
        val padX: Float,
        val padY: Float,
        val origW: Float,
        val origH: Float
    )
}
