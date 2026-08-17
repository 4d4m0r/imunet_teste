package com.shield.imutrajectory

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Real-time IMU dead-reckoning: reads accelerometer + gyroscope + rotation vector, rotates
 * gyro/acce into the device's current orientation frame (ori = rv(t), no init_rotor -- there is
 * no ARCore groundtruth at inference time, matching how the model was trained/evaluated), feeds
 * a sliding 200-sample window into the on-device model, integrates the predicted planar velocity
 * into a trajectory, and draws it live.
 */
class MainActivity : AppCompatActivity(), SensorEventListener {

    companion object {
        private const val MODEL_ASSET = "model.pte"
        private const val WINDOW_SIZE = 200
        private const val CHANNELS = 6
    }

    private lateinit var sensorManager: SensorManager
    private var accSensor: Sensor? = null
    private var gyroSensor: Sensor? = null
    private var rotSensor: Sensor? = null

    private var module: Module? = null

    private lateinit var trajectoryView: TrajectoryView

    private val worker = Executors.newSingleThreadExecutor()
    private val ui = Handler(Looper.getMainLooper())
    private val inferenceInFlight = AtomicBoolean(false)

    private var running = false

    private val window = Array(CHANNELS) { FloatArray(WINDOW_SIZE) }
    private var sampleCount = 0

    private var latestAcc = FloatArray(3)
    private var latestGyro = FloatArray(3)
    private var latestRotation: Quaternion? = null
    private var hasAcc = false

    private var lastGyroTimestampNs = 0L
    private var dtMeanSec = 0f
    private var dtCount = 0

    private var posX = 0f
    private var posY = 0f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        trajectoryView = findViewById(R.id.trajectoryView)

        findViewById<Button>(R.id.startButton).setOnClickListener { startRealtimePipeline() }
        findViewById<Button>(R.id.stopButton).setOnClickListener { stopRealtimePipeline() }

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        accSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        rotSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

        loadModel()
    }

    private fun loadModel() {
        try {
            module = Module.load(assetFilePath(MODEL_ASSET))
        } catch (e: Exception) {
            module = null
        }
    }

    /** ExecuTorch's Module.load() needs a real filesystem path, so copy the asset out once. */
    private fun assetFilePath(assetName: String): String {
        val outFile = File(filesDir, assetName)
        if (outFile.exists() && outFile.length() > 0) return outFile.absolutePath
        assets.open(assetName).use { input ->
            FileOutputStream(outFile).use { output ->
                input.copyTo(output)
            }
        }
        return outFile.absolutePath
    }

    private fun startRealtimePipeline() {
        if (running) return
        if (module == null) return
        resetPipelineState()
        running = true

        accSensor?.also { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        gyroSensor?.also { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        rotSensor?.also { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
    }

    private fun stopRealtimePipeline() {
        if (!running) return
        running = false
        sensorManager.unregisterListener(this)
    }

    private fun resetPipelineState() {
        for (c in 0 until CHANNELS) java.util.Arrays.fill(window[c], 0f)
        sampleCount = 0
        hasAcc = false
        latestRotation = null
        lastGyroTimestampNs = 0L
        dtMeanSec = 0f
        dtCount = 0
        posX = 0f
        posY = 0f
        inferenceInFlight.set(false)
        trajectoryView.clearTrajectory()
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (!running) return

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                latestAcc = event.values.copyOfRange(0, 3)
                hasAcc = true
            }

            Sensor.TYPE_ROTATION_VECTOR -> {
                // ori = rv(t), no init_rotor correction: there is no ARCore groundtruth at
                // inference time, matching how the model was evaluated (see PIPELINE_FINETUNING.md).
                latestRotation = Quaternion.fromRotationVector(event.values)
            }

            Sensor.TYPE_GYROSCOPE -> {
                latestGyro = event.values.copyOfRange(0, 3)
                processGyroTick(event.timestamp)
            }
        }
    }

    private fun processGyroTick(timestampNs: Long) {
        val ori = latestRotation ?: return
        if (!hasAcc) return

        if (lastGyroTimestampNs == 0L) {
            lastGyroTimestampNs = timestampNs
            return
        }
        val dt = (timestampNs - lastGyroTimestampNs) / 1_000_000_000.0f
        lastGyroTimestampNs = timestampNs
        if (dt <= 0f || dt > 0.2f) return

        dtCount += 1
        dtMeanSec += (dt - dtMeanSec) / dtCount

        val orientedGyro = ori.rotateVector(latestGyro)
        val orientedAcc = ori.rotateVector(latestAcc)

        appendWindow(orientedGyro, orientedAcc)
        if (sampleCount < WINDOW_SIZE) return

        if (!inferenceInFlight.compareAndSet(false, true)) return

        val input = snapshotInput()
        val dtForInference = dtMeanSec

        worker.execute {
            try {
                val inputTensor = Tensor.fromBlob(input, longArrayOf(1, CHANNELS.toLong(), WINDOW_SIZE.toLong()))
                val outputs = module?.forward(EValue.from(inputTensor)) ?: return@execute
                val v = outputs[0].toTensor().dataAsFloatArray

                val dx = v[0] * dtForInference
                val dy = v[1] * dtForInference
                posX += dx
                posY += dy

                ui.post { trajectoryView.addPoint(posX, posY) }
            } finally {
                inferenceInFlight.set(false)
            }
        }
    }

    private fun appendWindow(gyro: FloatArray, acc: FloatArray) {
        if (sampleCount < WINDOW_SIZE) {
            writeSample(sampleCount, gyro, acc)
            sampleCount += 1
            return
        }
        for (c in 0 until CHANNELS) {
            val arr = window[c]
            System.arraycopy(arr, 1, arr, 0, WINDOW_SIZE - 1)
        }
        writeSample(WINDOW_SIZE - 1, gyro, acc)
        sampleCount += 1
    }

    private fun writeSample(index: Int, gyro: FloatArray, acc: FloatArray) {
        window[0][index] = gyro[0]
        window[1][index] = gyro[1]
        window[2][index] = gyro[2]
        window[3][index] = acc[0]
        window[4][index] = acc[1]
        window[5][index] = acc[2]
    }

    private fun snapshotInput(): FloatArray {
        val flat = FloatArray(CHANNELS * WINDOW_SIZE)
        for (c in 0 until CHANNELS) {
            System.arraycopy(window[c], 0, flat, c * WINDOW_SIZE, WINDOW_SIZE)
        }
        return flat
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    override fun onPause() {
        super.onPause()
        stopRealtimePipeline()
    }

    override fun onDestroy() {
        super.onDestroy()
        stopRealtimePipeline()
        worker.shutdownNow()
    }
}
