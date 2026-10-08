package nz.mahere.app

import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log

/**
 * The front camera as a light meter: the widest raw-capable front lens, fully manual (no auto-exposure, ISO at the floor), its exposure stepped down until no sample clips and up when the frame is dim, a few frames a second. Every frame goes to Rust as the raw Bayer samples with the sensor's colour matrix and lens geometry; nothing is kept or shown.
 */
class Probe(private val context: Context, private val onFrame: (RawFrame, FloatArray) -> Unit) {

    class RawFrame(
        val buffer: java.nio.ByteBuffer, val width: Int, val height: Int, val rowStride: Int,
        val cfa: Int, val black: FloatArray, val white: Int, val orientation: Int,
        val tanW: Float, val tanH: Float, val xyzToCam: FloatArray,
    )

    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    private var thread: HandlerThread? = null
    private var handler: Handler? = null
    private var request: CaptureRequest.Builder? = null
    private var exposureNs = 4_000_000L
    private var exposureRange = LongRange(100_000L, 50_000_000L)
    private var lastFrameNs = 0L
    val running get() = device != null

    /** Open the widest raw front camera. False when there is none. */
    fun start(): Boolean {
        if (running) return true
        val cm = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        var best: String? = null
        var bestFocal = Float.MAX_VALUE
        for (id in cm.cameraIdList) {
            val c = cm.getCameraCharacteristics(id)
            if (c.get(CameraCharacteristics.LENS_FACING) != CameraCharacteristics.LENS_FACING_FRONT) continue
            val caps = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: continue
            if (!caps.contains(CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_RAW)) continue
            if (!caps.contains(CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR)) continue
            val focal = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.minOrNull() ?: continue
            if (focal < bestFocal) {
                bestFocal = focal
                best = id
            }
        }
        val id = best ?: run {
            Log.w("mahere", "probe: no raw manual front camera")
            return false
        }
        val c = cm.getCameraCharacteristics(id)
        val sizes = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)?.getOutputSizes(ImageFormat.RAW_SENSOR) ?: return false
        val size = sizes.minByOrNull { it.width.toLong() * it.height } ?: return false
        val physical = c.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE) ?: return false
        val array = c.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE) ?: return false
        // The raw stream may be a crop of the array: scale the physical size by what it delivers.
        val tanW = (physical.width * size.width / array.width) / (2f * bestFocal)
        val tanH = (physical.height * size.height / array.height) / (2f * bestFocal)
        val orientation = c.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
        val cfa = c.get(CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT) ?: 0
        val white = c.get(CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL) ?: 1023
        // The pedestal at each Bayer position, row-major within the quad.
        val black = FloatArray(4) { 64f }
        c.get(CameraCharacteristics.SENSOR_BLACK_LEVEL_PATTERN)?.let { p -> for (i in 0 until 4) black[i] = p.getOffsetForIndex(i % 2, i / 2).toFloat() }
        // The second transform is conventionally the daylight one; XYZ to camera, row-major.
        val cst = c.get(CameraCharacteristics.SENSOR_COLOR_TRANSFORM2) ?: c.get(CameraCharacteristics.SENSOR_COLOR_TRANSFORM1)
        val xyzToCam = FloatArray(9)
        if (cst != null) for (i in 0 until 9) xyzToCam[i] = cst.getElement(i % 3, i / 3).toFloat()
        val isoRange = c.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        val iso = isoRange?.lower ?: 100
        c.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)?.let { exposureRange = LongRange(it.lower, minOf(it.upper, 50_000_000L)) }
        exposureNs = exposureNs.coerceIn(exposureRange)
        Log.i("mahere", "probe: camera $id ${size.width}x${size.height} focal $bestFocal tan $tanW x $tanH orientation $orientation cfa $cfa black ${black.toList()} white $white iso $iso")

        val t = HandlerThread("probe").also { it.start() }
        thread = t
        val h = Handler(t.looper)
        handler = h
        val r = ImageReader.newInstance(size.width, size.height, ImageFormat.RAW_SENSOR, 3)
        reader = r
        r.setOnImageAvailableListener({ rd ->
            val img = rd.acquireLatestImage() ?: return@setOnImageAvailableListener
            try {
                // A few frames a second are plenty for a light that changes with the clouds.
                val now = System.nanoTime()
                if (now - lastFrameNs < 250_000_000L) return@setOnImageAvailableListener
                lastFrameNs = now
                val plane = img.planes[0]
                val stats = FloatArray(2)
                onFrame(RawFrame(plane.buffer, img.width, img.height, plane.rowStride, cfa, black, white, orientation, tanW, tanH, xyzToCam), stats)
                steer(stats[0], stats[1])
            } finally {
                img.close()
            }
        }, h)
        try {
            cm.openCamera(id, object : CameraDevice.StateCallback() {
                override fun onOpened(d: CameraDevice) {
                    device = d
                    val b = d.createCaptureRequest(CameraDevice.TEMPLATE_MANUAL)
                    b.addTarget(r.surface)
                    b.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
                    b.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
                    b.set(CaptureRequest.SENSOR_SENSITIVITY, iso)
                    b.set(CaptureRequest.SENSOR_EXPOSURE_TIME, exposureNs)
                    // Ten frames a second at most: the sensor idles between.
                    b.set(CaptureRequest.SENSOR_FRAME_DURATION, 100_000_000L)
                    request = b
                    @Suppress("DEPRECATION")
                    d.createCaptureSession(listOf(r.surface), object : CameraCaptureSession.StateCallback() {
                        override fun onConfigured(s: CameraCaptureSession) {
                            session = s
                            s.setRepeatingRequest(b.build(), null, h)
                        }
                        override fun onConfigureFailed(s: CameraCaptureSession) {
                            Log.w("mahere", "probe: session failed")
                            stop()
                        }
                    }, h)
                }
                override fun onDisconnected(d: CameraDevice) { stop() }
                override fun onError(d: CameraDevice, error: Int) {
                    Log.w("mahere", "probe: camera error $error")
                    stop()
                }
            }, h)
        } catch (e: SecurityException) {
            Log.w("mahere", "probe: no permission")
            stop()
            return false
        }
        return true
    }

    /** Halve the exposure while more than a twentieth of a percent of the frame clips (a lamp or the sun clips at any exposure; the rest of the frame must not), double it while 99.9% of the frame sits under a quarter of white, within the sensor's range and a 50 ms cap. */
    private fun steer(clipped: Float, p999: Float) {
        val next = when {
            clipped > 0.0005f -> exposureNs / 2
            p999 < 0.25f -> exposureNs * 2
            else -> return
        }.coerceIn(exposureRange)
        if (next == exposureNs) return
        Log.i("mahere", "probe: clipped $clipped p999 $p999, exposure $exposureNs -> $next ns")
        exposureNs = next
        val b = request ?: return
        b.set(CaptureRequest.SENSOR_EXPOSURE_TIME, exposureNs)
        try {
            session?.setRepeatingRequest(b.build(), null, handler)
        } catch (_: Exception) {
        }
    }

    fun stop() {
        try { session?.close() } catch (_: Exception) {}
        session = null
        try { device?.close() } catch (_: Exception) {}
        device = null
        reader?.close()
        reader = null
        request = null
        thread?.quitSafely()
        thread = null
        handler = null
    }
}
