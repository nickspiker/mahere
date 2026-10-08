package nz.mahere.app

import android.content.pm.ActivityInfo
import android.os.Build
import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.PixelFormat
import android.hardware.GeomagneticField
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.view.Choreographer
import android.view.MotionEvent
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import kotlin.concurrent.thread

/**
 * mahere's Android shell: a fullscreen SurfaceView over fluor's AndroidShell.
 *
 * Deliberately minimal — no gesture detectors anywhere. Raw pointer
 * positions (up to two) forward straight to Rust, which solves the map
 * transform so the geography under each finger stays under that finger.
 */
class MahereActivity : Activity(), SurfaceHolder.Callback, Choreographer.FrameCallback {

    companion object {
        init {
            System.loadLibrary("mahere_android")
        }
        private const val LOCATION_PERMISSION_REQUEST = 1
        private const val CAMERA_PERMISSION_REQUEST = 2
    }

    private external fun nativeInit(width: Int, height: Int, dataDir: String): Long
    private external fun nativeResize(ptr: Long, width: Int, height: Int)
    private external fun nativeDraw(ptr: Long, surface: android.view.Surface): Boolean
    private external fun nativeOnTouch(
        ptr: Long, action: Int, count: Int,
        x0: Float, y0: Float, x1: Float, y1: Float,
    ): Int
    private external fun nativeOnLocation(ptr: Long, lat: Double, lon: Double, accuracy: Float)
    private external fun nativeOnDeclination(ptr: Long, declinationDeg: Float)
    private external fun nativeOnOrientation(ptr: Long, r0: Float, r1: Float, r2: Float, r3: Float, r4: Float, r5: Float, r6: Float, r7: Float, r8: Float)
    private external fun nativeOnPause(ptr: Long)
    private external fun nativeProbeWanted(ptr: Long): Boolean
    private external fun nativeProbeDenied(ptr: Long)
    private external fun nativeOnProbe(
        ptr: Long, buffer: java.nio.ByteBuffer, width: Int, height: Int, rowStride: Int, packed10: Boolean,
        cfa: Int, black: FloatArray, white: Int, orientation: Int, tanW: Float, tanH: Float, xyzToCam: FloatArray, stats: FloatArray,
        stop: Int, longStop: Int,
    )

    // The front camera as the light, opened when the engine asks for it (the Real light row) and closed when it stops asking.
    private val probe by lazy {
        Probe(this) { f, stats ->
            val p = nativePtr
            if (p != 0L) nativeOnProbe(p, f.buffer, f.width, f.height, f.rowStride, f.packed10, f.cfa, f.black, f.white, f.orientation, f.tanW, f.tanH, f.xyzToCam, stats, f.stop, f.longStop)
        }
    }
    private var cameraAsked = false

    private lateinit var surfaceView: SurfaceView
    @Volatile private var nativePtr = 0L
    private var surfaceReady = false
    private var assetsStaged = false
    private var initInFlight = false
    private var pendingSize: Pair<Int, Int>? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        surfaceView = SurfaceView(this)
        // fluor writes 32-bit pixels into the locked buffer; without this the Surface can default to a 16-bit format and every frame writes past the allocation (SEGV_ACCERR in nativeDraw).
        surfaceView.holder.setFormat(PixelFormat.RGBA_8888)
        surfaceView.holder.addCallback(this)
        setContentView(surfaceView)
        hideSystemBars()
        // The map tags its buffers BT.2020 (gpu_host::tag_bt2020); wide-gamut mode lets the panel show them without an sRGB clamp, and minimal post-processing asks the compositor to skip vendor saturation passes.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            window.colorMode = ActivityInfo.COLOR_MODE_WIDE_COLOR_GAMUT
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.attributes = window.attributes.also { it.preferMinimalPostProcessing = true }
        }

        // Cells stream from the bucket into the vault now; nothing ships in the APK. Drop the staged copy earlier builds left behind.
        thread {
            for (legacy in listOf("cells", "dem", "featpack.vsf")) {
                java.io.File(filesDir, legacy).deleteRecursively()
            }
            runOnUiThread {
                assetsStaged = true
                maybeInit()
            }
        }

        if (checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION),
                LOCATION_PERMISSION_REQUEST,
            )
        } else {
            startLocation()
        }
    }

    private fun hideSystemBars() {
        @Suppress("DEPRECATION")
        window.decorView.systemUiVisibility = (
            View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                or View.SYSTEM_UI_FLAG_FULLSCREEN
                or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                or View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                or View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
            )
    }

    /// Idempotent frame-loop (re)start — the resume/surfaceChanged ordering / varies by path (screen-off vs app-switch), so every reentry point / calls this instead of guessing which event comes last.
    private fun startFrames() {
        Choreographer.getInstance().removeFrameCallback(this)
        if (nativePtr != 0L && surfaceReady) {
            Choreographer.getInstance().postFrameCallback(this)
        }
    }

    private fun maybeInit() {
        val (w, h) = pendingSize ?: return
        if (!assetsStaged || !surfaceReady) return
        if (nativePtr == 0L) {
            if (initInFlight) return
            initInFlight = true
            // The first init decodes ~500 MB of GeoTIFF — off the UI thread, or the app black-screens (and ANRs on touch) for ~30 s.
            thread {
                val ptr = nativeInit(w, h, filesDir.absolutePath)
                runOnUiThread {
                    nativePtr = ptr
                    initInFlight = false
                    // The surface may have changed size during the long init.
                    pendingSize?.let { (pw, ph) ->
                        if (pw != w || ph != h) nativeResize(ptr, pw, ph)
                    }
                    startFrames()
                }
            }
        } else {
            nativeResize(nativePtr, w, h)
            startFrames()
        }
    }

    // ---- Surface lifecycle ----

    override fun surfaceCreated(holder: SurfaceHolder) {}

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        surfaceReady = true
        pendingSize = Pair(width, height)
        maybeInit()
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        surfaceReady = false
        Choreographer.getInstance().removeFrameCallback(this)
    }

    // ---- Frame loop ----

    override fun doFrame(frameTimeNanos: Long) {
        if (nativePtr != 0L && surfaceReady) {
            nativeDraw(nativePtr, surfaceView.holder.surface)
            syncProbe()
            Choreographer.getInstance().postFrameCallback(this)
        }
    }

    /// Open or close the front camera to match the engine's wish; the first wish asks for the permission, and a refusal turns the mode off.
    private fun syncProbe() {
        val want = nativeProbeWanted(nativePtr)
        if (want && !probe.running) {
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                if (!cameraAsked) {
                    cameraAsked = true
                    requestPermissions(arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_REQUEST)
                }
                return
            }
            if (!probe.start()) nativeProbeDenied(nativePtr)
        } else if (!want && probe.running) {
            probe.stop()
        }
    }

    override fun onResume() {
        super.onResume()
        hideSystemBars()
        startFrames()
        startHeading()
    }

    override fun onPause() {
        super.onPause()
        stopHeading()
        probe.stop()
        Choreographer.getInstance().removeFrameCallback(this)
        if (nativePtr != 0L) nativeOnPause(nativePtr)
    }

    // ---- Input: raw pointers, no detectors ----

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (nativePtr == 0L) return true
        val count = event.pointerCount
        val x1 = if (count > 1) event.getX(1) else 0f
        val y1 = if (count > 1) event.getY(1) else 0f
        nativeOnTouch(
            nativePtr, event.actionMasked, count,
            event.getX(0), event.getY(0), x1, y1,
        )
        return true
    }

    // ---- Heading: the rotation vector, as degrees clockwise from north, so the engine can keep the sun where it physically is ----

    private val headingListener = object : SensorEventListener {
        private val rot = FloatArray(9)
        override fun onSensorChanged(event: SensorEvent) {
            if (nativePtr == 0L) return
            SensorManager.getRotationMatrixFromVector(rot, event.values)
            nativeOnOrientation(nativePtr, rot[0], rot[1], rot[2], rot[3], rot[4], rot[5], rot[6], rot[7], rot[8])
        }
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }

    private fun startHeading() {
        val sm = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        sm.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)?.let { sm.registerListener(headingListener, it, SensorManager.SENSOR_DELAY_UI) }
    }

    private fun stopHeading() {
        (getSystemService(Context.SENSOR_SERVICE) as SensorManager).unregisterListener(headingListener)
    }

    // ---- GPS ----

    private val locationListener = LocationListener { loc: Location ->
        if (nativePtr != 0L) {
            nativeOnLocation(nativePtr, loc.latitude, loc.longitude, loc.accuracy)
            // The sensor's north is magnetic; the engine needs the local declination to light by the true sun.
            val field = GeomagneticField(loc.latitude.toFloat(), loc.longitude.toFloat(), loc.altitude.toFloat(), System.currentTimeMillis())
            nativeOnDeclination(nativePtr, field.declination)
        }
    }

    private fun startLocation() {
        val lm = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        try {
            lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1000L, 0f, locationListener)
        } catch (_: SecurityException) {
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == LOCATION_PERMISSION_REQUEST &&
            grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED
        ) {
            startLocation()
        }
        if (requestCode == CAMERA_PERMISSION_REQUEST) {
            cameraAsked = false
            if (grantResults.firstOrNull() != PackageManager.PERMISSION_GRANTED && nativePtr != 0L) nativeProbeDenied(nativePtr)
        }
    }
}
