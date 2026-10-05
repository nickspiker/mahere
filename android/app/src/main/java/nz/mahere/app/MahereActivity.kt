package nz.mahere.app

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.PixelFormat
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
    }

    private external fun nativeInit(width: Int, height: Int, dataDir: String): Long
    private external fun nativeResize(ptr: Long, width: Int, height: Int)
    private external fun nativeDraw(ptr: Long, surface: android.view.Surface): Boolean
    private external fun nativeOnTouch(
        ptr: Long, action: Int, count: Int,
        x0: Float, y0: Float, x1: Float, y1: Float,
    ): Int
    private external fun nativeOnLocation(ptr: Long, lat: Double, lon: Double, accuracy: Float)
    private external fun nativeOnPause(ptr: Long)

    private lateinit var surfaceView: SurfaceView
    @Volatile private var nativePtr = 0L
    private var surfaceReady = false
    private var assetsStaged = false
    private var initInFlight = false
    private var pendingSize: Pair<Int, Int>? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        surfaceView = SurfaceView(this)
        // fluor writes 32-bit pixels into the locked buffer; without this the
        // Surface can default to a 16-bit format and every frame writes past
        // the allocation (SEGV_ACCERR in nativeDraw).
        surfaceView.holder.setFormat(PixelFormat.RGBA_8888)
        surfaceView.holder.addCallback(this)
        setContentView(surfaceView)
        hideSystemBars()

        // Stage bundled DEM tiles into filesDir once (the tiff reader wants
        // real file paths; assets are zip entries).
        thread {
            stageAssetDir("cells")
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

    /// Recursively copy an asset directory into filesDir (skip already-staged
    /// files by size). The cells dir is {layer}/{depth}/{prefix}.vsf.zst.
    private fun stageAssetDir(rel: String) {
        val children = assets.list(rel).orEmpty()
        if (children.isEmpty()) {
            val out = java.io.File(filesDir, rel)
            if (!out.exists() || out.length() == 0L) {
                out.parentFile?.mkdirs()
                assets.open(rel).use { input ->
                    out.outputStream().use { input.copyTo(it) }
                }
            }
            return
        }
        for (c in children) stageAssetDir("$rel/$c")
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

    /// Idempotent frame-loop (re)start — the resume/surfaceChanged ordering
    /// varies by path (screen-off vs app-switch), so every reentry point
    /// calls this instead of guessing which event comes last.
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
            // The first init decodes ~500 MB of GeoTIFF — off the UI thread,
            // or the app black-screens (and ANRs on touch) for ~30 s.
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
            Choreographer.getInstance().postFrameCallback(this)
        }
    }

    override fun onResume() {
        super.onResume()
        hideSystemBars()
        startFrames()
    }

    override fun onPause() {
        super.onPause()
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

    // ---- GPS ----

    private val locationListener = LocationListener { loc: Location ->
        if (nativePtr != 0L) {
            nativeOnLocation(nativePtr, loc.latitude, loc.longitude, loc.accuracy)
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
    }
}
