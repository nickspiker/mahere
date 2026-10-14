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
import android.content.BroadcastReceiver
import android.content.Intent
import android.content.IntentFilter
import android.net.Uri
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
        tool: Int, pressure: Float,
    ): Int
    private external fun nativePenOut(ptr: Long, out: Boolean)
    private external fun nativeOnLocation(ptr: Long, lat: Double, lon: Double, accuracy: Float)
    private external fun nativeGoTo(ptr: Long, lat: Double, lon: Double, ppd: Double)
    private external fun nativeOpenArea(ptr: Long, bytes: ByteArray)
    private external fun nativeBack(ptr: Long): Boolean
    private external fun nativeTakeInstall(ptr: Long): String?
    private external fun nativeSetUnmetered(ptr: Long, unmetered: Boolean)
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
        Probe(this, { f, stats ->
            val p = nativePtr
            if (p != 0L) nativeOnProbe(p, f.buffer, f.width, f.height, f.rowStride, f.packed10, f.cfa, f.black, f.white, f.orientation, f.tanW, f.tanH, f.xyzToCam, stats, f.stop, f.longStop)
        }) {
            // The camera gave up: the mode turns itself off, on the UI thread where the engine lives.
            runOnUiThread { if (nativePtr != 0L) nativeProbeDenied(nativePtr) }
        }
    }
    private var cameraAsked = false

    private lateinit var surfaceView: SurfaceView
    @Volatile private var nativePtr = 0L
    private var surfaceReady = false
    private var assetsStaged = false
    private var initInFlight = false
    private var pendingSize: Pair<Int, Int>? = null
    /// A place from a geo: intent, waiting for the native side to exist: latitude, longitude, pixels per degree (or -1 for the zoom the map has).
    private var pendingGeo: Triple<Double, Double, Double>? = null

    /// An area document shared to the app, waiting for the native side to exist.
    private var pendingArea: ByteArray? = null

    /// Open the system's share sheet on a file the native side wrote under files/share (an area document), through the app's own provider. Called from native, on the UI thread.
    fun shareFile(path: String) {
        val f = java.io.File(path)
        val uri = Uri.parse("content://nz.mahere.app.share/" + Uri.encode(f.name))
        android.util.Log.i("mahere", "share $path as $uri (${f.length()} bytes)")
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "application/vnd.mahere.area"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_SUBJECT, f.name.removeSuffix(".vsf.zst"))
            putExtra(Intent.EXTRA_TEXT, "A mahere area: " + f.name.removeSuffix(".vsf.zst").replace('-', ' '))
            // The clip carries the grant to the share sheet itself, so it can show the file.
            clipData = android.content.ClipData.newUri(contentResolver, "mahere area", uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        val chooser = Intent.createChooser(send, "Share area").apply { addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION) }
        runCatching { startActivity(chooser) }.onFailure { android.util.Log.e("mahere", "share failed", it) }
    }

    /// An area document sent or opened with the app: its bytes to the native side, or kept until it is up. Anything that does not decode is ignored there.
    private fun takeArea(intent: Intent?) {
        if (intent == null) return
        val found: Uri? = when (intent.action) {
            Intent.ACTION_SEND -> @Suppress("DEPRECATION") (intent.getParcelableExtra(Intent.EXTRA_STREAM) as? Uri)
            Intent.ACTION_VIEW -> intent.data?.takeIf { it.scheme == "content" || it.scheme == "file" }
            else -> null
        }
        val uri: Uri = found ?: return
        val bytes = runCatching { contentResolver.openInputStream(uri)?.use { it.readBytes() } }.getOrNull() ?: return
        if (bytes.size > 64 shl 20) return
        val ptr = nativePtr
        if (ptr != 0L) nativeOpenArea(ptr, bytes) else pendingArea = bytes
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        surfaceView = SurfaceView(this)
        // fluor writes 32-bit pixels into the locked buffer; without this the Surface can default to a 16-bit format and every frame writes past the allocation (SEGV_ACCERR in nativeDraw).
        surfaceView.holder.setFormat(PixelFormat.RGBA_8888)
        surfaceView.holder.addCallback(this)
        setContentView(surfaceView)
        hideSystemBars()
        takeGeo(intent)
        takeArea(intent)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(penReceiver, IntentFilter("com.samsung.pen.INSERT"), Context.RECEIVER_EXPORTED)
        } else {
            @Suppress("DEPRECATION")
            registerReceiver(penReceiver, IntentFilter("com.samsung.pen.INSERT"))
        }
        runCatching { getSystemService(android.net.ConnectivityManager::class.java).registerDefaultNetworkCallback(networkCallback) }
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
                    pendingGeo?.let { (lat, lon, ppd) -> nativeGoTo(ptr, lat, lon, ppd) }
                    pendingGeo = null
                    pendingArea?.let { nativeOpenArea(ptr, it) }
                    pendingArea = null
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
            nativeTakeInstall(nativePtr)?.let { installApk(it) }
            Choreographer.getInstance().postFrameCallback(this)
        }
    }

    // ---- Updates (after Photon's): a downloaded, verified build installed through a package installer session ----

    /// Install an update the engine downloaded and checked against the signed manifest: a PackageInstaller session, off the main thread (it copies the whole APK). Once Mahere is its own installer of record, Android 12 and later install without asking; the first time, the system's confirm comes up through MahereInstallReceiver.
    private fun installApk(path: String) {
        Thread({
            try {
                val installer = packageManager.packageInstaller
                val params = android.content.pm.PackageInstaller.SessionParams(android.content.pm.PackageInstaller.SessionParams.MODE_FULL_INSTALL).apply {
                    setAppPackageName(packageName)
                    if (android.os.Build.VERSION.SDK_INT >= 31) {
                        setRequireUserAction(android.content.pm.PackageInstaller.SessionParams.USER_ACTION_NOT_REQUIRED)
                    }
                }
                val id = installer.createSession(params)
                installer.openSession(id).use { session ->
                    java.io.File(path).inputStream().use { input ->
                        session.openWrite("mahere-update.apk", 0, -1).use { out ->
                            input.copyTo(out)
                            session.fsync(out)
                        }
                    }
                    // Mutable: the installer writes its status into it.
                    val intent = android.content.Intent(this, MahereInstallReceiver::class.java).apply { action = "nz.mahere.app.INSTALL_RESULT" }
                    val flags = if (android.os.Build.VERSION.SDK_INT >= 31) android.app.PendingIntent.FLAG_UPDATE_CURRENT or android.app.PendingIntent.FLAG_MUTABLE else android.app.PendingIntent.FLAG_UPDATE_CURRENT
                    session.commit(android.app.PendingIntent.getBroadcast(this, id, intent, flags).intentSender)
                }
            } catch (e: Exception) {
                android.util.Log.e("mahere", "update install failed: ${e.message}")
            }
        }, "mahere-install").start()
    }

    /// Whether the network is unmetered, told to the engine as it changes: automatic updates download only then.
    private val networkCallback = object : android.net.ConnectivityManager.NetworkCallback() {
        override fun onCapabilitiesChanged(network: android.net.Network, caps: android.net.NetworkCapabilities) {
            val ptr = nativePtr
            if (ptr != 0L) nativeSetUnmetered(ptr, caps.hasCapability(android.net.NetworkCapabilities.NET_CAPABILITY_NOT_METERED))
        }
        override fun onLost(network: android.net.Network) {
            val ptr = nativePtr
            if (ptr != 0L) nativeSetUnmetered(ptr, false)
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

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        takeGeo(intent)
        takeArea(intent)
    }

    /// A geo: URI's place, applied now or once the native side is up. `geo:lat,lon`, `geo:0,0?q=lat,lon`, and `?z=` as a web zoom level (metres per pixel 156543 / 2^z at the equator, so pixels per degree 0.711 · 2^z).
    private fun takeGeo(intent: Intent?) {
        val uri = intent?.data ?: return
        if (uri.scheme != "geo") return
        val text = uri.schemeSpecificPart ?: return
        val pair = Regex("(-?\\d+(?:\\.\\d+)?),(-?\\d+(?:\\.\\d+)?)")
        val q = Regex("[?&]q=([^&]*)").find(text)?.groupValues?.get(1)
        val m = (q?.let { pair.find(it) }) ?: pair.find(text) ?: return
        val lat = m.groupValues[1].toDoubleOrNull() ?: return
        val lon = m.groupValues[2].toDoubleOrNull() ?: return
        if (lat == 0.0 && lon == 0.0 && q == null) return
        val z = Regex("[?&]z=(\\d+(?:\\.\\d+)?)").find(text)?.groupValues?.get(1)?.toDoubleOrNull()
        val ppd = if (z != null) 0.711 * Math.pow(2.0, z) else -1.0
        val ptr = nativePtr
        if (ptr != 0L) nativeGoTo(ptr, lat, lon, ppd) else pendingGeo = Triple(lat, lon, ppd)
    }

    // Back steps through the map's own pages first (the panel one page up, then closed, then a measurement cleared); only with nothing left there does it leave the app.
    @Deprecated("Back is routed through the map first")
    override fun onBackPressed() {
        val ptr = nativePtr
        if (ptr != 0L && nativeBack(ptr)) return
        @Suppress("DEPRECATION")
        super.onBackPressed()
    }

    override fun onDestroy() {
        runCatching { unregisterReceiver(penReceiver) }
        runCatching { getSystemService(android.net.ConnectivityManager::class.java).unregisterNetworkCallback(networkCallback) }
        super.onDestroy()
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

    /// Samsung says when the S Pen leaves or returns to its slot: the broadcast `com.samsung.pen.INSERT` with `penInsert`. While the pen is out, fingers only navigate.
    private val penReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val inserted = intent?.getBooleanExtra("penInsert", true) ?: true
            val ptr = nativePtr
            if (ptr != 0L) nativePenOut(ptr, !inserted)
        }
    }

    /// A stylus hovering over the screen is a pen that is out, on any phone.
    override fun onGenericMotionEvent(event: MotionEvent): Boolean {
        if (event.getToolType(0) == MotionEvent.TOOL_TYPE_STYLUS && event.actionMasked == MotionEvent.ACTION_HOVER_MOVE) {
            val ptr = nativePtr
            if (ptr != 0L) nativePenOut(ptr, true)
        }
        return super.onGenericMotionEvent(event)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (nativePtr == 0L) return true
        val count = event.pointerCount
        val x1 = if (count > 1) event.getX(1) else 0f
        val y1 = if (count > 1) event.getY(1) else 0f
        nativeOnTouch(
            nativePtr, event.actionMasked, count,
            event.getX(0), event.getY(0), x1, y1,
            event.getToolType(0), event.getPressure(0),
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
