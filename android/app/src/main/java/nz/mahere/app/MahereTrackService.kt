package nz.mahere.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.TriggerEvent
import android.hardware.TriggerEventListener
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper

/// Recording a track, the screen on or off (Nick 2026-10-13): a foreground service that owns the GPS while recording.
/// The screen on, fixes come live every second for the map; off, the GNSS chip keeps taking one a second into its own buffer and hands them over every 90 s, so the main processor sleeps between batches.
/// Stillness is the gyroscope's call (Nick: gyro polling): no turn above a small rate for 45 s, with the GPS agreeing the phone has stopped (a smooth highway would fool the gyro alone), and the GPS is switched off; a turn, Android's significant-motion trigger, or a check every five minutes switches it back on.
class MahereTrackService : Service(), SensorEventListener {
    companion object {
        init {
            System.loadLibrary("mahere_android")
        }

        @Volatile var running = false
        private var instance: MahereTrackService? = null

        /// The app is in front (fixes live) or not (fixes batched).
        fun setLive(live: Boolean) {
            instance?.let { s -> s.handler.post { s.live = live; s.applyGps() } }
        }

        private const val CHANNEL = "track"
        private const val NOTIFICATION = 2
        private const val BATCH_MS = 90_000L
        private const val STILL_MS = 45_000L
        private const val CHECK_MS = 300_000L
        /// Rad/s: below this the phone is not turning; above `MOVE`, it is moving.
        private const val STILL = 0.12f
        private const val MOVE = 0.35f
        /// M/s: the GPS's speed below which the phone has stopped.
        private const val STOPPED = 0.7f
    }

    private external fun nativeOnTrackFixes(ptr: Long, lats: DoubleArray, lons: DoubleArray, accs: FloatArray, times: LongArray, heights: DoubleArray, vaccs: FloatArray, speeds: FloatArray, bearings: FloatArray)

    private val handler = Handler(Looper.getMainLooper())
    private var live = true
    private var paused = false
    private var lastTurn = 0L
    private var lastSpeed = 0f
    private var moveSamples = 0
    private var moveStart = 0L

    private val gps = object : LocationListener {
        override fun onLocationChanged(loc: Location) = deliver(listOf(loc))
        override fun onLocationChanged(locs: MutableList<Location>) = deliver(locs)
        @Deprecated("Deprecated in Java")
        override fun onStatusChanged(provider: String?, status: Int, extras: android.os.Bundle?) {}
        override fun onProviderEnabled(provider: String) {}
        override fun onProviderDisabled(provider: String) {}
    }

    private val significantMotion = object : TriggerEventListener() {
        override fun onTrigger(event: TriggerEvent?) {
            handler.post { resume() }
        }
    }

    private val check = Runnable { if (paused) resume() }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        instance = this
        running = true
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(CHANNEL, "Track recording", NotificationManager.IMPORTANCE_LOW))
        val open = PendingIntent.getActivity(this, 0, Intent(this, MahereActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
        val n = Notification.Builder(this, CHANNEL)
            .setContentTitle("Mahere")
            .setContentText("Recording a track")
            .setSmallIcon(R.mipmap.ic_launcher)
            .setOngoing(true)
            .setContentIntent(open)
            .build()
        if (Build.VERSION.SDK_INT >= 29) {
            startForeground(NOTIFICATION, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION)
        } else {
            startForeground(NOTIFICATION, n)
        }
        lastTurn = System.currentTimeMillis()
        applyGps()
        applySensors()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int = START_STICKY

    override fun onDestroy() {
        handler.removeCallbacks(check)
        stopGps()
        (getSystemService(SENSOR_SERVICE) as SensorManager).let { sm ->
            sm.unregisterListener(this)
            sm.getDefaultSensor(Sensor.TYPE_SIGNIFICANT_MOTION)?.let { sm.cancelTriggerSensor(significantMotion, it) }
        }
        running = false
        instance = null
        super.onDestroy()
    }

    /// The fixes into the track, at the times they were taken; the GPS's speed remembered for the stillness rule.
    private fun deliver(locs: List<Location>) {
        if (locs.isEmpty()) return
        locs.last().let { if (it.hasSpeed()) lastSpeed = it.speed }
        val ptr = MahereActivity.sharedPtr
        if (ptr == 0L) return
        // Everything the receiver measured, NaN where it gave nothing; the height is above the WGS84 ellipsoid, as GPS gives it.
        nativeOnTrackFixes(
            ptr,
            DoubleArray(locs.size) { locs[it].latitude },
            DoubleArray(locs.size) { locs[it].longitude },
            FloatArray(locs.size) { locs[it].accuracy },
            LongArray(locs.size) { locs[it].time },
            DoubleArray(locs.size) { if (locs[it].hasAltitude()) locs[it].altitude else Double.NaN },
            FloatArray(locs.size) { if (locs[it].hasVerticalAccuracy()) locs[it].verticalAccuracyMeters else Float.NaN },
            FloatArray(locs.size) { if (locs[it].hasSpeed()) locs[it].speed else Float.NaN },
            FloatArray(locs.size) { if (locs[it].hasBearing()) locs[it].bearing else Float.NaN },
        )
    }

    /// The GPS as the moment wants it: off while still, else one fix a second, live or batched.
    private fun applyGps() {
        stopGps()
        if (paused) return
        val lm = getSystemService(LOCATION_SERVICE) as LocationManager
        try {
            if (Build.VERSION.SDK_INT >= 31) {
                val req = android.location.LocationRequest.Builder(1000L)
                    .setQuality(android.location.LocationRequest.QUALITY_HIGH_ACCURACY)
                    .setMaxUpdateDelayMillis(if (live) 0L else BATCH_MS)
                    .build()
                lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, req, mainExecutor, gps)
            } else {
                lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1000L, 0f, gps, Looper.getMainLooper())
            }
        } catch (_: SecurityException) {
        }
    }

    private fun stopGps() {
        (getSystemService(LOCATION_SERVICE) as LocationManager).removeUpdates(gps)
    }

    /// The gyroscope a few times a second; with the screen off its samples gather on the sensor hub and arrive every few seconds, stopped or moving, so the processor sleeps between (the significant-motion trigger, a wake-up sensor, catches a start sooner).
    private fun applySensors() {
        val sm = getSystemService(SENSOR_SERVICE) as SensorManager
        sm.unregisterListener(this)
        val gyro = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE) ?: return
        sm.registerListener(this, gyro, 200_000, if (live) 0 else 5_000_000)
    }

    private fun pause() {
        if (paused) return
        paused = true
        applyGps()
        applySensors()
        val sm = getSystemService(SENSOR_SERVICE) as SensorManager
        sm.getDefaultSensor(Sensor.TYPE_SIGNIFICANT_MOTION)?.let { sm.requestTriggerSensor(significantMotion, it) }
        handler.postDelayed(check, CHECK_MS)
    }

    private fun resume() {
        if (!paused) return
        paused = false
        lastTurn = System.currentTimeMillis()
        lastSpeed = Float.MAX_VALUE
        handler.removeCallbacks(check)
        val sm = getSystemService(SENSOR_SERVICE) as SensorManager
        sm.getDefaultSensor(Sensor.TYPE_SIGNIFICANT_MOTION)?.let { sm.cancelTriggerSensor(significantMotion, it) }
        applyGps()
        applySensors()
    }

    override fun onSensorChanged(event: SensorEvent) {
        val (x, y, z) = Triple(event.values[0], event.values[1], event.values[2])
        val rate = kotlin.math.sqrt(x * x + y * y + z * z)
        val now = System.currentTimeMillis()
        if (rate > STILL) lastTurn = now
        if (paused) {
            // Moving again: a few clear turns within two seconds.
            if (rate > MOVE) {
                if (now - moveStart > 2_000L) {
                    moveStart = now
                    moveSamples = 0
                }
                if (++moveSamples >= 3) resume()
            }
        } else if (now - lastTurn > STILL_MS && lastSpeed < STOPPED) {
            pause()
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
