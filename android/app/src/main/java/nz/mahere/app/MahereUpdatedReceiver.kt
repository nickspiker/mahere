package nz.mahere.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

/// Mahere back after it updated itself (Nick 2026-10-18: an update installed while the map was open, the app vanished, and it read as a crash). Android closes an app while it replaces it and starts nothing afterwards; the system's MY_PACKAGE_REPLACED (after Photon's) opens the map again where it was, and a tap-to-open notification is always posted in case Android refuses a start from the background.
class MahereUpdatedReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Intent.ACTION_MY_PACKAGE_REPLACED) return
        val launch = Intent(context, MahereActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        }
        try {
            context.startActivity(launch)
            android.util.Log.i("mahere", "reopened after the update")
        } catch (e: Exception) {
            android.util.Log.i("mahere", "reopen after the update refused (${e.message}): the notification carries it")
        }
        try {
            val nm = context.getSystemService(NotificationManager::class.java)
            nm.createNotificationChannel(NotificationChannel(CHANNEL, "Updates", NotificationManager.IMPORTANCE_HIGH))
            val pi = PendingIntent.getActivity(context, 7, launch, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
            val version = try { context.packageManager.getPackageInfo(context.packageName, 0).versionName } catch (e: Exception) { null }
            val n = Notification.Builder(context, CHANNEL)
                .setSmallIcon(R.mipmap.ic_launcher)
                .setContentTitle("Mahere updated" + if (version.isNullOrEmpty()) "" else " to ${version.substringBefore('+')}")
                .setContentText("Tap to open")
                .setContentIntent(pi)
                .setAutoCancel(true)
                .build()
            nm.notify(NOTIFICATION, n)
        } catch (e: Exception) {
            android.util.Log.e("mahere", "update notification failed: ${e.message}")
        }
    }

    companion object {
        private const val CHANNEL = "updates"
        private const val NOTIFICATION = 7001
    }
}
