package com.example.disastersmsclassifier;

import android.content.Context;
import android.content.SharedPreferences;

public final class ClassificationDeduplicator {
    private static final String PREFS = "classification_deduplicator";
    private static final String KEY_LAST_TEXT = "last_text";
    private static final String KEY_LAST_TIME = "last_time";
    private static final long DUPLICATE_WINDOW_MS = 15000;

    private ClassificationDeduplicator() {
    }

    public static synchronized boolean shouldSkip(Context context, String text) {
        String normalized = normalize(text);
        if (normalized.isEmpty()) {
            return true;
        }

        long now = System.currentTimeMillis();
        SharedPreferences prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        String lastText = prefs.getString(KEY_LAST_TEXT, "");
        long lastTime = prefs.getLong(KEY_LAST_TIME, 0L);

        boolean recentlySeen = now - lastTime <= DUPLICATE_WINDOW_MS;
        boolean sameMessage = !lastText.isEmpty()
                && (normalized.contains(lastText) || lastText.contains(normalized));

        if (recentlySeen && sameMessage) {
            return true;
        }

        prefs.edit()
                .putString(KEY_LAST_TEXT, normalized)
                .putLong(KEY_LAST_TIME, now)
                .apply();
        return false;
    }

    private static String normalize(String text) {
        if (text == null) {
            return "";
        }
        return text
                .replaceAll("\\s+", " ")
                .trim();
    }
}
