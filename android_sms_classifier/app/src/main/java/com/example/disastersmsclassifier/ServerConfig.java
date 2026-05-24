package com.example.disastersmsclassifier;

import android.content.Context;
import android.content.SharedPreferences;

public final class ServerConfig {
    private static final String PREFS = "server_config";
    private static final String KEY_URL = "predict_url";
    private static final String DEFAULT_URL = "https://vbcbxb-disaster-classifier.hf.space/predict";

    private ServerConfig() {
    }

    public static String getPredictUrl(Context context) {
        SharedPreferences prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        return prefs.getString(KEY_URL, DEFAULT_URL);
    }

    public static void setPredictUrl(Context context, String url) {
        SharedPreferences prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        prefs.edit().putString(KEY_URL, url).apply();
    }
}
