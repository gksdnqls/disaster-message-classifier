package com.example.disastersmsclassifier;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

public final class PredictionClient {
    public interface Callback {
        void onSuccess(String label, int labelId, String rawJson);
        void onError(Exception exception);
    }

    private PredictionClient() {
    }

    public static void predictAsync(String endpoint, String text, Callback callback) {
        new Thread(() -> {
            try {
                JSONObject request = new JSONObject();
                request.put("text", text);

                HttpURLConnection connection = (HttpURLConnection) new URL(endpoint).openConnection();
                connection.setRequestMethod("POST");
                connection.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
                connection.setConnectTimeout(10000);
                connection.setReadTimeout(20000);
                connection.setDoOutput(true);

                byte[] payload = request.toString().getBytes(StandardCharsets.UTF_8);
                try (OutputStream os = connection.getOutputStream()) {
                    os.write(payload);
                }

                int status = connection.getResponseCode();
                BufferedReader reader = new BufferedReader(new InputStreamReader(
                        status >= 200 && status < 300
                                ? connection.getInputStream()
                                : connection.getErrorStream(),
                        StandardCharsets.UTF_8
                ));

                StringBuilder responseBuilder = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    responseBuilder.append(line);
                }

                if (status < 200 || status >= 300) {
                    throw new IllegalStateException("HTTP " + status + ": " + responseBuilder);
                }

                JSONObject response = new JSONObject(responseBuilder.toString());
                String label = response.getString("label");
                int labelId = response.getInt("label_id");
                callback.onSuccess(label, labelId, response.toString(2));
            } catch (Exception e) {
                callback.onError(e);
            }
        }).start();
    }
}
