package com.example.disastersmsclassifier;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public class MainActivity extends android.app.Activity {
    private static final int REQUEST_PERMISSIONS = 1001;

    private EditText serverUrlEditText;
    private EditText manualTextEditText;
    private TextView resultTextView;
    private TextView statusTextView;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        NotificationHelper.ensureChannel(this);

        serverUrlEditText = findViewById(R.id.serverUrlEditText);
        manualTextEditText = findViewById(R.id.manualTextEditText);
        resultTextView = findViewById(R.id.resultTextView);
        statusTextView = findViewById(R.id.statusTextView);
        Button saveServerButton = findViewById(R.id.saveServerButton);
        Button notificationAccessButton = findViewById(R.id.notificationAccessButton);
        Button manualPredictButton = findViewById(R.id.manualPredictButton);

        serverUrlEditText.setText(ServerConfig.getPredictUrl(this));

        saveServerButton.setOnClickListener(v -> {
            String url = serverUrlEditText.getText().toString().trim();
            ServerConfig.setPredictUrl(this, url);
            Toast.makeText(this, "서버 주소 저장 완료", Toast.LENGTH_SHORT).show();
        });

        notificationAccessButton.setOnClickListener(v ->
                startActivity(new Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS))
        );

        manualPredictButton.setOnClickListener(v -> runManualPrediction());
        requestRuntimePermissions();
    }

    private void requestRuntimePermissions() {
        java.util.ArrayList<String> permissions = new java.util.ArrayList<>();
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECEIVE_SMS) != PackageManager.PERMISSION_GRANTED) {
            permissions.add(Manifest.permission.RECEIVE_SMS);
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_SMS) != PackageManager.PERMISSION_GRANTED) {
            permissions.add(Manifest.permission.READ_SMS);
        }
        if (Build.VERSION.SDK_INT >= 33
                && ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS);
        }
        if (!permissions.isEmpty()) {
            ActivityCompat.requestPermissions(this, permissions.toArray(new String[0]), REQUEST_PERMISSIONS);
        }
    }

    private void runManualPrediction() {
        String endpoint = serverUrlEditText.getText().toString().trim();
        String text = manualTextEditText.getText().toString().trim();
        if (text.isEmpty()) {
            Toast.makeText(this, "문장을 입력하세요.", Toast.LENGTH_SHORT).show();
            return;
        }

        resultTextView.setText("분류 중...");
        PredictionClient.predictAsync(endpoint, text, new PredictionClient.Callback() {
            @Override
            public void onSuccess(String label, int labelId, String rawJson) {
                runOnUiThread(() -> {
                    resultTextView.setText("예측 결과: " + label + " (" + labelId + ")");
                    statusTextView.setText(rawJson);
                });
            }

            @Override
            public void onError(Exception exception) {
                runOnUiThread(() -> {
                    resultTextView.setText("예측 실패");
                    statusTextView.setText(exception.getMessage());
                });
            }
        });
    }
}
