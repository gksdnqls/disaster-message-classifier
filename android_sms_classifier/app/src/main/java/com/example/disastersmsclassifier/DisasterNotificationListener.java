package com.example.disastersmsclassifier;

import android.app.Notification;
import android.service.notification.NotificationListenerService;
import android.service.notification.StatusBarNotification;

public class DisasterNotificationListener extends NotificationListenerService {
    @Override
    public void onNotificationPosted(StatusBarNotification sbn) {
        if (sbn == null || getPackageName().equals(sbn.getPackageName())) {
            return;
        }

        Notification notification = sbn.getNotification();
        if (notification == null || notification.extras == null) {
            return;
        }

        String title = charSequenceToString(notification.extras.getCharSequence(Notification.EXTRA_TITLE));
        String text = charSequenceToString(notification.extras.getCharSequence(Notification.EXTRA_TEXT));
        String bigText = charSequenceToString(notification.extras.getCharSequence(Notification.EXTRA_BIG_TEXT));
        String message = joinText(title, text, bigText);

        if (message.isEmpty()
                || !DisasterMessageFilter.isAllowedNotificationSource(sbn.getPackageName())
                || !DisasterMessageFilter.isDisasterRelatedText(message)) {
            return;
        }
        if (ClassificationDeduplicator.shouldSkip(this, message)) {
            return;
        }

        String endpoint = ServerConfig.getPredictUrl(this);
        PredictionClient.predictAsync(endpoint, message, new PredictionClient.Callback() {
            @Override
            public void onSuccess(String label, int labelId, String rawJson) {
                NotificationHelper.showClassificationResult(
                        DisasterNotificationListener.this,
                        "재난 알림 자동 분류: " + label,
                        message
                );
            }

            @Override
            public void onError(Exception exception) {
                NotificationHelper.showClassificationResult(
                        DisasterNotificationListener.this,
                        "재난 알림 자동 분류 실패",
                        exception.getMessage()
                );
            }
        });
    }

    private static String charSequenceToString(CharSequence value) {
        return value == null ? "" : value.toString().trim();
    }

    private static String joinText(String title, String text, String bigText) {
        StringBuilder builder = new StringBuilder();
        appendIfPresent(builder, title);
        appendIfPresent(builder, text);
        appendIfPresent(builder, bigText);
        return builder.toString().trim();
    }

    private static void appendIfPresent(StringBuilder builder, String value) {
        if (value == null || value.trim().isEmpty()) {
            return;
        }
        if (builder.length() > 0) {
            builder.append('\n');
        }
        builder.append(value.trim());
    }

}
