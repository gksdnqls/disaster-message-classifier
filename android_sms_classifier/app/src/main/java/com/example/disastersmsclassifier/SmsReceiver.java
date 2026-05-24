package com.example.disastersmsclassifier;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.provider.Telephony;
import android.telephony.SmsMessage;

public class SmsReceiver extends BroadcastReceiver {
    @Override
    public void onReceive(Context context, Intent intent) {
        if (!Telephony.Sms.Intents.SMS_RECEIVED_ACTION.equals(intent.getAction())) {
            return;
        }

        SmsMessage[] messages = Telephony.Sms.Intents.getMessagesFromIntent(intent);
        StringBuilder body = new StringBuilder();
        for (SmsMessage message : messages) {
            body.append(message.getMessageBody());
        }

        String text = body.toString();
        if (!DisasterMessageFilter.isDisasterRelatedText(text)) {
            return;
        }
        if (ClassificationDeduplicator.shouldSkip(context, text)) {
            return;
        }

        String endpoint = ServerConfig.getPredictUrl(context);
        PendingResult pendingResult = goAsync();

        PredictionClient.predictAsync(endpoint, text, new PredictionClient.Callback() {
            @Override
            public void onSuccess(String label, int labelId, String rawJson) {
                NotificationHelper.showClassificationResult(context, "SMS 자동 분류: " + label, text);
                pendingResult.finish();
            }

            @Override
            public void onError(Exception exception) {
                NotificationHelper.showClassificationResult(context, "SMS 자동 분류 실패", exception.getMessage());
                pendingResult.finish();
            }
        });
    }
}
