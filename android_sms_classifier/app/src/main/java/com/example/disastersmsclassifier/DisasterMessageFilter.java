package com.example.disastersmsclassifier;

public final class DisasterMessageFilter {
    private static final String[] DISASTER_KEYWORDS = {
            "재난", "긴급", "안전", "대피", "주의", "위험", "경보", "특보", "통제",
            "산사태", "지진", "호우", "태풍", "화재", "폭염", "한파", "침수", "홍수",
            "해일", "낙뢰", "강풍", "대설", "붕괴", "폭발", "누출", "실종", "우회",
            "접근금지", "접근 금지", "대피소", "신고", "행동요령"
    };

    private static final String[] ALLOWED_NOTIFICATION_PACKAGES = {
            "cellbroadcast",
            "messaging",
            "mms",
            "sms",
            "message"
    };

    private DisasterMessageFilter() {
    }

    public static boolean isDisasterRelatedText(String text) {
        if (text == null || text.trim().isEmpty()) {
            return false;
        }
        for (String keyword : DISASTER_KEYWORDS) {
            if (text.contains(keyword)) {
                return true;
            }
        }
        return false;
    }

    public static boolean isAllowedNotificationSource(String packageName) {
        String packageLower = packageName == null ? "" : packageName.toLowerCase();
        for (String allowedPackage : ALLOWED_NOTIFICATION_PACKAGES) {
            if (packageLower.contains(allowedPackage)) {
                return true;
            }
        }
        return false;
    }
}
