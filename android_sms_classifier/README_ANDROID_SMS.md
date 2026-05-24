# Disaster SMS Classifier Android App

재난 관련 SMS 또는 재난 관련 알림 내용을 감지하여 Hugging Face Spaces에 배포된 KLUE-BERT 예측 서버로 전송하고, 일반 / 주의 / 긴급 분류 결과를 알림으로 표시하는 Android 앱입니다.

## 구조

```text
Android 앱
→ SMS 수신 또는 알림 감지
→ 재난 관련 키워드 포함 여부 확인
→ Hugging Face Spaces /predict 호출
→ KLUE-BERT 3중 처리 모델 예측
→ 일반 / 주의 / 긴급 결과 수신
→ 분류 결과 알림 표시
```

## 사용 모델

최종 모델은 KLUE-BERT 3중 처리 모델입니다.

```text
Random Oversampling
EDA 텍스트 증강
Weighted Loss
```

성능:

```text
Test Accuracy: 99.34%
Test 주의 Recall: 97.62%
Test 긴급 Recall: 98.67%
```

## 서버 주소

Android 앱에서 사용하는 예측 API 주소:

```text
https://vbcbxb-disaster-classifier.hf.space/predict
```

PC 로컬 서버, 같은 Wi-Fi, USB reverse 설정은 필요하지 않습니다.

## Android Studio에서 APK 생성

```text
Build
→ Generate App Bundles or APKs
→ Generate APKs
```

생성되는 APK:

```text
app/build/outputs/apk/debug/app-debug.apk
```

APK를 휴대폰에 복사한 뒤 설치합니다.

## 앱 초기 설정

앱 실행 후 서버 주소 칸에 아래 주소를 입력합니다.

```text
https://vbcbxb-disaster-classifier.hf.space/predict
```

그 다음 `서버 주소 저장`을 누릅니다.

## 권한 설정

자동 분류를 위해 다음 권한을 허용합니다.

```text
SMS 수신 권한
알림 표시 권한
알림 접근 권한
```

앱 안의 `알림 접근 권한 열기` 버튼을 누른 뒤 `DisasterSmsClassifier`를 허용합니다.

## 자동 분류 방식

앱은 두 방식으로 문자를 감지합니다.

```text
1. SMS_RECEIVED 이벤트 감지
2. 알림 접근 권한을 통한 문자/재난 알림 감지
```

감지된 텍스트 중 재난 관련 키워드가 포함된 경우에만 Hugging Face 서버로 전송하고, 서버는 일반 / 주의 / 긴급 중 하나로 분류합니다.

재난 관련 키워드 예시:

```text
재난, 긴급, 안전, 대피, 주의, 위험, 경보, 특보, 산사태, 지진, 호우, 태풍, 화재, 침수, 우회
```

## 중복 알림 방지

같은 문자가 SMS 수신과 알림 접근에서 동시에 감지될 수 있으므로 `ClassificationDeduplicator`를 사용하여 중복 알림을 방지합니다.

```text
같은 내용의 문자가 15초 안에 다시 감지되면 중복으로 판단하고 무시합니다.
```

## 수동 테스트

앱 화면에서 문장을 입력하고 `수동 분류 테스트` 버튼을 누르면 서버 연결과 모델 예측을 확인할 수 있습니다.

예시:

```text
산사태 위험으로 즉시 대피 바랍니다
```

예상 결과:

```text
긴급
```

## 주의사항

현재 구현은 발표 및 시연용 구조입니다.

문자 내용이 Hugging Face Spaces 서버로 전송되므로 실제 개인정보가 포함된 문자는 테스트하지 않는 것이 좋습니다.

실제 서비스로 확장하려면 서버 인증, 요청 제한, 민감정보 마스킹, 로그 비저장 정책이 필요합니다.
