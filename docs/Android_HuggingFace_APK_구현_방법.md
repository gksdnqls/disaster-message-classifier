# Android + Hugging Face 기반 재난문자 자동 분류 구현 방법

## 1. 구현 방식

이 구현은 PC에서 예측 서버를 직접 실행하지 않고, Hugging Face Spaces에 KLUE-BERT 예측 서버를 배포한 뒤 Android 앱이 해당 서버를 호출하는 방식이다.

```text
휴대폰 앱
→ Hugging Face Spaces 예측 서버
→ KLUE-BERT 3중 처리 모델
→ 일반 / 주의 / 긴급 분류 결과 반환
```

## 2. 사용 모델

Android 앱이 호출하는 서버에는 최종 모델인 `KLUE-BERT 3중 처리 모델`을 사용하였다.

적용 방식:

```text
Random Oversampling
EDA 텍스트 증강
Weighted Loss
```

최종 성능:

```text
Test Accuracy: 99.34%
Test 주의 Recall: 97.62%
Test 긴급 Recall: 98.67%
```

## 3. Hugging Face 서버 구성

업로드용 폴더 위치:

```text
huggingface_space/
```

폴더 구성:

```text
huggingface_space
├─ app.py
├─ Dockerfile
├─ requirements.txt
└─ best_model
```

각 파일 역할:

```text
app.py
- FastAPI 서버 코드
- /predict API 제공
- 입력 문장을 KLUE-BERT 모델로 분류

Dockerfile
- Hugging Face Spaces에서 서버를 실행하기 위한 Docker 설정

requirements.txt
- 서버 실행에 필요한 Python 라이브러리 목록

best_model
- KLUE-BERT 3중 처리 학습 결과 모델
```

## 4. Hugging Face Space 주소

Space 주소:

```text
https://vbcbxb-disaster-classifier.hf.space/
```

Android 앱에서 사용하는 예측 API 주소:

```text
https://vbcbxb-disaster-classifier.hf.space/predict
```

브라우저에서 Space 주소를 열었을 때 아래와 같은 응답이 나오면 서버가 실행 중인 상태이다.

```json
{"status":"ok","message":"disaster classifier server"}
```

## 5. Android 앱 설치 방식

Android Studio에서 APK를 생성한 뒤 휴대폰에 직접 설치한다.

APK 생성 순서:

```text
Build
→ Generate App Bundles or APKs
→ Generate APKs
```

생성되는 APK 위치:

```text
android_sms_classifier/app/build/outputs/apk/debug/app-debug.apk
```

휴대폰 설치 방법:

```text
1. app-debug.apk를 휴대폰으로 복사
2. 휴대폰에서 APK 실행
3. 출처를 알 수 없는 앱 설치 허용
4. 앱 설치
```

## 6. 앱 초기 설정

앱 실행 후 서버 주소 칸에 아래 주소를 입력한다.

```text
https://vbcbxb-disaster-classifier.hf.space/predict
```

그 다음 `서버 주소 저장` 버튼을 누른다.

## 7. 권한 설정

앱에서 자동 분류를 사용하려면 다음 권한을 허용한다.

```text
SMS 수신 권한
알림 표시 권한
알림 접근 권한
```

알림 접근 권한 설정:

```text
앱 실행
→ 알림 접근 권한 열기
→ DisasterSmsClassifier 허용
```

## 8. 자동 분류 동작 방식

앱은 두 경로로 문자를 감지한다.

```text
1. SMS 수신 이벤트 감지
2. 알림 접근 권한을 통한 재난/문자 알림 감지
```

문자가 감지되면 앱은 먼저 재난 관련 키워드가 포함되어 있는지 확인한다.

재난 관련 키워드가 포함된 경우에만 해당 문장을 Hugging Face 서버의 `/predict` API로 전송한다.

재난 관련 키워드 예시:

```text
재난, 긴급, 안전, 대피, 주의, 위험, 경보, 특보, 산사태, 지진, 호우, 태풍, 화재, 침수, 우회
```

서버는 문장을 KLUE-BERT 모델로 분류한 뒤 다음 중 하나의 라벨을 반환한다.

```text
일반 = 0
주의 = 1
긴급 = 2
```

앱은 반환된 결과를 휴대폰 알림으로 표시한다.

예시:

```text
입력 문자:
산사태 위험으로 즉시 대피 바랍니다

출력 알림:
SMS 자동 분류: 긴급
```

## 9. 중복 알림 방지

SMS 수신 감지와 알림 감지가 같은 문자를 동시에 잡을 수 있으므로, 중복 제거 로직을 추가하였다.

추가 파일:

```text
ClassificationDeduplicator.java
```

동작 방식:

```text
같은 내용의 문자가 15초 안에 다시 감지되면 중복으로 판단하고 무시한다.
```

이를 통해 같은 문자에 대해 분류 알림이 두 번 뜨는 문제를 줄였다.

## 10. 보안 및 한계

현재 방식은 발표 및 시연용 구현이다.

장점:

```text
PC와 같은 Wi-Fi에 연결하지 않아도 실제 휴대폰에서 테스트 가능
HTTPS 기반 서버 통신 사용
Android 앱에 모델 파일을 직접 넣지 않아 앱 용량이 작음
```

주의할 점:

```text
문자 내용이 Hugging Face 서버로 전송됨
Public Space인 경우 서버 주소를 아는 사람이 API를 호출할 수 있음
실제 서비스로 확장하려면 인증, 요청 제한, 민감정보 마스킹, 로그 비저장 정책이 필요함
```

발표용 설명 문장:

```text
Android 앱은 수신된 문자 내용을 Hugging Face Spaces에 배포된 KLUE-BERT 예측 서버로 전송하고,
서버는 문장을 일반, 주의, 긴급 중 하나로 분류하여 결과를 앱에 반환한다.
현재 구현은 시연용 구조이며, 실제 서비스로 확장할 경우 개인정보 보호와 서버 접근 제어가 추가로 필요하다.
```

