# Android + Hugging Face 서버 코드 정리

## 1. 전체 구현 구조

최신 구현은 PC에서 로컬 서버를 실행하지 않고, Hugging Face Spaces에 배포된 KLUE-BERT 예측 서버를 Android 앱이 호출하는 방식이다.

```text
Android 앱
→ SMS 또는 알림 감지
→ 재난 관련 키워드 필터링
→ Hugging Face Spaces /predict API 호출
→ KLUE-BERT 3중 처리 모델 예측
→ 일반 / 주의 / 긴급 결과 반환
→ Android 알림 표시
```

Android 앱에서 사용하는 서버 주소:

```text
https://vbcbxb-disaster-classifier.hf.space/predict
```

## 2. Hugging Face 서버 코드

서버 코드 위치:

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

### 2.1 `app.py`

FastAPI 기반 예측 서버 코드이다.

역할:

```text
1. best_model 폴더에서 KLUE-BERT 모델과 tokenizer 로드
2. /predict API 제공
3. Android 앱에서 받은 문장을 모델에 입력
4. 예측 결과를 일반 / 주의 / 긴급 라벨로 변환
5. JSON 형식으로 결과 반환
```

주요 코드 흐름:

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
```

```python
@app.post("/predict")
def predict(request: PredictRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
```

```python
with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)[0]
    label_id = int(torch.argmax(probabilities).item())
```

반환 형식:

```json
{
  "text": "산사태 위험으로 즉시 대피 바랍니다",
  "label_id": 2,
  "label": "긴급",
  "confidence": 0.99,
  "probabilities": {
    "일반": 0.01,
    "주의": 0.00,
    "긴급": 0.99
  }
}
```

### 2.2 `Dockerfile`

Hugging Face Spaces에서 FastAPI 서버를 실행하기 위한 Docker 설정 파일이다.

역할:

```text
1. Python 3.10 환경 생성
2. requirements.txt의 라이브러리 설치
3. 서버 파일 복사
4. uvicorn으로 app.py 실행
```

실행 포트:

```text
7860
```

Hugging Face Spaces는 내부적으로 7860 포트를 사용하므로 Dockerfile에서 해당 포트로 서버를 실행한다.

### 2.3 `requirements.txt`

서버 실행에 필요한 Python 라이브러리 목록이다.

포함 라이브러리:

```text
fastapi
uvicorn[standard]
torch
transformers
safetensors
pydantic
```

### 2.4 `best_model`

KLUE-BERT 3중 처리 학습 결과 모델 폴더이다.

역할:

```text
1. tokenizer 정보 저장
2. 모델 config 저장
3. 학습된 model.safetensors 저장
4. Hugging Face 서버에서 실제 예측에 사용
```

이 폴더는 용량이 크기 때문에 GitHub 제출용 코드에는 포함하지 않고, Hugging Face Space에는 직접 업로드하였다.

## 3. Android 앱 코드

Android Studio 프로젝트 위치:

```text
android_sms_classifier/
```

제출용 Android 앱 폴더:

```text
mobile_demo/Android_SMS_자동분류앱/
```

주요 Java 파일:

```text
MainActivity.java
ServerConfig.java
PredictionClient.java
SmsReceiver.java
DisasterNotificationListener.java
DisasterMessageFilter.java
ClassificationDeduplicator.java
NotificationHelper.java
```

### 3.1 `MainActivity.java`

앱의 메인 화면을 담당한다.

역할:

```text
1. 서버 주소 입력 및 저장
2. 수동 분류 테스트 실행
3. SMS 권한과 알림 권한 요청
4. 알림 접근 권한 설정 화면 열기
```

앱에서 기본 서버 주소는 Hugging Face 예측 API로 설정되어 있다.

```text
https://vbcbxb-disaster-classifier.hf.space/predict
```

### 3.2 `ServerConfig.java`

예측 서버 주소를 저장하고 불러오는 설정 파일이다.

역할:

```text
1. 기본 서버 주소 저장
2. 사용자가 앱에서 입력한 서버 주소 저장
3. SMS/알림 자동 분류 시 저장된 서버 주소 제공
```

기본 주소:

```java
private static final String DEFAULT_URL = "https://vbcbxb-disaster-classifier.hf.space/predict";
```

### 3.3 `PredictionClient.java`

Android 앱에서 Hugging Face 서버의 `/predict` API를 호출하는 코드이다.

역할:

```text
1. 감지된 문자 내용을 JSON으로 변환
2. 서버에 POST 요청 전송
3. 서버 응답 JSON 파싱
4. label, label_id를 앱으로 전달
```

요청 형식:

```json
{
  "text": "산사태 위험으로 즉시 대피 바랍니다"
}
```

### 3.4 `SmsReceiver.java`

일반 SMS 수신 이벤트를 감지하는 BroadcastReceiver이다.

역할:

```text
1. SMS_RECEIVED 이벤트 감지
2. 수신된 SMS 본문 추출
3. 재난 관련 키워드 포함 여부 확인
4. 중복 문자 여부 확인
5. 서버에 예측 요청
6. 분류 결과 알림 표시
```

재난 관련 키워드가 없는 일반 문자는 서버로 보내지 않는다.

### 3.5 `DisasterNotificationListener.java`

휴대폰 알림 내용을 감지하는 NotificationListenerService이다.

역할:

```text
1. 알림 접근 권한을 통해 새 알림 감지
2. 알림 제목과 본문 추출
3. 문자/재난 알림으로 보이는 출처인지 확인
4. 재난 관련 키워드 포함 여부 확인
5. 중복 알림 여부 확인
6. 서버에 예측 요청
7. 분류 결과 알림 표시
```

이 기능은 실제 긴급재난문자가 SMS가 아니라 시스템 알림 형태로 표시될 수 있다는 점을 고려하여 추가하였다.

### 3.6 `DisasterMessageFilter.java`

재난 관련 텍스트만 분류하도록 필터링하는 코드이다.

역할:

```text
1. 재난 관련 키워드 포함 여부 확인
2. 허용할 알림 출처 패키지 확인
3. 일반 알림이나 일반 문자가 서버로 전송되는 것을 방지
```

재난 관련 키워드 예시:

```text
재난, 긴급, 안전, 대피, 주의, 위험, 경보, 특보,
산사태, 지진, 호우, 태풍, 화재, 침수, 우회
```

### 3.7 `ClassificationDeduplicator.java`

같은 문자가 두 번 분류되는 문제를 방지하는 코드이다.

역할:

```text
1. SMS 수신 감지와 알림 감지가 같은 문자를 동시에 잡는 경우 방지
2. 최근 분류된 문자 내용과 시간을 저장
3. 같은 내용이 15초 안에 다시 들어오면 중복으로 판단하고 무시
```

이를 통해 같은 문자에 대해 `SMS 자동 분류`와 `재난 알림 자동 분류`가 동시에 뜨는 문제를 줄였다.

### 3.8 `NotificationHelper.java`

분류 결과를 Android 알림으로 표시하는 코드이다.

역할:

```text
1. 알림 채널 생성
2. 분류 결과 알림 표시
3. 알림 클릭 시 앱 화면으로 이동
```

예시 알림:

```text
SMS 자동 분류: 긴급
산사태 위험으로 즉시 대피 바랍니다
```

## 4. Android 권한

`AndroidManifest.xml`에 포함된 주요 권한:

```text
INTERNET
RECEIVE_SMS
READ_SMS
POST_NOTIFICATIONS
BIND_NOTIFICATION_LISTENER_SERVICE
```

권한 역할:

```text
INTERNET
- Hugging Face 서버 호출

RECEIVE_SMS
- SMS 수신 이벤트 감지

READ_SMS
- SMS 본문 처리 보조

POST_NOTIFICATIONS
- 분류 결과 알림 표시

BIND_NOTIFICATION_LISTENER_SERVICE
- 알림 접근 권한을 통한 알림 감지
```

## 5. APK 생성 및 설치

Android Studio에서 APK 생성:

```text
Build
→ Generate App Bundles or APKs
→ Generate APKs
```

생성 위치:

```text
android_sms_classifier/app/build/outputs/apk/debug/app-debug.apk
```

휴대폰에 APK를 복사한 뒤 설치한다.

## 6. 앱 실행 후 설정

앱 실행 후 서버 주소 칸에 아래 주소가 들어 있는지 확인한다.

```text
https://vbcbxb-disaster-classifier.hf.space/predict
```

그 다음 권한을 허용한다.

```text
SMS 권한
알림 표시 권한
알림 접근 권한
```

## 7. 최종 동작 예시

입력 문자:

```text
산사태 위험으로 즉시 대피 바랍니다
```

처리 과정:

```text
SMS 수신
→ 재난 관련 키워드 확인
→ Hugging Face 서버 전송
→ KLUE-BERT 모델 예측
→ 긴급 라벨 반환
→ Android 알림 표시
```

출력:

```text
SMS 자동 분류: 긴급
```

## 8. 보안 및 한계

현재 구현은 발표 및 시연용이다.

주의사항:

```text
문자 내용이 Hugging Face Spaces 서버로 전송된다.
실제 개인정보가 포함된 문자는 테스트하지 않는 것이 좋다.
Public Space인 경우 서버 주소를 아는 사람이 API를 호출할 수 있다.
```

실제 서비스로 확장할 경우 필요한 보완:

```text
서버 인증
요청 제한
민감정보 마스킹
로그 비저장 정책
HTTPS 유지
모델 서버 접근 제어
```

