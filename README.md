# Disaster Message Importance Classification

재난문자 본문을 `일반`, `주의`, `긴급` 3개 등급으로 분류하는 프로젝트입니다.

본 저장소에는 모델 학습 코드, 예측 코드, Hugging Face 서버 코드, Android 앱 코드가 포함됩니다. 데이터 파일, 학습된 모델 파일, 체크포인트, 결과 zip은 용량 문제로 포함하지 않습니다.

## 라벨 매핑

```text
일반 = 0
주의 = 1
긴급 = 2
```

## 구현 모델

### 1. TF-IDF + Logistic Regression

빠른 기준 성능 확인을 위한 베이스라인 모델입니다.

구현 파일:

```text
src/train_tfidf_lr.py
src/predict_tfidf.py
```

비교한 불균형 처리 방식:

```text
SMOTE
class_weight
```

### 2. KLUE-BERT 기본 비교 모델

KLUE-BERT 기반 문맥 분류 성능을 확인하기 위한 비교 모델입니다.

구현 파일:

```text
src/train_klue_bert.py
src/predict_klue_bert.py
```

불균형 처리:

```text
Weighted Cross Entropy Loss
```

### 3. KLUE-BERT 최종 모델

최종 사용 모델은 KLUE-BERT에 세 가지 불균형 처리 방식을 함께 적용한 모델입니다.

구현 파일:

```text
src/train_klue_bert_ros_eda_weighted.py
src/predict_klue_bert.py
```

최종 불균형 처리 방식:

```text
Random Oversampling
EDA 텍스트 증강
Weighted Loss
```

최종 모델 성능:

```text
Test Accuracy: 99.34%
Test 주의 Recall: 97.62%
Test 긴급 Recall: 98.67%
```

### 4. Android + Hugging Face 자동 분류

Android 앱이 SMS 또는 재난 관련 알림을 감지한 뒤 Hugging Face Spaces 서버로 문장을 보내고, KLUE-BERT 최종 모델의 예측 결과를 알림으로 표시합니다.

구현 파일:

```text
android_sms_classifier/
huggingface_space/
```

서버 주소:

```text
https://vbcbxb-disaster-classifier.hf.space/predict
```

## 폴더 구조

```text
src/
  train_tfidf_lr.py
  predict_tfidf.py
  train_klue_bert.py
  train_klue_bert_ros_eda_weighted.py
  predict_klue_bert.py
  config.py
  data_utils.py
  metrics.py

scripts/
  prepare_data.ps1

android_sms_classifier/
  Android Studio 앱 프로젝트

huggingface_space/
  Hugging Face Spaces 배포용 FastAPI 서버 코드

docs/
  구현 설명 문서
  모델 비교 문서
```

## 데이터 준비

데이터셋은 용량 문제로 GitHub 저장소에 포함하지 않았습니다.
코드를 실행하려면 아래 파일을 `data/` 폴더에 배치해야 합니다.

```text
data/train.csv
data/valid.csv
data/test.csv
data/dataset_meta.json
```

데이터 분할:

```text
Train: 70%
Validation: 15%
Test: 15%
Stratified Split
```

## 설치

```powershell
pip install -r requirements.txt
```

## TF-IDF + Logistic Regression 실행

### SMOTE 적용

```powershell
python -m src.train_tfidf_lr --data-dir data --output-dir outputs\tfidf_lr_smote --imbalance smote --max-iter 500
```

### class_weight 적용

```powershell
python -m src.train_tfidf_lr --data-dir data --output-dir outputs\tfidf_lr_class_weight --imbalance class_weight --max-iter 500
```

## KLUE-BERT 실행

### Weighted Loss 비교 모델

```powershell
python -m src.train_klue_bert ^
  --data-dir data ^
  --output-dir outputs\klue_bert ^
  --model-name klue/bert-base ^
  --use-class-weight ^
  --epochs 3 ^
  --batch-size 8 ^
  --eval-batch-size 16 ^
  --fp16
```

### 최종 모델: Random Oversampling + EDA + Weighted Loss

```powershell
python -m src.train_klue_bert_ros_eda_weighted ^
  --data-dir data ^
  --output-dir outputs\klue_bert_ros_eda_weighted ^
  --model-name klue/bert-base ^
  --epochs 3 ^
  --batch-size 8 ^
  --eval-batch-size 16 ^
  --eda-copies 1 ^
  --eda-prob 0.25 ^
  --ros-target max ^
  --fp16
```

## 평가 지표

```text
Accuracy
Macro Precision
Weighted Precision
Macro F1
Weighted F1
주의 Recall
긴급 Recall
Confusion Matrix
```

## Android 앱

Android 앱은 다음 기능을 포함합니다.

```text
SMS 수신 감지
알림 접근 권한을 통한 문자/재난 알림 감지
재난 관련 키워드 필터링
중복 알림 방지
Hugging Face /predict API 호출
분류 결과 알림 표시
```

주요 파일:

```text
android_sms_classifier/app/src/main/java/com/example/disastersmsclassifier/MainActivity.java
android_sms_classifier/app/src/main/java/com/example/disastersmsclassifier/SmsReceiver.java
android_sms_classifier/app/src/main/java/com/example/disastersmsclassifier/DisasterNotificationListener.java
android_sms_classifier/app/src/main/java/com/example/disastersmsclassifier/DisasterMessageFilter.java
android_sms_classifier/app/src/main/java/com/example/disastersmsclassifier/ClassificationDeduplicator.java
android_sms_classifier/app/src/main/java/com/example/disastersmsclassifier/PredictionClient.java
```

자세한 설명:

```text
android_sms_classifier/README_ANDROID_SMS.md
docs/Android_HuggingFace_APK_구현_방법.md
docs/Android_HuggingFace_서버_앱_코드_정리.md
```

## Hugging Face 서버

Hugging Face 서버는 FastAPI로 구현했습니다.

주요 파일:

```text
huggingface_space/app.py
huggingface_space/Dockerfile
huggingface_space/requirements.txt
```

`best_model/`은 KLUE-BERT 최종 모델 파일이 들어가는 폴더입니다. 모델 파일은 용량 문제로 GitHub에는 포함하지 않고, Hugging Face Space에 직접 업로드합니다.
