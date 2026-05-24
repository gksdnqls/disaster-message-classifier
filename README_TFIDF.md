# Disaster Message Importance Classification

재난문자 중요도 라벨을 분류하기 위한 TF-IDF + Logistic Regression 학습 및 평가 코드입니다.

## Label Mapping

- 일반 = 0
- 주의 = 1
- 긴급 = 2

## Data

분할된 데이터 파일을 사용합니다.

| Split | Ratio | 일반 | 주의 | 긴급 |
|---|---:|---:|---:|---:|
| Train | 70% | 118,873 | 20,988 | 4,222 |
| Validation | 15% | 25,473 | 4,498 | 904 |
| Test | 15% | 25,473 | 4,497 | 905 |

사용 컬럼:

- `메시지내용`: 재난문자 본문
- `중요도라벨`: 원본 라벨명
- `label_id`: 숫자 라벨

## Install

```powershell
pip install -r requirements.txt
```

## Prepare Data

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\prepare_data.ps1
```

## TF-IDF + Logistic Regression

이 코드는 하나의 학습 파일에서 두 가지 불균형 처리 방식을 선택해서 실행할 수 있습니다.

- SMOTE 적용 버전
- class_weight 적용 버전

두 방식 모두 Validation/Test 데이터에는 오버샘플링을 적용하지 않고, 원본 분포 그대로 평가합니다.

## Method 1. SMOTE 적용

TF-IDF 벡터화 후 Train 데이터에만 SMOTE를 적용하고 Logistic Regression을 학습합니다.

```powershell
python -m src.train_tfidf_lr --data-dir data --output-dir outputs\tfidf_lr_smote --imbalance smote --max-iter 500
```

생성 결과:

- `outputs/tfidf_lr_smote/model.joblib`
- `outputs/tfidf_lr_smote/valid_metrics.json`
- `outputs/tfidf_lr_smote/test_metrics.json`
- `outputs/tfidf_lr_smote/valid_confusion_matrix.png`
- `outputs/tfidf_lr_smote/test_confusion_matrix.png`
- `outputs/tfidf_lr_smote/top_words.json`

## Method 2. class_weight 적용

Train 기준 class weight를 Logistic Regression에 적용합니다. 이 방식에서는 SMOTE를 사용하지 않습니다.

Train 기준 class weight:

- 일반: 0.4040
- 주의: 2.2883
- 긴급: 11.3756

```powershell
python -m src.train_tfidf_lr --data-dir data --output-dir outputs\tfidf_lr_class_weight --imbalance class_weight --max-iter 500
```

생성 결과:

- `outputs/tfidf_lr_class_weight/model.joblib`
- `outputs/tfidf_lr_class_weight/valid_metrics.json`
- `outputs/tfidf_lr_class_weight/test_metrics.json`
- `outputs/tfidf_lr_class_weight/valid_confusion_matrix.png`
- `outputs/tfidf_lr_class_weight/test_confusion_matrix.png`
- `outputs/tfidf_lr_class_weight/top_words.json`

## Prediction

SMOTE 모델 예측:

```powershell
python -m src.predict_tfidf --model-path outputs\tfidf_lr_smote\model.joblib --text "산사태 위험으로 즉시 대피 바랍니다"
```

class_weight 모델 예측:

```powershell
python -m src.predict_tfidf --model-path outputs\tfidf_lr_class_weight\model.joblib --text "산사태 위험으로 즉시 대피 바랍니다"
```

## Metrics

- Accuracy
- Macro Precision
- Weighted Precision
- Macro F1
- Weighted F1
- 주의 Recall
- 긴급 Recall
- Confusion Matrix
