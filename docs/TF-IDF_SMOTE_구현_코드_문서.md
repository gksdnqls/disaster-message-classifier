# TF-IDF + Logistic Regression SMOTE 구현 코드 문서

## 1. 모델 개요

이 문서는 TF-IDF + Logistic Regression 모델 중 **SMOTE 오버샘플링을 적용한 버전**의 구현 내용을 정리한다.

라벨 매핑:

- 일반 = 0
- 주의 = 1
- 긴급 = 2

## 2. 사용 코드 파일

SMOTE 버전에서 사용하는 주요 파일은 다음과 같다.

- `src/train_tfidf_lr.py`
- `src/predict_tfidf.py`
- `src/data_utils.py`
- `src/metrics.py`

주의:

- `src/config.py`에는 전체 프로젝트 공통 라벨 매핑과 class weight 값이 저장되어 있다.
- 그러나 **SMOTE 버전 학습에서는 class weight를 사용하지 않는다.**
- SMOTE 버전의 Logistic Regression은 `class_weight=None`으로 학습한다.

## 3. 학습 흐름

```text
재난문자 본문
→ TF-IDF 벡터화
→ Train 데이터에만 SMOTE 적용
→ Logistic Regression 학습
→ Validation/Test 평가
```

## 4. 불균형 처리 방식

SMOTE를 사용한다.

적용 위치:

```text
Train 데이터에만 적용
```

적용 시점:

```text
TF-IDF 벡터화 이후 적용
```

Validation/Test 데이터에는 SMOTE를 적용하지 않는다.

## 5. 핵심 코드

TF-IDF 벡터화:

```python
vectorizer = TfidfVectorizer(
    max_features=args.max_features,
    ngram_range=(args.ngram_min, args.ngram_max),
    min_df=args.min_df,
    sublinear_tf=True,
)
x_train_vec = vectorizer.fit_transform(x_train)
```

SMOTE 적용:

```python
smote = SMOTE(random_state=42, k_neighbors=5)
x_resampled, y_resampled = smote.fit_resample(x_train_vec, y_train)
```

Logistic Regression 학습:

```python
clf = LogisticRegression(
    C=2.0,
    class_weight=None,
    max_iter=args.max_iter,
    solver="saga",
    random_state=42,
)
clf.fit(x_resampled, y_resampled)
```

## 6. 실행 명령

```powershell
python -m src.train_tfidf_lr --data-dir data --output-dir outputs\tfidf_lr_smote --imbalance smote --max-iter 500
```

## 7. 저장 결과

```text
outputs/tfidf_lr_smote/model.joblib
outputs/tfidf_lr_smote/valid_metrics.json
outputs/tfidf_lr_smote/test_metrics.json
outputs/tfidf_lr_smote/valid_confusion_matrix.png
outputs/tfidf_lr_smote/test_confusion_matrix.png
outputs/tfidf_lr_smote/top_words.json
```

## 8. Test 결과

```text
Accuracy: 97.61%
Macro F1: 95.11%
Weighted F1: 97.64%
주의 Recall: 95.95%
긴급 Recall: 96.91%
```

