# TF-IDF + Logistic Regression Class Weight 구현 코드 문서

## 1. 모델 개요

이 문서는 TF-IDF + Logistic Regression 모델 중 **class_weight를 적용한 비교 실험 버전**의 구현 내용을 정리한다.

라벨 매핑:

- 일반 = 0
- 주의 = 1
- 긴급 = 2

## 2. 사용 코드 파일

class_weight 버전에서 사용하는 주요 파일은 다음과 같다.

- `src/train_tfidf_lr.py`
- `src/predict_tfidf.py`
- `src/config.py`
- `src/data_utils.py`
- `src/metrics.py`

## 3. 학습 흐름

```text
재난문자 본문
→ TF-IDF 벡터화
→ Logistic Regression에 class_weight 적용
→ Validation/Test 평가
```

## 4. 불균형 처리 방식

Train 기준 class weight를 Logistic Regression에 적용한다.

사용한 class weight:

```text
일반: 0.4040
주의: 2.2883
긴급: 11.3756
```

이 버전에서는 SMOTE를 사용하지 않는다.

## 5. 핵심 코드

class weight 설정:

```python
class_weight = TRAIN_CLASS_WEIGHTS
```

Logistic Regression 학습:

```python
LogisticRegression(
    C=2.0,
    class_weight=class_weight,
    max_iter=args.max_iter,
    solver="saga",
    random_state=42,
)
```

## 6. 실행 명령

```powershell
python -m src.train_tfidf_lr --data-dir data --output-dir outputs\tfidf_lr_class_weight --imbalance class_weight --max-iter 500
```

## 7. 저장 결과

```text
outputs/tfidf_lr_class_weight/model.joblib
outputs/tfidf_lr_class_weight/valid_metrics.json
outputs/tfidf_lr_class_weight/test_metrics.json
outputs/tfidf_lr_class_weight/valid_confusion_matrix.png
outputs/tfidf_lr_class_weight/test_confusion_matrix.png
outputs/tfidf_lr_class_weight/top_words.json
```

## 8. Test 결과

```text
Accuracy: 78.89%
Macro F1: 74.62%
Weighted F1: 81.79%
주의 Recall: 99.93%
긴급 Recall: 68.29%
```

