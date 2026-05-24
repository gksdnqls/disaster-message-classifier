# KLUE-BERT 구현 코드 문서

## 1. 모델 개요

이 문서는 KLUE-BERT 기반 재난문자 중요도 분류 모델의 구현 내용을 정리한다.

사용 모델:

```text
klue/bert-base
```

라벨 매핑:

- 일반 = 0
- 주의 = 1
- 긴급 = 2

## 2. 사용 코드 파일

KLUE-BERT에서 사용하는 주요 파일은 다음과 같다.

- `src/train_klue_bert.py`
- `src/predict_klue_bert.py`
- `src/config.py`
- `src/data_utils.py`
- `src/metrics.py`

## 3. 학습 흐름

```text
재난문자 본문
→ KLUE-BERT Tokenizer
→ klue/bert-base
→ Sequence Classification Head
→ Weighted Cross Entropy Loss
→ Validation/Test 평가
```

## 4. 불균형 처리 방식

KLUE-BERT에서는 SMOTE를 사용하지 않는다.

대신 Train 기준 class weight를 Weighted Cross Entropy Loss에 적용한다.

사용한 class weight:

```text
일반: 0.4040
주의: 2.2883
긴급: 11.3756
```

## 5. 핵심 코드

모델 로드:

```python
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=3,
    id2label=ID_TO_LABEL,
    label2id=LABEL_TO_ID,
)
```

Weighted Cross Entropy Loss:

```python
weight = torch.tensor(self.class_weights, dtype=logits.dtype, device=logits.device)
loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
```

class weight 적용:

```python
class_weights = [TRAIN_CLASS_WEIGHTS[idx] for idx in range(3)]
```

## 6. 실행 명령

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

## 7. 저장 결과

```text
outputs/klue_bert/best_model/
outputs/klue_bert/valid_metrics.json
outputs/klue_bert/test_metrics.json
outputs/klue_bert/valid_confusion_matrix.png
outputs/klue_bert/test_confusion_matrix.png
```

## 8. Test 결과

```text
Accuracy: 99.30%
Macro Precision: 98.24%
Macro F1: 98.37%
Weighted F1: 99.30%
주의 Recall: 97.42%
긴급 Recall: 98.45%
```

