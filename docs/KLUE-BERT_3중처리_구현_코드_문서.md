# KLUE-BERT 3중 불균형 처리 구현 코드 문서

## 1. 실험 목적

KLUE-BERT 기반 재난문자 중요도 분류 모델에 다음 세 가지 불균형 처리 방식을 동시에 적용한다.

- Random Oversampling
- EDA 텍스트 증강
- Weighted Cross Entropy Loss

Train 데이터의 클래스 불균형을 보완하고, Validation/Test 데이터는 원본 분포를 유지한 상태로 평가한다.

## 2. 코드 파일

```text
src/train_klue_bert_ros_eda_weighted.py
```

직접 예측용 파일:

```text
src/predict_klue_bert.py
```

공통 사용 파일:

```text
src/config.py
src/data_utils.py
src/metrics.py
```

## 3. Random Oversampling

Train 데이터에서 소수 클래스 샘플을 복제하여 클래스 수를 맞춘다.

```python
sampled_idx = rng.choice(label_df.index.to_numpy(), size=need, replace=True)
sampled_parts.append(label_df.loc[sampled_idx])
```

Validation/Test 데이터에는 적용하지 않는다.

## 4. EDA 텍스트 증강

주의/긴급 라벨에 대해 일부 단어를 유사 표현으로 교체하여 새로운 학습 문장을 생성한다.

예시 치환:

```text
즉시 → 바로, 신속히
대피 → 피난, 안전한 곳으로 이동
주의 → 유의, 조심
침수 → 물잠김, 침수 피해
산불 → 산림화재, 산불 발생
```

핵심 코드:

```python
new_text = eda_replace(str(row[TEXT_COLUMN]), rng, prob)
```

증강 적용 라벨:

```text
주의 = 1
긴급 = 2
```

사용 옵션:

```text
--eda-copies 1
--eda-prob 0.25
```

## 5. Weighted Cross Entropy Loss

Train 기준 class weight를 손실 함수에 적용한다.

```python
weight = torch.tensor(self.class_weights, dtype=logits.dtype, device=logits.device)
loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
```

사용 class weight:

```text
일반: 0.4040
주의: 2.2883
긴급: 11.3756
```

## 6. 학습 데이터 변화

3중 처리 적용 전후 Train 데이터 수는 다음과 같다.

```text
원래 Train:
일반 118,873
주의 20,988
긴급 4,222

EDA 후:
일반 118,873
주의 28,572
긴급 6,901

Random Oversampling 후:
일반 118,873
주의 118,873
긴급 118,873
```

위 정보는 결과 폴더의 `augmentation_info.json`에 저장된다.

## 7. 실행 명령

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

## 8. 저장 결과

```text
outputs/klue_bert_ros_eda_weighted/best_model/
outputs/klue_bert_ros_eda_weighted/augmentation_info.json
outputs/klue_bert_ros_eda_weighted/valid_metrics.json
outputs/klue_bert_ros_eda_weighted/test_metrics.json
outputs/klue_bert_ros_eda_weighted/valid_confusion_matrix.png
outputs/klue_bert_ros_eda_weighted/test_confusion_matrix.png
```

## 9. Test 결과

```text
Accuracy: 99.34%
Macro Precision: 98.44%
Macro F1: 98.54%
Weighted F1: 99.34%
주의 Recall: 97.62%
긴급 Recall: 98.67%
```

Confusion Matrix:

```text
실제 일반: 일반 25,389 / 주의 72 / 긴급 12
실제 주의: 일반 95 / 주의 4,390 / 긴급 12
실제 긴급: 일반 10 / 주의 2 / 긴급 893
```

## 10. 결과 비교

| Model | Accuracy | Macro F1 | Weighted F1 | 주의 Recall | 긴급 Recall |
|---|---:|---:|---:|---:|---:|
| KLUE-BERT Weighted Loss | 99.30% | 98.37% | 99.30% | 97.42% | 98.45% |
| KLUE-BERT ROS + EDA + Weighted Loss | 99.34% | 98.54% | 99.34% | 97.62% | 98.67% |

## 11. 요약

- SMOTE는 사용하지 않았다.
- 텍스트 모델에 맞게 Random Oversampling, EDA, Weighted Loss를 적용했다.
- Validation/Test 데이터는 원본 분포 그대로 평가했다.
- ROS + EDA + Weighted Loss 적용 모델에서 Accuracy, Macro F1, 주의 Recall, 긴급 Recall이 소폭 향상되었다.
