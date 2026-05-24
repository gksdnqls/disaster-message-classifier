# KLUE-BERT Random Oversampling + EDA + Weighted Loss

이 코드는 KLUE-BERT에 다음 세 가지 불균형 처리 방식을 동시에 적용하는 실험용 학습 코드입니다.

1. Random Oversampling
2. EDA 텍스트 증강
3. Weighted Cross Entropy Loss

기존 KLUE-BERT 기본 실험은 class weight 기반 weighted loss만 적용했습니다. 이 실험은 세 가지 방식을 함께 적용한 결과와 기존 결과를 비교하기 위해 분리했습니다.

## 실행 파일

```text
src/train_klue_bert_ros_eda_weighted.py
```

## 처리 흐름

```text
Train 데이터
→ 소수 클래스 EDA 텍스트 증강
→ Random Oversampling으로 클래스 수 맞춤
→ KLUE-BERT 학습
→ Weighted Cross Entropy Loss 적용
→ Validation/Test 원본 분포로 평가
```

Validation/Test 데이터에는 EDA나 오버샘플링을 적용하지 않습니다.

## 실행 명령

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

## 주요 옵션

- `--eda-labels 1 2`: 주의/긴급 라벨에 EDA 적용
- `--eda-copies 1`: 선택된 샘플당 EDA 문장 1개 생성
- `--eda-prob 0.25`: 치환 후보 단어의 교체 확률
- `--ros-target max`: 각 클래스 수를 가장 많은 클래스 수에 맞춤

## 저장 결과

```text
outputs/klue_bert_ros_eda_weighted/best_model/
outputs/klue_bert_ros_eda_weighted/augmentation_info.json
outputs/klue_bert_ros_eda_weighted/valid_metrics.json
outputs/klue_bert_ros_eda_weighted/test_metrics.json
outputs/klue_bert_ros_eda_weighted/valid_confusion_matrix.png
outputs/klue_bert_ros_eda_weighted/test_confusion_matrix.png
```

## 비교 대상

기존 KLUE-BERT 결과:

```text
outputs/klue_bert/test_metrics.json
```

새 실험 결과:

```text
outputs/klue_bert_ros_eda_weighted/test_metrics.json
```
