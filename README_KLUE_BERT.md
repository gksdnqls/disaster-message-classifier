# KLUE-BERT Disaster Message Classification

KLUE-BERT 기반 재난문자 중요도 분류 모델 학습 및 평가 코드입니다.

## Label Mapping

- 일반 = 0
- 주의 = 1
- 긴급 = 2

## Model

- Base model: `klue/bert-base`
- Input column: `메시지내용`
- Label column: `label_id`
- Imbalance handling: Weighted Cross Entropy Loss

Train 기준 class weight:

- 일반: `0.4040`
- 주의: `2.2883`
- 긴급: `11.3756`

## Data

분할된 데이터 파일을 사용합니다.

- Train: 70%
- Validation: 15%
- Test: 15%
- Stratified Split 적용

Validation/Test 데이터는 원본 분포를 유지합니다.

## Install

```powershell
pip install -r requirements.txt
```

## Prepare Data

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\prepare_data.ps1
```

## Train

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

## Prediction

```powershell
python -m src.predict_klue_bert --model-dir outputs\klue_bert\best_model --text "산사태 위험으로 즉시 대피 바랍니다"
```

## Outputs

학습 및 평가 후 다음 파일이 생성됩니다.

- `outputs/klue_bert/best_model/`
- `outputs/klue_bert/valid_metrics.json`
- `outputs/klue_bert/test_metrics.json`
- `outputs/klue_bert/valid_confusion_matrix.png`
- `outputs/klue_bert/test_confusion_matrix.png`

## Metrics

- Accuracy
- Macro Precision
- Weighted Precision
- Macro F1
- Weighted F1
- 주의 Recall
- 긴급 Recall
- Confusion Matrix
