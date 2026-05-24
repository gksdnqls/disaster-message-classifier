from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.config import ID_TO_LABEL, LABEL_COLUMN, LABEL_TO_ID, TEXT_COLUMN, TRAIN_CLASS_WEIGHTS
from src.data_utils import load_dataset
from src.metrics import save_evaluation


SYNONYM_MAP = {
    "즉시": ["바로", "신속히"],
    "대피": ["피난", "안전한 곳으로 이동"],
    "위험": ["위험성", "위험 상황"],
    "주의": ["유의", "조심"],
    "금지": ["자제", "삼가"],
    "통제": ["제한", "차단"],
    "우회": ["돌아가", "다른 길 이용"],
    "침수": ["물잠김", "침수 피해"],
    "화재": ["불", "화재 발생"],
    "산불": ["산림화재", "산불 발생"],
    "신고": ["신고해 주시기", "제보"],
    "발생": ["일어남", "발생함"],
    "안전": ["안전한 장소", "안전 확보"],
    "이동": ["이동 바랍니다", "이동해 주시기"],
}


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: list[float] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = None
        if self.class_weights is not None:
            weight = torch.tensor(self.class_weights, dtype=logits.dtype, device=logits.device)
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune KLUE-BERT with random oversampling, EDA augmentation, and weighted loss."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/klue_bert_ros_eda_weighted")
    parser.add_argument("--model-name", default="klue/bert-base")
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--ros-target",
        choices=["max", "none"],
        default="max",
        help="Random oversampling target. max means each class is matched to the largest class count.",
    )
    parser.add_argument(
        "--eda-labels",
        nargs="*",
        type=int,
        default=[1, 2],
        help="Labels to augment with EDA. Default: attention(1), emergency(2).",
    )
    parser.add_argument("--eda-copies", type=int, default=1, help="Number of EDA variants per selected sample.")
    parser.add_argument("--eda-prob", type=float, default=0.25, help="Replacement probability per synonym key.")
    parser.add_argument("--train-sample-per-class", type=int, default=0, help="Optional smoke-test sample per class.")
    return parser.parse_args()


def eda_replace(text: str, rng: random.Random, prob: float) -> str:
    augmented = text
    keys = list(SYNONYM_MAP.keys())
    rng.shuffle(keys)
    changed = False
    for key in keys:
        if key in augmented and rng.random() < prob:
            augmented = augmented.replace(key, rng.choice(SYNONYM_MAP[key]), 1)
            changed = True
    return augmented if changed else text


def apply_eda(train_df: pd.DataFrame, labels: list[int], copies: int, prob: float, seed: int) -> pd.DataFrame:
    if copies <= 0:
        return train_df

    rng = random.Random(seed)
    augmented_rows = []
    source_df = train_df[train_df[LABEL_COLUMN].isin(labels)]
    for _, row in source_df.iterrows():
        for _ in range(copies):
            new_text = eda_replace(str(row[TEXT_COLUMN]), rng, prob)
            if new_text != row[TEXT_COLUMN]:
                augmented_rows.append({TEXT_COLUMN: new_text, LABEL_COLUMN: int(row[LABEL_COLUMN])})

    if not augmented_rows:
        return train_df
    return pd.concat([train_df, pd.DataFrame(augmented_rows)], ignore_index=True)


def random_oversample(train_df: pd.DataFrame, target: str, seed: int) -> pd.DataFrame:
    if target == "none":
        return train_df

    rng = np.random.default_rng(seed)
    counts = train_df[LABEL_COLUMN].value_counts()
    target_count = int(counts.max())
    sampled_parts = [train_df]

    for label_id, count in counts.items():
        need = target_count - int(count)
        if need <= 0:
            continue
        label_df = train_df[train_df[LABEL_COLUMN] == label_id]
        sampled_idx = rng.choice(label_df.index.to_numpy(), size=need, replace=True)
        sampled_parts.append(label_df.loc[sampled_idx])

    return pd.concat(sampled_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_pandas(df[[TEXT_COLUMN, LABEL_COLUMN]], preserve_index=False)
    dataset = dataset.rename_column(LABEL_COLUMN, "labels")

    def tokenize(batch):
        return tokenizer(batch[TEXT_COLUMN], truncation=True, max_length=max_length)

    dataset = dataset.map(tokenize, batched=True)
    return dataset.remove_columns([TEXT_COLUMN])


def compute_hf_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "weighted_precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "recall_attention": recall_score(labels, preds, labels=[1], average="macro", zero_division=0),
        "recall_emergency": recall_score(labels, preds, labels=[2], average="macro", zero_division=0),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, test_df = load_dataset(args.data_dir)
    if args.train_sample_per_class > 0:
        train_df = train_df.groupby(LABEL_COLUMN, group_keys=False).sample(
            n=args.train_sample_per_class,
            replace=True,
            random_state=args.seed,
        )

    before_counts = train_df[LABEL_COLUMN].value_counts().sort_index().to_dict()
    train_df = apply_eda(train_df, args.eda_labels, args.eda_copies, args.eda_prob, args.seed)
    after_eda_counts = train_df[LABEL_COLUMN].value_counts().sort_index().to_dict()
    train_df = random_oversample(train_df, args.ros_target, args.seed)
    after_ros_counts = train_df[LABEL_COLUMN].value_counts().sort_index().to_dict()

    augmentation_info = {
        "before_counts": {str(k): int(v) for k, v in before_counts.items()},
        "after_eda_counts": {str(k): int(v) for k, v in after_eda_counts.items()},
        "after_random_oversampling_counts": {str(k): int(v) for k, v in after_ros_counts.items()},
        "eda_labels": args.eda_labels,
        "eda_copies": args.eda_copies,
        "eda_prob": args.eda_prob,
        "ros_target": args.ros_target,
    }
    with (output_dir / "augmentation_info.json").open("w", encoding="utf-8-sig") as f:
        json.dump(augmentation_info, f, ensure_ascii=False, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = build_dataset(train_df, tokenizer, args.max_length)
    valid_dataset = build_dataset(valid_df, tokenizer, args.max_length)
    test_dataset = build_dataset(test_df, tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=args.fp16,
        seed=args.seed,
    )

    class_weights = [TRAIN_CLASS_WEIGHTS[idx] for idx in range(3)]
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_hf_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir / "best_model")
    tokenizer.save_pretrained(output_dir / "best_model")

    for split_name, dataset, original_df in [
        ("valid", valid_dataset, valid_df),
        ("test", test_dataset, test_df),
    ]:
        prediction = trainer.predict(dataset)
        preds = np.argmax(prediction.predictions, axis=-1)
        result = save_evaluation(original_df[LABEL_COLUMN].to_numpy(), preds, output_dir, split_name)
        print(f"[{split_name}] {json.dumps(result['metrics'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
