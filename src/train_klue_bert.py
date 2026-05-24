from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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
    parser = argparse.ArgumentParser(description="Fine-tune KLUE-BERT for disaster message labels.")
    parser.add_argument("--data-dir", default="data", help="Directory containing train.csv, valid.csv, test.csv")
    parser.add_argument("--output-dir", default="outputs/klue_bert")
    parser.add_argument("--model-name", default="klue/bert-base")
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-class-weight", action="store_true", help="Use weighted cross entropy.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 when CUDA supports it.")
    parser.add_argument("--train-sample", type=int, default=0, help="Optional small sample for quick smoke tests.")
    return parser.parse_args()


def build_dataset(df, tokenizer, max_length: int) -> Dataset:
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
    if args.train_sample > 0:
        train_df = train_df.groupby(LABEL_COLUMN, group_keys=False).sample(
            n=min(args.train_sample, train_df[LABEL_COLUMN].value_counts().min()),
            random_state=args.seed,
        )

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

    class_weights = None
    if args.use_class_weight:
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
