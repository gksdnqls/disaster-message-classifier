from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import LABEL_COLUMN, TEXT_COLUMN, TRAIN_CLASS_WEIGHTS
from src.data_utils import load_dataset
from src.metrics import save_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression baseline.")
    parser.add_argument("--data-dir", default="data", help="Directory containing train.csv, valid.csv, test.csv")
    parser.add_argument("--output-dir", default="outputs/tfidf_lr", help="Directory for model and reports")
    parser.add_argument("--max-features", type=int, default=100000)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument(
        "--imbalance",
        choices=["class_weight", "smote", "none"],
        default="smote",
        help="Imbalance handling. SMOTE is applied only to vectorized train data.",
    )
    parser.add_argument("--top-k-words", type=int, default=30)
    return parser.parse_args()


def train_with_class_weight(args, x_train, y_train) -> Pipeline:
    class_weight = TRAIN_CLASS_WEIGHTS if args.imbalance == "class_weight" else None
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=args.max_features,
                    ngram_range=(args.ngram_min, args.ngram_max),
                    min_df=args.min_df,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=2.0,
                    class_weight=class_weight,
                    max_iter=args.max_iter,
                    solver="saga",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


def train_with_smote(args, x_train, y_train) -> Pipeline:
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        sublinear_tf=True,
    )
    x_train_vec = vectorizer.fit_transform(x_train)

    smote = SMOTE(random_state=42, k_neighbors=5)
    x_resampled, y_resampled = smote.fit_resample(x_train_vec, y_train)
    if not sparse.issparse(x_resampled):
        x_resampled = sparse.csr_matrix(x_resampled)

    clf = LogisticRegression(
        C=2.0,
        class_weight=None,
        max_iter=args.max_iter,
        solver="saga",
        random_state=42,
    )
    clf.fit(x_resampled, y_resampled)
    return Pipeline(steps=[("tfidf", vectorizer), ("clf", clf)])


def extract_top_words(model: Pipeline, output_dir: Path, top_k: int) -> None:
    vectorizer: TfidfVectorizer = model.named_steps["tfidf"]
    clf: LogisticRegression = model.named_steps["clf"]
    feature_names = np.array(vectorizer.get_feature_names_out())

    rows = {}
    for class_id, coef in enumerate(clf.coef_):
        top_idx = np.argsort(coef)[-top_k:][::-1]
        bottom_idx = np.argsort(coef)[:top_k]
        rows[str(class_id)] = {
            "positive": [
                {"word": feature_names[idx], "weight": float(coef[idx])}
                for idx in top_idx
            ],
            "negative": [
                {"word": feature_names[idx], "weight": float(coef[idx])}
                for idx in bottom_idx
            ],
        }

    with (output_dir / "top_words.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, test_df = load_dataset(args.data_dir)
    x_train = train_df[TEXT_COLUMN].tolist()
    y_train = train_df[LABEL_COLUMN].to_numpy()

    if args.imbalance == "smote":
        model = train_with_smote(args, x_train, y_train)
    else:
        model = train_with_class_weight(args, x_train, y_train)

    joblib.dump(model, output_dir / "model.joblib")
    extract_top_words(model, output_dir, args.top_k_words)

    for split_name, df in [("valid", valid_df), ("test", test_df)]:
        preds = model.predict(df[TEXT_COLUMN].tolist())
        result = save_evaluation(df[LABEL_COLUMN].to_numpy(), preds, output_dir, split_name)
        print(f"[{split_name}] {json.dumps(result['metrics'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
