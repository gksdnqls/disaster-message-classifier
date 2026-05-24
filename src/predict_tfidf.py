from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from src.config import ID_TO_LABEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a disaster message label with a TF-IDF model.")
    parser.add_argument("--model-path", default="outputs/tfidf_lr_smote/model.joblib")
    parser.add_argument("--text", required=True, help="Message text to classify.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    pred = int(model.predict([args.text])[0])

    print(f"입력 문장: {args.text}")
    print(f"예측 라벨: {ID_TO_LABEL[pred]}")
    print(f"label_id: {pred}")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([args.text])[0]
        print("확률:")
        for idx, prob in enumerate(probs):
            print(f"- {ID_TO_LABEL[idx]}({idx}): {prob:.4f}")


if __name__ == "__main__":
    main()
