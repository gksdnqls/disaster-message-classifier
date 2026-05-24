from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import ID_TO_LABEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a disaster message label with a fine-tuned KLUE-BERT model.")
    parser.add_argument("--model-dir", default="outputs/klue_bert/best_model")
    parser.add_argument("--text", required=True, help="Message text to classify.")
    parser.add_argument("--max-length", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    encoded = tokenizer(
        args.text,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

    pred = int(max(range(len(probs)), key=lambda idx: probs[idx]))
    print(f"입력 문장: {args.text}")
    print(f"예측 라벨: {ID_TO_LABEL[pred]}")
    print(f"label_id: {pred}")
    print("확률:")
    for idx, prob in enumerate(probs):
        print(f"- {ID_TO_LABEL[idx]}({idx}): {prob:.4f}")


if __name__ == "__main__":
    main()
