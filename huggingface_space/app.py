from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


app = FastAPI()

MODEL_DIR = "best_model"
ID_TO_LABEL = {
    0: "일반",
    1: "주의",
    2: "긴급",
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()


class PredictRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "disaster classifier server",
    }


@app.post("/predict")
def predict(request: PredictRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        label_id = int(torch.argmax(probabilities).item())

    return {
        "text": request.text,
        "label_id": label_id,
        "label": ID_TO_LABEL[label_id],
        "confidence": float(probabilities[label_id].item()),
        "probabilities": {
            ID_TO_LABEL[i]: float(probabilities[i].item())
            for i in range(len(ID_TO_LABEL))
        },
    }
