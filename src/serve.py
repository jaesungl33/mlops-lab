import os

import boto3
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

S3_BUCKET = os.environ["S3_BUCKET"]
S3_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")

LABELS = {0: "thap", 1: "trung_binh", 2: "cao"}


def download_model():
    """Tải model.pkl từ S3 về máy khi server khởi động."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    s3 = boto3.client("s3")
    s3.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
    print(f"Downloaded s3://{S3_BUCKET}/{S3_MODEL_KEY} -> {MODEL_PATH}")


download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """GitHub Actions gọi endpoint này sau khi deploy để xác nhận server sống."""
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Đầu vào: JSON {"features": [f1, f2, ..., f12]}
    Đầu ra:  JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}
    """
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400,
            detail="Expected exactly 12 features",
        )

    pred = int(model.predict([req.features])[0])
    return {"prediction": pred, "label": LABELS[pred]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
