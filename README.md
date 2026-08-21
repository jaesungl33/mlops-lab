# MLOps Lab — Phân loại chất lượng rượu vang

**Sinh viên:** Lee Jae Sung  
**MSSV:** 01731  
**Repo:** https://github.com/jaesungl33/mlops-lab

Lab xây dựng pipeline MLOps: một commit dữ liệu tự huấn luyện, kiểm tra chất lượng (cổng accuracy ≥ 0.70) và triển khai API lên máy ảo AWS, không cần thao tác thủ công các bước giữa.

Báo cáo ngắn: [BAO_CAO.md](BAO_CAO.md).

## Mục tiêu

- Phân loại chất lượng rượu (UCI Wine Quality) thành 3 nhãn: `thap` (0), `trung_binh` (1), `cao` (2).
- Theo dõi thí nghiệm bằng MLflow.
- Version dữ liệu bằng DVC, lưu trên Amazon S3.
- CI/CD GitHub Actions: **Test → Train → Eval → Deploy**.
- Serving FastAPI trên EC2, cổng 8000.

## Cấu trúc

```
mlops-lab/
├── .github/workflows/mlops.yml   # Pipeline CI/CD
├── .dvc/config                   # Remote DVC (S3)
├── data/*.csv.dvc                # Con trỏ DVC, không commit CSV
├── src/train.py                  # Huấn luyện RandomForest + MLflow
├── src/serve.py                  # API /health và /predict
├── tests/test_train.py
├── generate_data.py
├── add_new_data.py
├── params.yaml
└── requirements.txt
```

12 đặc trưng (đúng thứ tự API):

`fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, wine_type`

## Môi trường local

Yêu cầu: Python 3.10+, Git, AWS CLI.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python generate_data.py
# train_phase1.csv : 2998 mẫu
# eval.csv         :  500 mẫu
# train_phase2.csv : 2998 mẫu

export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
export MLFLOW_ARTIFACT_ROOT=./mlartifacts
python src/train.py

pytest tests/ -v
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

MLflow UI: http://localhost:5000

## Cloud (AWS)

| Khái niệm lab | AWS đã dùng |
|---|---|
| Object storage | S3 `mlops-lab-590999019517` |
| Máy ảo | EC2 `mlops-serve` (Ubuntu 22.04, us-east-1) |
| DVC extra | `dvc[s3]` |
| SDK | boto3 |

Dữ liệu nằm dưới prefix `dvc/`. Mô hình production: `s3://mlops-lab-590999019517/models/latest/model.pkl`.

GitHub Secrets: `CLOUD_CREDENTIALS`, `CLOUD_BUCKET`, `VM_HOST`, `VM_USER`, `VM_SSH_KEY`.

## Pipeline CI/CD

Push lên `main` (thay `data/*.dvc`, `src/**/*.py`, hoặc `params.yaml`) hoặc chạy tay `workflow_dispatch`.

1. **Test** — `pytest tests/ -v` trên dữ liệu giả, không cần cloud.
2. **Train** — `dvc pull` → `python src/train.py` → upload `models/latest/model.pkl` lên S3. Xuất `accuracy` cho job sau.
3. **Eval** — nếu `float(accuracy) < 0.70` thì fail; Deploy không chạy.
4. **Deploy** — SSH restart `mlops-serve`, `curl` `/health`.

Chứng minh cổng eval: commit `n_estimators: 5, max_depth: 2` cho accuracy 0.548; Eval đỏ, Deploy bị bỏ. Sau đó khôi phục bộ tham số tốt.

## Thêm dữ liệu mới (huấn luyện liên tục)

`dvc push` **trước** `git push`, nếu không Actions sẽ `dvc pull` object chưa có trên S3.

```bash
python add_new_data.py          # 2998 -> 5996 mẫu
dvc add data/train_phase1.csv
dvc push
git add data/train_phase1.csv.dvc
git commit -m "data: bổ sung 2998 mẫu dữ liệu mới (train_phase2)"
git push origin main
```

## API đang chạy

VM: `http://3.220.247.179:8000`

```bash
curl http://3.220.247.179:8000/health

curl -X POST http://3.220.247.179:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]}'
```

Kết quả mong đợi:

```json
{"status": "ok"}
{"prediction": 0, "label": "thap"}
```

## Siêu tham số đang dùng

```yaml
n_estimators: 150
max_depth: 18
min_samples_split: 3
criterion: gini
max_features: 0.8
```

`random_state=42` bắt buộc trong `train.py` để thí nghiệm so sánh được.
