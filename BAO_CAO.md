# Báo cáo MLOps Lab

**Sinh viên:** Lee Jae Sung  
**MSSV:** 01731  
**Môn / bài:** MLOps Lab — pipeline huấn luyện, cổng chất lượng và triển khai  
**Hạ tầng:** AWS (S3 + EC2), GitHub Actions, DVC, MLflow

## 1. Bộ siêu tham số đã chọn

Đã chạy nhiều thí nghiệm RandomForest (`random_state=42`) trên tập eval 500 mẫu, ghi MLflow (accuracy, f1_score). Ba bộ gợi ý trong đề (100/5/2, 50/3, 200/10/5) đều **dưới 0.70**, không đủ để Deploy.

Bộ chọn sau khi so sánh trên UI MLflow:

| Tham số | Giá trị | Lý do |
|---|---|---|
| `n_estimators` | 150 | Đủ cây, không quá nặng trên CI |
| `max_depth` | 18 | Sâu hơn 3–5 để tách được lớp chất lượng |
| `min_samples_split` | 3 | Giảm overfit nhẹ so với 2 |
| `criterion` | gini | Tốt nhất trong lưới đã thử |
| `max_features` | 0.8 | Dùng ~80% đặc trưng mỗi lần tách |

Accuracy local trên 2.998 mẫu: **0.7140** (F1 **0.7136**). Cùng bộ này trên GitHub Actions (Linux) vẫn ≥ 0.70, trong khi bộ nông hơn bị CI hạ còn 0.696 và bị cổng chặn.

## 2. So sánh 2.998 mẫu và 5.996 mẫu

Cùng siêu tham số và cùng held-out `eval.csv` (500 mẫu, không dùng để train):

| Lần chạy | Số mẫu train | Accuracy | F1 (weighted) |
|---|---|---|---|
| Phase 1 | 2.998 | 0.7140 | 0.7136 |
| Phase 1 + phase 2 | 5.996 | 0.7380 | 0.7381 |

Thêm 2.998 mẫu (`add_new_data.py` + commit `data/train_phase1.csv.dvc`) làm mô hình khá hơn trên eval. Pipeline [data: bổ sung 2998 mẫu…](https://github.com/jaesungl33/mlops-lab/actions/runs/32453286028) chạy đủ Test → Train → Eval → Deploy màu xanh.

## 3. Khó khăn và cách xử lý

- **GCP không có billing / project hợp lệ.** Chuyển AWS theo bảng ánh xạ đề bài: S3 thay GCS, EC2 thay GCE, `dvc[s3]` + boto3.
- **Cổng 0.70 sát ngưỡng.** Cùng params, Mac được 0.702 nhưng runner Linux được 0.696 → Eval fail, Deploy không chạy. Đã grid lại và chọn bộ 0.714 để có biên.
- **`dvc push` không nhận credential `aws login`.** boto3 không đọc session CLI; đẩy object cache bằng AWS CLI trước, rồi mới `git push` (đúng thứ tự đề: DVC lên storage trước khi Actions `dvc pull`).
- **Eval gate.** Cố ý push `n_estimators: 5, max_depth: 2` → accuracy **0.5480**, job Eval thất bại, job Deploy bị `needs: eval` bỏ qua ([run](https://github.com/jaesungl33/mlops-lab/actions/runs/32453521204)). Template upload model ở job Train nên file S3 bị ghi model yếu; VM không restart nên API cũ vẫn sống. Đã restore params tốt và Deploy lại.

Endpoint kiểm tra: `http://3.220.247.179:8000/health` trả `{"status":"ok"}`; `/predict` với vector 12 đặc trưng mẫu đề trả `{"prediction":0,"label":"thap"}`.
