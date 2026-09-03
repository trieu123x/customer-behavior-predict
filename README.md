# Customer Interest Discovery — dự đoán khách hàng có recommend sản phẩm không

Dự đoán một khách hàng thương mại điện tử có giới thiệu (recommend) sản phẩm hay không,
từ hồ sơ khách hàng, danh mục sản phẩm và **nội dung bình luận**.

Ứng dụng chạy **hoàn toàn trong trình duyệt**: không có API, không có server suy luận.
Model đã huấn luyện được xuất ra `public/model.json` (84 KB) và `src/model/infer.js`
thực hiện lại đúng các phép biến đổi của scikit-learn bằng JavaScript. Dữ liệu người
dùng nhập không rời khỏi máy của họ.

## Model

| | |
|---|---|
| Bài toán | Phân loại nhị phân — `Recommended IND` |
| Thuật toán | Logistic Regression (`max_iter=2000`, `random_state=42`) |
| Biểu diễn đặc trưng | 3 numeric (chuẩn hoá) + 29 one-hot + 2000 TF-IDF (1–2 gram, English stop words, L2) = **2032 chiều** |
| Dữ liệu | 23.486 review · chia 70/15/15 phân tầng (16.440 / 3.523 / 3.523) |
| Test Accuracy | 0.9421 |
| Test F1 | 0.9649 |
| Test ROC-AUC | 0.9792 |

So sánh biểu diễn (trên tập validation): tabular-only F1 0.9603 · text-only F1 0.9378 ·
**kết hợp cả hai F1 0.9638** — kết hợp tốt hơn từng loại riêng lẻ.

Chi tiết EDA, tiền xử lý và so sánh 6 mô hình nằm trong
[`ml/notebook/customer_behavior.ipynb`](ml/notebook/customer_behavior.ipynb).

## Vì sao không cần backend

Pipeline gồm ba khối nối nhau, và mọi phép tính lúc suy luận đều là tra bảng + nhân +
tích vô hướng:

```
[ 3 numeric  ]  điền median  -> (x - mean) / scale
[ 29 one-hot ]  điền mode    -> OneHotEncoder(handle_unknown='ignore')
[ 2000 tfidf ]  TextCleaner  -> TfidfVectorizer(1-2 gram, stop words, L2)
                                     |
                                     v
                    z = intercept + w · x   ->   p = sigmoid(z)
```

`ml/export_model.py` ghi toàn bộ hằng số đã học (median, mean/scale, danh sách category,
vocabulary + IDF, hệ số hồi quy, danh sách stop words của scikit-learn) ra
`public/model.json`. `src/model/infer.js` nạp file đó và lặp lại đúng các bước trên —
kể cả chi tiết dễ sai như: stop words bị loại **trước** khi ghép n-gram, token pattern
`\b\w\w+\b`, chuẩn hoá L2 **sau** khi nhân IDF, và category lạ thì mã hoá thành khối 0.

### Kiểm chứng

`ml/parity_check.py` chấm điểm cả 23.486 dòng dữ liệu gốc bằng file `.joblib` thật rồi
so với kết quả của `infer.js` chạy trên Node:

```
scikit-learn scored 23486 rows
javascript scored  : 23486 rows
max |p_js - p_py|  : 1.623e-12
label mismatches   : 0
PARITY OK
```

Sai số 1e-12 chỉ đến từ việc làm tròn số thực khi xuất JSON; không dòng nào đổi nhãn.

## Chạy tại máy

```bash
npm install
npm run dev      # http://localhost:5173
```

Build production: `npm run build` (kết quả ở `dist/`).

## Deploy lên Vercel

Repo này là zero-config: import repo vào Vercel, Vercel tự nhận Vite và dùng
`npm run build` → `dist/`. Không cần biến môi trường, không cần serverless function.

## Cấu trúc

```
├── index.html, vite.config.js, package.json
├── public/
│   └── model.json          ← model đã xuất, fetch trực tiếp từ trình duyệt
├── src/
│   ├── App.jsx             ← form nhập + hiển thị kết quả
│   ├── index.css
│   └── model/infer.js      ← bản port scikit-learn sang JavaScript
└── ml/                     ← phần huấn luyện (không tham gia vào build web)
    ├── notebook/customer_behavior.ipynb
    ├── common.py           ← TextCleaner, dùng chung notebook + API
    ├── model/model_pipeline.joblib, metadata.json
    ├── data/ecommerce_raw.csv
    ├── api/app.py          ← bản FastAPI cũ, giữ lại để đối chiếu
    ├── export_model.py     ← .joblib -> public/model.json
    └── parity_check.py     ← so sánh JS với scikit-learn
```

## Huấn luyện lại

```bash
cd ml
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Linux/macOS: .venv/bin/pip
.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace notebook/customer_behavior.ipynb
.venv/Scripts/python export_model.py                 # cập nhật public/model.json
.venv/Scripts/python parity_check.py                 # phải in PARITY OK
```

Dataset gốc: [Women's E-Commerce Clothing Reviews](https://raw.githubusercontent.com/AFAgarap/ecommerce-reviews-analysis/master/Womens%20Clothing%20E-Commerce%20Reviews.csv)
(đã kèm sẵn trong `ml/data/`).
