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

---

# 🧠 Assignment 03 — Bài toán mới: dự đoán sở thích ngành hàng bằng mạng nơ-ron sâu 5 tầng

Trang web nay có **hai tab**, mỗi tab một bài toán và một mô hình:

| Tab | Bài toán | Mô hình | Nguồn |
|---|---|---|---|
| **Sở thích ngành hàng** (mặc định) | Multiclass, 6 lớp — `Department Name` | Deep MLP 5 tầng, NumPy from scratch | Assignment 03 |
| Khách có giới thiệu? | Binary — `Recommended IND` | Logistic Regression | Assignment 02 |

## Bài toán

Từ **hồ sơ khách hàng + nội dung bình luận**, suy ra khách đang quan tâm ngành hàng nào
trong 6 nhóm: Tops, Dresses, Bottoms, Intimate, Jackets, Trend.

| Ngành hàng | Số mẫu | Tỉ lệ |
|---|---|---|
| Tops | 10.468 | 44.60% |
| Dresses | 6.319 | 26.92% |
| Bottoms | 3.799 | 16.19% |
| Intimate | 1.735 | 7.39% |
| Jackets | 1.032 | 4.40% |
| Trend | **119** | **0.51%** |

**Mất cân bằng 88:1** — đây là thách thức trung tâm của bài toán.

## ⚠️ Rò rỉ nhãn: vì sao `Class Name` bị loại khỏi đầu vào

`Class Name` (20 giá trị: Blouses, Jeans, Dresses…) là **hạng mục con** của
`Department Name`. Kiểm tra thực tế: **20/20** giá trị `Class Name` ánh xạ tới đúng một
`Department Name` — biết "Jeans" thì chắc chắn là "Bottoms". Đưa cột này vào đầu vào cho
~100% accuracy nhưng mô hình **không học được gì**. Giao diện web đánh dấu rõ trường này
là *"rò rỉ nhãn — bị loại"*.

## Không gian đặc trưng (606 chiều)

```
[   3 numeric ]  Age, Rating, Positive Feedback Count  -> median-impute -> standard-scale
[   3 one-hot ]  Division Name
[ 600 tfidf   ]  Review Text -> TfidfVectorizer(1-2 gram, english stop words, L2)
```

TF-IDF giới hạn 600 (thay vì 2000 như Assignment 02) vì tầng 1 có `600 × 128` trọng số —
2000 chiều sẽ đẩy `model_deep.json` lên ~2.4 MB, quá nặng cho một trang tĩnh.

**Ba đặc trưng bảng gần như vô dụng ở bài toán này:** rating và tuổi phân bố gần như
đồng nhất giữa 6 ngành hàng (xem `figures/p3_fig1_eda.png`). Toàn bộ tín hiệu nằm trong
nội dung bình luận.

## Kiến trúc

| Tầng | W shape | Kích hoạt | Tham số |
|---|---|---|---|
| Layer 1 | (606, 128) | ReLU | **77.696** |
| Layer 2 | (128, 64) | ReLU | 8.256 |
| Layer 3 | (64, 32) | ReLU | 2.080 |
| Layer 4 | (32, 16) | ReLU | 528 |
| Layer 5 | (16, 6) | **Softmax** | 102 |
| **Tổng** | | | **88.662** |

Riêng tầng 1 chiếm **87.6%** tham số — quy luật chung của mạng xử lý dữ liệu chiều cao.

## Ablation trọng số lớp

Ba mạng giống hệt nhau về kiến trúc và siêu tham số, chỉ khác cách đặt trọng số lớp;
chọn phương án tốt nhất theo **macro-F1 trên tập validation**:

| Trọng số lớp | Val Accuracy | Val Balanced Acc | Val Macro-F1 |
|---|---|---|---|
| `none` | **0.8339** | 0.6147 | 0.6413 |
| **`sqrt`** ★ chọn | 0.8206 | **0.6586** | **0.6583** |
| `balanced` | 0.7541 | 0.6413 | 0.6259 |

Phương án `balanced` "chuẩn sách vở" lại **tệ nhất**: trọng số 32.99 cho lớp Trend khiến
mạng đoán lớp này quá dễ dãi, precision rơi xuống 0.012. Phương án `sqrt` làm mềm trọng
số (2.97) cho kết quả tốt nhất. **Công thức chống mất cân bằng phải được kiểm chứng
bằng thực nghiệm, không áp dụng máy móc.**

## Kết quả trên tập Test (3.521 mẫu)

Đường cơ sở "đoán lớp phổ biến nhất": accuracy **0.4459**.

| Mô hình | Accuracy | Balanced Acc | Macro-F1 | Top-2 Acc |
|---|---|---|---|---|
| Logistic Regression | **0.8481** | 0.6334 | **0.6615** | **0.9341** |
| Deep MLP-5 (PyTorch) | 0.8270 | 0.6552 | 0.6588 | 0.9179 |
| **Deep MLP-5 (NumPy)** | 0.8242 | **0.6614** | 0.6576 | 0.9270 |
| Decision Tree | 0.8185 | 0.6178 | 0.6474 | 0.8915 |
| Random Forest | 0.8273 | 0.5686 | 0.5993 | 0.9284 |
| Multinomial NB | 0.7060 | 0.4772 | 0.4956 | 0.8895 |

**Kết luận trung thực:** mạng sâu **ngang bằng** Logistic Regression về accuracy và
macro-F1, nhưng **vượt trội về balanced accuracy** (0.661 so với 0.633) và về F1 trên
lớp khó `Jackets` (0.571 so với 0.502; Random Forest chỉ 0.209). Nói cách khác, mạng sâu
**công bằng hơn giữa các lớp**.

Lý do LogReg vẫn bám sát: sau khi qua TF-IDF, bài toán đã gần như tách được tuyến tính —
từ "dress" một mình đủ quyết định lớp Dresses. TF-IDF thực chất là một **đặc trưng thủ
công rất tốt** đã làm sẵn phần khó. Ưu thế thật sự của kiến trúc sâu sẽ xuất hiện khi bỏ
TF-IDF và cho mạng đọc thẳng chuỗi token (embedding + LSTM/Transformer).

Lớp `Trend` có F1 = 0.000 với **mọi** mô hình: 119 mẫu và không có từ vựng riêng —
"Trend" là nhãn marketing gộp sản phẩm thời thượng thuộc mọi kiểu dáng, nên không có tín
hiệu văn bản nào để học. Đây là giới hạn của bài toán, không phải của mô hình.

## Mạng đã học được gì

Saliency (`d logit_c / d x`, trừ trung bình trên 6 lớp) cho thấy tầng 1 đã tự gom các từ
đồng nghĩa mà **không ai lập trình quan hệ đó**:

| Áo (Tops) | Váy đầm (Dresses) | Quần (Bottoms) |
|---|---|---|
| shirt (+16.23) | dress (+30.12) | skirt (+5.58) |
| tops (+14.48) | dresses (+13.76) | pants (+4.33) |
| blouse (+13.69) | wedding (+10.86) | jumpsuit (+3.97) |
| sweater (+12.62) | knee (+7.72) | shorts (+3.93) |
| tank (+11.02) | belt (+7.54) | thighs (+3.44) |

Cột `Dresses` còn học được cả **ngữ cảnh sử dụng** (wedding, knee, belt, slip), không chỉ
tên sản phẩm. Trang web tính saliency này **ngay trên trình duyệt** cho từng dự đoán.

## Triển khai serverless

`public/model_deep.json` (741.9 KB) chứa trọng số, vocabulary + IDF, stop words, tham số
chuẩn hoá và các chỉ số đánh giá. `src/model/deepInfer.js` dựng lại đúng pipeline bằng
JavaScript, dùng chung bộ phân tích n-gram với `infer.js`.

### Kiểm chứng

```bash
node ml/deep_parity.mjs
```

```
Đã kiểm tra 25 mẫu tham chiếu.
Sai lệch xác suất lớn nhất : 3.744e-5 (ngưỡng 0.0005)
Số mẫu lệch nhãn           : 0
✓ PARITY PASSED — suy luận trên trình duyệt trùng khớp với notebook.
```

Sai lệch ~3.7e-5 đến từ việc làm tròn trọng số về 5 chữ số thập phân để giữ file ở mức
740 KB; nó không đổi bất kỳ nhãn dự đoán nào.

Notebook huấn luyện: [`ml/notebook/customer_interest_deep.ipynb`](ml/notebook/customer_interest_deep.ipynb).
Biểu đồ phân tích: [`figures/`](figures/).
