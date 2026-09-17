# Assignment 04 — CNN 1D trên văn bản

> **Môn học:** Intelligent System Development — TS. Trần Đình Quế
> **Sinh viên:** Đinh Hải Triều — B23DCCN843 — Lớp 06

Bổ sung cho `README.md`. Tài liệu này mô tả tab **"CNN 1D trên văn bản"** được
thêm vào ở Assignment 04 — mô hình duy nhất trong ba bài mà giả định cục bộ của
tích chập **thực sự đúng**.

## Tệp mới

| File | Nội dung |
|---|---|
| `ml/notebook/customer_review_cnn.ipynb` | Notebook huấn luyện CNN văn bản bằng **NumPy from scratch**, PyTorch và TensorFlow |
| `public/model_cnn.json` | Trọng số + **từ điển 4.000 token** (742 KB) |
| `ml/model_cnn_samples.json` | 12 mẫu tham chiếu kèm **văn bản thô** |
| `src/model/cnnInfer.js` | Bản suy luận CNN + tokenizer thuần JavaScript |
| `ml/check_cnn_web.mjs` | Kiểm chứng mô-đun React khớp notebook |
| `figures/c3_fig*.png` | 5 biểu đồ do notebook sinh ra |

## Kiến trúc

```
tokens(100) → Embedding(4000 × 16) → Conv1D(32, K=3) → ReLU
            → Conv1D(32, K=3) → ReLU → MaxPool(2)
            → Conv1D(48, K=3) → ReLU → GlobalMaxPool1D
            → Dropout(0.25) → Dense(32) → ReLU → Dense(6) → Softmax
```

**75.094 tham số · 6 tầng huấn luyện được**, trong đó bảng nhúng chiếm
**64.000 (85%)**. Độ dài chuỗi: $100 \to 98 \to 96 \to 48 \to 46 \to 1$.

`GlobalMaxPool1D` lấy max trên **toàn** trục thời gian, nên mỗi bộ lọc trả lời
đúng một câu hỏi: *"n-gram mà tôi phụ trách có xuất hiện trong câu không?"* — nhờ
vậy câu dài hay ngắn đều cho ra vector cùng số chiều.

> **Mô hình chỉ đọc trường `Review Text`.** Mọi đặc trưng bảng đều bị bỏ qua.
> Cột `Class Name` xác định `Department Name` một–một (Jeans → Bottoms) nên bắt
> buộc phải loại — giữ lại là rò rỉ nhãn.

## Triển khai serverless trên Vercel

`public/model_cnn.json` được **commit thẳng lên GitHub** và Vercel phục vụ như
tệp tĩnh. Bundle mang theo **cả từ điển**, nên trình duyệt làm được trọn vẹn
chuỗi: tách token → tra từ điển → nhúng → tích chập → pooling → softmax.

```javascript
export async function loadCnnModel(url = `${import.meta.env.BASE_URL}model_cnn.json`) {
  const res = await fetch(url);
  return prepareCnnModel(await res.json());
}
```

Tokenizer đọc quy tắc thẳng từ bundle nên **không thể lệch** bản Python:

```javascript
export function encodeText(model, text) {
  const tk = model.tokenizer;
  const re = new RegExp(tk.regex, 'g');            // "[a-z0-9']+"
  const toks = String(tk.lowercase ? text.toLowerCase() : text).match(re) || [];
  const ids = new Array(tk.max_len).fill(tk.pad_index);
  for (let i = 0; i < Math.min(toks.length, tk.max_len); i++) {
    const idx = model._vocabIndex.get(toks[i]);
    ids[i] = idx === undefined ? tk.unk_index : idx;
  }
  return { ids, tokens: toks.slice(0, tk.max_len) };
}
```

## Giải thích dự đoán bằng phép che (occlusion)

Giao diện hiển thị token nào củng cố kết luận, đo bằng cách lần lượt thay từng
token bằng `<PAD>` rồi xem xác suất lớp thắng tụt bao nhiêu:

```
"this dress is beautiful and the fabric feels great"  →  Đầm (97,2%)
   dress +80,3%   this +3,5%   is +2,7%   and −1,2%   beautiful +1,1%
```

Cách này **đo trực tiếp** thứ ta muốn biết ("bỏ từ này đi thì mô hình còn tin vào
kết luận không?"), không cần giả định tuyến tính hoá và không phải lan truyền
ngược qua max-pool — nơi gradient vốn không liên tục. Chi phí: một lượt forward
cho mỗi token thật, dưới một phần mười giây.

## Kết quả trên tập kiểm thử (3.395 bình luận)

| Mô hình | Accuracy | Balanced Acc | Macro-F1 |
|---|---|---|---|
| TF-IDF 1-2gram + LogReg | **0,8483** | 0,6444 | **0,6354** |
| TF-IDF unigram + LogReg | 0,8221 | **0,6484** | 0,6209 |
| **CNN (NumPy from scratch)** | 0,7944 | 0,6408 | 0,6035 |
| CNN (TensorFlow) | 0,7703 | 0,6163 | 0,5831 |
| CNN (PyTorch) | 0,7464 | 0,5775 | 0,5568 |

Dữ liệu mất cân bằng **88:1** (Tops 10.468 mẫu, Trend 119 mẫu), nên Accuracy gây
hiểu lầm — chỉ đoán `Tops` cho mọi mẫu đã đạt 44,4%. Chỉ số chính là **Macro-F1**
và **Balanced Accuracy**.

## Thí nghiệm then chốt: thứ tự từ CÓ quan trọng

Xáo trộn ngẫu nhiên thứ tự từ trong từng bình luận của tập test (giữ nguyên đúng
tập từ), rồi dự đoán lại:

| Mô hình | Nhìn thấy thứ tự? | Macro-F1 gốc | Sau xáo trộn | Mức sụt |
|---|---|---|---|---|
| **CNN** | **Có** | 0,6035 | 0,5220 | **−0,0814** |
| TF-IDF 1-2gram | Một phần (bigram) | 0,6354 | 0,6253 | −0,0101 |
| TF-IDF unigram | **Không (túi từ)** | 0,6209 | 0,6209 | **0,0000** |

Mô hình túi từ sụt **đúng 0,0000** — không phải "xấp xỉ 0" mà bằng 0 tuyệt đối,
đúng như lý thuyết (notebook có `assert` cho điều này). Mức phụ thuộc vào thứ tự
xếp thành một **quan hệ đơn điệu** khớp chính xác khả năng "nhìn thấy thứ tự":

$$0{,}0000 \;<\; 0{,}0101 \;<\; 0{,}0814$$

**Đây là bài duy nhất trong ba bài mà tích chập có cơ sở về bản chất dữ liệu.**
Khác hai bài dữ liệu bảng (nơi thứ tự cột do file CSV quyết định), token liền kề
trong câu **thực sự** tạo thành cụm có nghĩa.

Trực quan hoá bộ lọc củng cố kết luận — mỗi kernel đã thành một bộ dò cụm từ:

| Bộ lọc | 3-gram kích hoạt mạnh nhất | Ngành hàng |
|---|---|---|
| #11 | `yoga pants they` | Bottoms |
| #10 | `byron lars dresses` | Dresses (tên nhà thiết kế) |
| #19 | `kimono style jackets` | Jackets |
| #15 | `ag stevie jeans` | Bottoms (thương hiệu) |
| #30 | `great lounge sleep` | Intimate |

Nhiều bộ lọc bắt được **tên thương hiệu** hai từ — thứ mà mô hình unigram sẽ tách
rời và làm mất nghĩa.

## Nhưng "có cơ sở" ≠ "thắng"

TF-IDF vẫn nhỉnh hơn 0,032 Macro-F1. Lý do: nhãn ở đây là *loại sản phẩm*, phần
lớn quyết định bởi **sự có mặt của từ khoá** (`dress`, `jeans`, `blouse`) hơn là
bởi trật tự. CNN chỉ thật sự vượt trội khi bài toán **bắt buộc** phân biệt cụm từ
— ví dụ phân tích cảm xúc, nơi `not good` phải khác `very good`.

Chi tiết ở Chương III và IV báo cáo `A04_06_trieu.843.PDF`.

## Kiểm chứng

```bash
node ml/check_cnn_web.mjs
```

Mẫu tham chiếu lưu **văn bản thô** (không lưu mảng chỉ số), nên bản JavaScript
buộc phải tự tách token đúng như Python thì mới khớp. Kết quả: 12/12 nhãn khớp,
sai lệch xác suất tối đa **4,2 × 10⁻⁶**.

## Chạy local

```bash
npm install
npm run dev
```
