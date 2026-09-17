/**
 * Suy luận CNN 1D trên văn bản, thuần JavaScript — Assignment 04.
 *
 * Bản dịch nguyên văn của `forward_reference()` trong `tuan 4/lib/cnn_numpy.py`.
 * Mô hình được huấn luyện trong `ml/notebook/customer_review_cnn.ipynb`, xuất ra
 * `public/model_cnn.json` và đẩy thẳng lên GitHub; Vercel phục vụ nó như một tệp
 * tĩnh nên **không có hàm serverless nào chạy** — toàn bộ phép tính diễn ra trên
 * trình duyệt của người dùng.
 *
 * Luồng đầy đủ, tái lập đúng bản Python:
 *
 *   văn bản → tách token → tra từ điển → đệm/cắt về MAX_LEN
 *           → Embedding → Conv → ReLU → Conv → ReLU → MaxPool
 *           → Conv → ReLU → GlobalMaxPool → Dense → ReLU → Dense → Softmax
 *
 * Quy ước tensor (channels-first, giống PyTorch và bản NumPy):
 *   giai đoạn chuỗi  : { c, l, v } với v[ch * l + pos]
 *   giai đoạn vector : { d, v }    với v[i]
 *
 * `ml/check_cnn_web.mjs` kiểm chứng file này khớp từng chữ số với notebook.
 */

/** Tích chập 'valid', stride 1: (c_in, L) -> (c_out, L - k + 1). */
function conv1d(a, layer) {
  const W = layer.W, b = layer.b;
  const cOut = layer.c_out, cIn = layer.c_in, k = layer.k;
  const L = a.l, Lo = L - k + 1;
  const out = new Float64Array(cOut * Lo);
  for (let o = 0; o < cOut; o++) {
    const Wo = W[o], base = o * Lo;
    for (let p = 0; p < Lo; p++) {
      let s = b[o];
      for (let c = 0; c < cIn; c++) {
        const Woc = Wo[c], off = c * L + p;
        for (let j = 0; j < k; j++) s += Woc[j] * a.v[off + j];
      }
      out[base + p] = s;
    }
  }
  return { c: cOut, l: Lo, v: out };
}

function relu(a) {
  const v = new Float64Array(a.v.length);
  for (let i = 0; i < v.length; i++) v[i] = a.v[i] > 0 ? a.v[i] : 0;
  return { ...a, v };
}

function maxPool1d(a, p) {
  const Lo = Math.floor(a.l / p);
  const out = new Float64Array(a.c * Lo);
  for (let c = 0; c < a.c; c++) {
    for (let i = 0; i < Lo; i++) {
      let m = -Infinity;
      const off = c * a.l + i * p;
      for (let j = 0; j < p; j++) if (a.v[off + j] > m) m = a.v[off + j];
      out[c * Lo + i] = m;
    }
  }
  return { c: a.c, l: Lo, v: out };
}

/** Max trên toàn trục thời gian: (c, L) -> vector độ dài c. */
function globalMaxPool1d(a) {
  const out = new Float64Array(a.c);
  for (let c = 0; c < a.c; c++) {
    let m = -Infinity;
    for (let p = 0; p < a.l; p++) if (a.v[c * a.l + p] > m) m = a.v[c * a.l + p];
    out[c] = m;
  }
  return { d: a.c, v: out };
}

function dense(a, layer) {
  const W = layer.W, b = layer.b, dOut = b.length;
  const out = new Float64Array(dOut);
  for (let j = 0; j < dOut; j++) out[j] = b[j];
  for (let i = 0; i < a.d; i++) {
    const ai = a.v[i];
    if (ai === 0) continue;
    const Wi = W[i];
    for (let j = 0; j < dOut; j++) out[j] += ai * Wi[j];
  }
  return { d: dOut, v: out };
}

/** Tra bảng nhúng: mảng chỉ số độ dài L -> (dim, L). */
function embedding(ids, layer) {
  const E = layer.E, dim = layer.dim, L = ids.length;
  const out = new Float64Array(dim * L);
  for (let p = 0; p < L; p++) {
    const row = E[ids[p]] || E[layer.pad_idx];
    for (let c = 0; c < dim; c++) out[c * L + p] = row[c];
  }
  return { c: dim, l: L, v: out };
}

function softmax(v) {
  let max = -Infinity;
  for (const x of v) if (x > max) max = x;
  let sum = 0;
  const e = Array.from(v, (x) => {
    const t = Math.exp(x - max);
    sum += t;
    return t;
  });
  return e.map((t) => t / sum);
}

/** Chạy xuôi từ mảng chỉ số token, trả về vector xác suất. */
export function cnnForward(model, ids) {
  const layers = model.layers;
  let a = embedding(ids, layers[0]);
  for (let i = 1; i < layers.length; i++) {
    const L = layers[i];
    switch (L.type) {
      case 'conv1d': a = conv1d(a, L); break;
      case 'relu': a = relu(a); break;
      case 'maxpool1d': a = maxPool1d(a, L.p); break;
      case 'globalmaxpool1d': a = globalMaxPool1d(a); break;
      case 'flatten': a = { d: a.c * a.l, v: a.v }; break;
      case 'dense': a = dense(a, L); break;
      case 'dropout': break;                 // suy luận: dropout vô hiệu
      default: throw new Error(`Tầng lạ trong bundle: ${L.type}`);
    }
  }
  return softmax(a.v);
}

/** Dựng bảng tra từ điển một lần cho mỗi bundle, thay vì mỗi lần dự đoán. */
export function prepareCnnModel(bundle) {
  const vocabIndex = new Map();
  bundle.tokenizer.vocab.forEach((w, i) => vocabIndex.set(w, i));
  return { ...bundle, _vocabIndex: vocabIndex };
}

/**
 * Tách token và mã hoá y hệt bản Python. Regex, quy tắc viết thường và độ dài
 * tối đa đều đọc từ `model.tokenizer` nên hai bên không thể lệch nhau vì sửa
 * một bên mà quên bên kia.
 */
export function encodeText(model, text) {
  const tk = model.tokenizer;
  const re = new RegExp(tk.regex, 'g');
  const raw = tk.lowercase ? String(text).toLowerCase() : String(text);
  const toks = raw.match(re) || [];
  const kept = toks.slice(0, tk.max_len);
  const ids = new Array(tk.max_len).fill(tk.pad_index);
  for (let i = 0; i < kept.length; i++) {
    const idx = model._vocabIndex.get(kept[i]);
    ids[i] = idx === undefined ? tk.unk_index : idx;
  }
  return { ids, tokens: kept };
}

/**
 * Giải thích bằng phép che (occlusion): lần lượt thay từng token bằng <PAD> và
 * đo xác suất của lớp thắng tụt bao nhiêu.
 *
 * Vì sao chọn cách này thay vì gradient: nó **đo trực tiếp** thứ ta muốn biết
 * ("bỏ từ này đi thì mô hình còn tin vào kết luận không?"), không cần giả định
 * tuyến tính hoá quanh điểm hiện tại, và không phụ thuộc vào việc lan truyền
 * ngược qua max-pool — nơi gradient vốn đã không liên tục.
 *
 * Chi phí: một lượt forward cho mỗi token thật (thường 40–90 lượt). Mạng chỉ có
 * ~75k tham số nên tổng thời gian dưới một phần mười giây.
 */
export function explainByOcclusion(model, ids, tokens, classIndex, baseProb) {
  const tk = model.tokenizer;
  const scores = [];
  for (let i = 0; i < tokens.length; i++) {
    const masked = ids.slice();
    masked[i] = tk.pad_index;
    const p = cnnForward(model, masked)[classIndex];
    scores.push({ token: tokens[i], position: i, drop: baseProb - p });
  }
  return scores
    .filter((s) => Math.abs(s.drop) > 1e-6)
    .sort((a, b) => Math.abs(b.drop) - Math.abs(a.drop));
}

/** Dự đoán ngành hàng từ một đoạn bình luận. */
export function predictCnn(model, text, { explain = true } = {}) {
  const { ids, tokens } = encodeText(model, text);
  if (tokens.length === 0) return null;

  const probs = cnnForward(model, ids);
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;

  const ranking = probs
    .map((p, i) => ({ label: model.classes[i], labelVi: model.class_labels_vi[i], p, i }))
    .sort((a, b) => b.p - a.p);

  const oov = tokens.filter((t) => !model._vocabIndex.has(t)).length;

  return {
    classIndex: best,
    label: model.classes[best],
    labelVi: model.class_labels_vi[best],
    probability: probs[best],
    probs,
    ranking,
    runnerUp: ranking[1],
    nTokens: tokens.length,
    nOov: oov,
    truncated: tokens.length >= model.tokenizer.max_len,
    topTokens: explain
      ? explainByOcclusion(model, ids, tokens, best, probs[best]).slice(0, 8)
      : [],
  };
}

/** Nạp public/model_cnn.json (cùng origin — không có backend nào tham gia). */
export async function loadCnnModel(url = `${import.meta.env.BASE_URL}model_cnn.json`) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Không tải được model_cnn.json (HTTP ${res.status})`);
  return prepareCnnModel(await res.json());
}
