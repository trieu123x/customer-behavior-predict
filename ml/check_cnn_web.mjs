/**
 * Kiểm chứng src/model/cnnInfer.js khớp với mô hình NumPy trong notebook.
 *
 * Notebook `ml/notebook/customer_review_cnn.ipynb` ghi ra `ml/model_cnn_samples.json`
 * gồm VĂN BẢN THÔ kèm vector xác suất kỳ vọng. Script này chạy lại bằng đúng mã
 * mà trình duyệt sẽ chạy — gồm cả bước tách token — rồi so từng con số.
 *
 *   node ml/check_cnn_web.mjs
 *
 * Ngưỡng sai số: model_cnn.json lưu trọng số làm tròn 6 chữ số, nên ngay cả bản
 * cài hoàn hảo cũng lệch cỡ 1e-5 so với mô hình gốc (notebook đo 8.7e-6).
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');

// cnnInfer.js dùng import.meta.env của Vite cho URL mặc định; ở Node thì không có,
// nhưng ta chỉ gọi các hàm thuần nên chỉ cần khai báo để module nạp được.
process.env.NODE_ENV ??= 'test';

const { prepareCnnModel, predictCnn, encodeText, cnnForward } =
  await import('../src/model/cnnInfer.js');

const read = (p) => JSON.parse(readFileSync(p, 'utf8'));
const bundle = read(join(ROOT, 'public', 'model_cnn.json'));
const samples = read(join(HERE, 'model_cnn_samples.json'));
const model = prepareCnnModel(bundle);

let failures = 0;
const ok = (cond, msg) => {
  console.log(`  [${cond ? 'PASS' : 'FAIL'}] ${msg}`);
  if (!cond) failures++;
};

console.log(`\nMô hình: ${model.architecture_text}`);
console.log(`${model.n_params.toLocaleString('vi-VN')} tham số · từ điển `
          + `${model.tokenizer.vocab.length.toLocaleString('vi-VN')} token\n`);

// --- 1. Xác suất khớp notebook, đi thẳng từ văn bản thô ----------------------
let worst = 0, labelHits = 0;
for (const s of samples) {
  const { ids } = encodeText(model, s.text);
  const probs = cnnForward(model, ids);
  for (let i = 0; i < probs.length; i++) {
    worst = Math.max(worst, Math.abs(probs[i] - s.expected_probs[i]));
  }
  const best = probs.indexOf(Math.max(...probs));
  if (model.classes[best] === s.expected_class) labelHits++;
}
ok(worst < 1e-4, `xác suất khớp notebook trên ${samples.length} mẫu — max|Δ| = ${worst.toExponential(2)}`);
ok(labelHits === samples.length, `nhãn dự đoán khớp ${labelHits}/${samples.length} mẫu`);

// --- 2. predictCnn() trả về cấu trúc mà giao diện cần -------------------------
const r = predictCnn(model, samples[0].text);
ok(r !== null, 'predictCnn trả về kết quả');
ok(Math.abs(r.probs.reduce((a, b) => a + b, 0) - 1) < 1e-9, 'tổng xác suất bằng 1');
ok(r.ranking.length === model.classes.length, `xếp hạng đủ ${model.classes.length} lớp`);
ok(r.ranking[0].p >= r.ranking[1].p, 'xếp hạng đã sắp giảm dần');
ok(r.topTokens.length > 0, `giải thích bằng phép che trả về ${r.topTokens.length} token`);
ok(r.nTokens > 0, `đếm được ${r.nTokens} token trong bình luận`);

// --- 3. Trường hợp biên -------------------------------------------------------
ok(predictCnn(model, '') === null, 'văn bản rỗng trả về null (không crash)');
ok(predictCnn(model, '!!! ??? ...') === null, 'văn bản không có token trả về null');
const zzz = predictCnn(model, 'zzzqqq wwwxxx yyyvvv');
ok(zzz !== null && zzz.nOov === 3, 'từ lạ được quy về <UNK> (3/3 ngoài từ điển)');

const long = predictCnn(model, Array(500).fill('dress').join(' '));
ok(long.truncated && long.nTokens === model.tokenizer.max_len,
   `văn bản quá dài bị cắt về ${model.tokenizer.max_len} token`);

// --- 4. Phép che thực sự đo được điều gì đó -----------------------------------
const demo = predictCnn(model, 'this dress is beautiful and the fabric feels great');
console.log(`\n  Ví dụ: "this dress is beautiful..." -> ${demo.labelVi} `
          + `(${(demo.probability * 100).toFixed(1)}%)`);
console.log('  Token quan trọng nhất: '
  + demo.topTokens.slice(0, 5)
      .map((t) => `${t.token} ${t.drop >= 0 ? '+' : '−'}${(Math.abs(t.drop) * 100).toFixed(1)}%`)
      .join(', '));
ok(demo.topTokens[0].drop > 0, 'token quan trọng nhất củng cố kết luận (drop > 0)');

console.log(failures === 0
  ? '\n=== TẤT CẢ PASS — giao diện React sẽ cho đúng kết quả notebook ===\n'
  : `\n=== ${failures} KIỂM TRA THẤT BẠI ===\n`);
process.exit(failures === 0 ? 0 : 1);
