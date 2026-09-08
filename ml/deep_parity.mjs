/**
 * Parity check: the JavaScript inference in src/model/deepInfer.js must agree with the
 * NumPy model that produced public/model_deep.json.
 *
 * The notebook writes ml/deep_parity_samples.json — raw form rows straight out of the
 * held-out test set, each with the probability vector the NumPy network produced. This
 * script replays them through the browser code path and fails loudly on any drift.
 *
 *   node ml/deep_parity.mjs
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");

// deepInfer.js imports from infer.js and uses import.meta.env only inside loadDeepModel,
// which we never call here — the bundle is read straight off disk instead.
const { prepareDeepModel, predictDeep } = await import(
  new URL("../src/model/deepInfer.js", import.meta.url)
);

const bundle = JSON.parse(readFileSync(join(root, "public", "model_deep.json"), "utf8"));
const samples = JSON.parse(readFileSync(join(here, "deep_parity_samples.json"), "utf8"));
const model = prepareDeepModel(bundle);

let maxErr = 0;
let labelMismatches = 0;

for (const [i, sample] of samples.entries()) {
  const got = predictDeep(model, sample.input);
  const want = sample.expected_proba;

  for (let c = 0; c < want.length; c++) {
    maxErr = Math.max(maxErr, Math.abs(got.probs[c] - want[c]));
  }

  const wantBest = want.indexOf(Math.max(...want));
  if (got.classIndex !== wantBest) {
    labelMismatches += 1;
    console.error(
      `  ✗ mẫu #${i}: JS dự đoán "${model.classes[got.classIndex]}" ` +
        `nhưng NumPy dự đoán "${model.classes[wantBest]}"`,
    );
  }
}

// The reference probabilities come from the full-precision NumPy model, while
// model_deep.json stores weights rounded to 5 decimals to keep the bundle at ~740 KB.
// That rounding alone moves probabilities by a few times 1e-5 — the notebook measures
// the same magnitude — so the tolerance is set just above it. Anything larger would be
// a real logic difference, not rounding.
const TOL = 5e-4;
console.log(`Đã kiểm tra ${samples.length} mẫu tham chiếu.`);
console.log(`Sai lệch xác suất lớn nhất : ${maxErr.toExponential(3)} (ngưỡng ${TOL})`);
console.log(`Số mẫu lệch nhãn           : ${labelMismatches}`);

if (maxErr > TOL || labelMismatches > 0) {
  console.error("✗ PARITY FAILED — code JS không khớp với mô hình NumPy.");
  process.exit(1);
}
console.log("✓ PARITY PASSED — suy luận trên trình duyệt trùng khớp với notebook.");
