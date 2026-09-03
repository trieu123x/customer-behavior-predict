// Node half of parity_check.py: score the dumped rows with the browser inference code
// and report the largest disagreement with scikit-learn.
// Usage: node ml/parity.mjs <model.json> <cases.json> <infer.js>
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [, , modelPath, casesPath, inferPath] = process.argv;
const { prepareModel, predict } = await import(pathToFileURL(path.resolve(inferPath)).href);

const model = prepareModel(JSON.parse(fs.readFileSync(modelPath, "utf8")));
const cases = JSON.parse(fs.readFileSync(casesPath, "utf8"));

let maxDiff = 0;
let labelMismatch = 0;
const examples = [];

for (const c of cases) {
  const out = predict(model, c.input);
  const diff = Math.abs(out.probability - c.proba);
  if (diff > maxDiff) maxDiff = diff;
  if (out.probability >= 0.5 !== c.proba >= 0.5) {
    labelMismatch += 1;
    if (examples.length < 5) examples.push({ js: out.probability, py: c.proba, input: c.input });
  }
}

console.log(`javascript scored  : ${cases.length} rows`);
console.log(`max |p_js - p_py|  : ${maxDiff.toExponential(3)}`);
console.log(`label mismatches   : ${labelMismatch}`);
if (examples.length) console.log(JSON.stringify(examples, null, 2));

const ok = labelMismatch === 0 && maxDiff < 1e-9;
console.log(ok ? "PARITY OK" : "PARITY FAILED");
process.exit(ok ? 0 : 1);
