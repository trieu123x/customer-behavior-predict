/**
 * Browser-side inference for the 5-layer deep MLP (Assignment 03).
 *
 * The training pipeline (ml/notebook/customer_interest_deep.ipynb) builds a single
 * 606-dimensional input vector out of three blocks, in this exact order:
 *
 *   [   3 numeric ]  median-impute -> standard-scale
 *   [   3 one-hot ]  Division Name
 *   [ 600 tfidf   ]  TextCleaner -> TfidfVectorizer(1-2 grams, english stop words, l2)
 *
 * and pushes it through Dense(128) -> Dense(64) -> Dense(32) -> Dense(16) -> Dense(6),
 * ReLU on the four hidden layers and Softmax on the head. All of that is arithmetic
 * over constants, so the weights are exported to public/model_deep.json and the exact
 * same steps are redone here — see ml/deep_parity.mjs, which asserts this file agrees
 * with the NumPy model on every reference sample the notebook wrote out.
 *
 * Note the deliberate omissions: `Department Name` is the label, and `Class Name`
 * determines it one-to-one (Jeans -> Bottoms), so neither is an input.
 */
import { analyze, cleanText } from "./infer.js";

/** Build the lookup tables once per loaded bundle instead of once per prediction. */
export function prepareDeepModel(bundle) {
  const vocabIndex = new Map();
  bundle.text.vocab.forEach((term, i) => vocabIndex.set(term, i));
  return {
    ...bundle,
    _vocabIndex: vocabIndex,
    _stopWords: new Set(bundle.text.stop_words),
    _divIndex: new Map(bundle.division.categories.map((c, i) => [c, i])),
    _textOffset: bundle.numeric.names.length + bundle.division.categories.length,
  };
}

/**
 * TF-IDF for one review: count the n-grams that exist in the vocabulary, weight them
 * by IDF, then L2-normalise. Only terms actually present are touched, so this is
 * O(tokens) rather than O(600).
 */
function tfidfEntries(model, reviewText) {
  const grams = analyze(cleanText(reviewText), {
    lowercase: model.text.lowercase,
    ngramRange: model.text.ngram_range,
    stopWords: model._stopWords,
  });

  const counts = new Map();
  for (const g of grams) {
    const col = model._vocabIndex.get(g);
    if (col !== undefined) counts.set(col, (counts.get(col) || 0) + 1);
  }
  if (counts.size === 0) return [];

  let sumSq = 0;
  const weighted = [];
  for (const [col, count] of counts) {
    const w = count * model.text.idf[col];
    weighted.push([col, w]);
    sumSq += w * w;
  }
  const norm = Math.sqrt(sumSq);
  if (norm === 0) return [];
  return weighted.map(([col, w]) => [col, w / norm]);
}

/** Assemble the full 606-dim input vector in the order the network was trained on. */
export function buildFeatures(model, input) {
  const x = new Float64Array(model.architecture[0]);

  model.numeric.names.forEach((name, i) => {
    const raw = input[name];
    const parsed = raw === "" || raw === null || raw === undefined ? NaN : Number(raw);
    const filled = Number.isFinite(parsed) ? parsed : model.numeric.fill[i];
    x[i] = (filled - model.numeric.mean[i]) / model.numeric.scale[i];
  });

  const divRaw = input[model.division.name];
  const div = divRaw === "" || divRaw === null || divRaw === undefined
    ? model.division.fill : String(divRaw);
  const pos = model._divIndex.get(div);
  // An unseen division contributes an all-zero block, matching handle_unknown='ignore'.
  if (pos !== undefined) x[model.numeric.names.length + pos] = 1;

  const entries = tfidfEntries(model, input[model.text.column]);
  for (const [col, value] of entries) x[model._textOffset + col] = value;

  return { x, entries };
}

/** z = a·W + b, followed by ReLU on hidden layers and nothing on the output layer. */
function denseLayer(a, layer, relu) {
  const { W, b } = layer;
  const out = new Float64Array(b.length);
  for (let j = 0; j < b.length; j++) {
    let s = b[j];
    for (let i = 0; i < a.length; i++) s += a[i] * W[i][j];
    out[j] = relu && s < 0 ? 0 : s;
  }
  return out;
}

function softmax(z) {
  let max = -Infinity;
  for (const v of z) if (v > max) max = v;
  let sum = 0;
  const e = Array.from(z, (v) => {
    const t = Math.exp(v - max);
    sum += t;
    return t;
  });
  return e.map((t) => t / sum);
}

/** Forward pass keeping every pre-activation, so saliency can be back-propagated. */
function forwardWithCache(model, x) {
  const zs = [];
  let a = x;
  const L = model.layers.length;
  for (let i = 0; i < L; i++) {
    const isHidden = i < L - 1;
    const z = denseLayer(a, model.layers[i], false);
    zs.push(z);
    a = isHidden ? z.map((v) => (v > 0 ? v : 0)) : z;
  }
  return { logits: a, zs };
}

/**
 * d(logit_c)/d(x) by hand-rolled chain rule, minus the average over all classes.
 * The raw gradient carries a component shared by every class ("this text is a
 * clothing review at all"); subtracting the mean leaves what is specific to c —
 * the same normalisation Softmax performs on the logits.
 */
function saliency(model, zs, classId) {
  const L = model.layers.length;
  const nClasses = model.layers[L - 1].b.length;
  const head = model.layers[L - 1].W;                       // (16, nClasses)

  let g = new Float64Array(head.length);
  for (let i = 0; i < head.length; i++) {
    let mean = 0;
    for (let c = 0; c < nClasses; c++) mean += head[i][c];
    g[i] = head[i][classId] - mean / nClasses;
  }

  for (let l = L - 2; l >= 0; l--) {
    const z = zs[l];
    for (let i = 0; i < g.length; i++) if (z[i] <= 0) g[i] = 0;   // qua ReLU
    const W = model.layers[l].W;                                  // (nIn, nOut)
    const next = new Float64Array(W.length);
    for (let i = 0; i < W.length; i++) {
      let s = 0;
      const row = W[i];
      for (let j = 0; j < g.length; j++) s += row[j] * g[j];
      next[i] = s;
    }
    g = next;
  }
  return g;
}

/**
 * Predict the customer's interest category from one form row.
 * Returns the winning class plus the full probability vector and the n-grams that
 * pushed the decision towards it.
 */
export function predictDeep(model, input) {
  const { x, entries } = buildFeatures(model, input);
  const { logits, zs } = forwardWithCache(model, x);
  const probs = softmax(logits);

  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;

  const g = saliency(model, zs, best);
  const terms = entries
    .map(([col, value]) => ({
      term: model.text.vocab[col],
      contribution: value * g[model._textOffset + col],
    }))
    .filter((t) => t.contribution !== 0)
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

  const ranking = probs
    .map((p, i) => ({ label: model.classes[i], labelVi: model.class_labels_vi[i], p, i }))
    .sort((a, b) => b.p - a.p);

  return {
    classIndex: best,
    label: model.classes[best],
    labelVi: model.class_labels_vi[best],
    probability: probs[best],
    probs,
    ranking,
    // Top-2 accuracy is 0.93 vs 0.82 top-1, so the runner-up is genuinely useful.
    runnerUp: ranking[1],
    topTerms: terms.slice(0, 6),
  };
}

/** Fetch public/model_deep.json (same origin — no backend involved). */
export async function loadDeepModel(url = `${import.meta.env.BASE_URL}model_deep.json`) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Không tải được model_deep.json (HTTP ${res.status})`);
  return prepareDeepModel(await res.json());
}
