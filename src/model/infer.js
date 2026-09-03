/**
 * Browser-side inference for the customer-behavior model.
 *
 * The training pipeline (notebook/customer_behavior.ipynb) is a Logistic Regression on
 * three concatenated blocks:
 *
 *   [ 3 numeric  ]  median-impute -> standard-scale
 *   [ 29 one-hot ]  most-frequent-impute -> OneHotEncoder(handle_unknown='ignore')
 *   [ 2000 tfidf ]  TextCleaner -> TfidfVectorizer(1-2 grams, english stop words, l2)
 *
 * All of that is arithmetic over constants, so instead of shipping a Python server we
 * export those constants to public/model.json (ml/export_model.py) and redo the exact
 * same steps here. The functions below mirror scikit-learn's implementation detail by
 * detail — see ml/parity_check.py, which asserts this file and the original .joblib
 * agree on every row of the raw dataset.
 */

// scikit-learn's default token_pattern is r"(?u)\b\w\w+\b": every maximal run of two or
// more word characters. \w in JS is ASCII-only, so spell the unicode class out.
const TOKEN_RE = /[\p{L}\p{N}\p{M}_]{2,}/gu;

/** TextCleaner from common.py: null -> "", collapse runs of whitespace, trim. */
export function cleanText(value) {
  if (value === null || value === undefined) return "";
  return String(value).split(/\s+/).filter(Boolean).join(" ");
}

/**
 * scikit-learn's word analyzer: lowercase, tokenize, drop stop words, then build the
 * n-grams. Stop words are removed *before* n-grams are formed (CountVectorizer's
 * _word_ngrams), so "fits perfectly the fabric" yields the bigram "fits perfectly",
 * never "perfectly the".
 */
export function analyze(text, { lowercase, ngramRange, stopWords }) {
  const source = lowercase ? text.toLowerCase() : text;
  const matched = source.match(TOKEN_RE) || [];
  const tokens = stopWords ? matched.filter((t) => !stopWords.has(t)) : matched;

  const [minN, maxN] = ngramRange;
  if (maxN === 1) return minN === 1 ? tokens : [];

  const grams = minN === 1 ? tokens.slice() : [];
  for (let n = Math.max(minN, 2); n <= Math.min(maxN, tokens.length); n += 1) {
    for (let i = 0; i + n <= tokens.length; i += 1) {
      grams.push(tokens.slice(i, i + n).join(" "));
    }
  }
  return grams;
}

/** Build the lookup tables once per loaded bundle instead of once per prediction. */
export function prepareModel(bundle) {
  const vocabIndex = new Map();
  bundle.text.vocab.forEach((term, i) => vocabIndex.set(term, i));

  const catIndex = bundle.categorical.categories.map((cats) => {
    const m = new Map();
    cats.forEach((c, i) => m.set(c, i));
    return m;
  });

  // Column offset of each one-hot block inside the categorical coefficient slice.
  const catOffsets = [];
  let running = 0;
  for (const cats of bundle.categorical.categories) {
    catOffsets.push(running);
    running += cats.length;
  }

  return {
    ...bundle,
    _vocabIndex: vocabIndex,
    _catIndex: catIndex,
    _catOffsets: catOffsets,
    _stopWords: new Set(bundle.text.stop_words),
  };
}

/**
 * The TF-IDF block's contribution to the decision value: counts -> count * idf ->
 * L2-normalise -> dot with the model coefficients. Only terms present in the review are
 * touched, so this is O(tokens), not O(2000).
 */
function textContribution(model, reviewText) {
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
  if (counts.size === 0) return { score: 0, terms: [] };

  let sumSq = 0;
  const weighted = [];
  for (const [col, count] of counts) {
    const w = count * model.text.idf[col];
    weighted.push([col, w]);
    sumSq += w * w;
  }
  const norm = Math.sqrt(sumSq);
  if (norm === 0) return { score: 0, terms: [] };

  let score = 0;
  const terms = [];
  for (const [col, w] of weighted) {
    const value = w / norm;
    const contribution = value * model.text.coef[col];
    score += contribution;
    terms.push({ term: model.text.vocab[col], contribution });
  }
  // Most influential words first — used by the UI to explain the prediction.
  terms.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  return { score, terms };
}

/**
 * Predict from one form row. `input` uses the same keys as the dataset columns, e.g.
 * { Age: 35, Rating: 5, "Positive Feedback Count": 2, "Division Name": "General", ... }.
 * Returns the same shape the FastAPI service used to return, plus `topTerms`.
 */
export function predict(model, input) {
  let z = model.intercept;

  model.numeric.names.forEach((name, i) => {
    const raw = input[name];
    const parsed = raw === "" || raw === null || raw === undefined ? NaN : Number(raw);
    const filled = Number.isFinite(parsed) ? parsed : model.numeric.fill[i];
    const scaled = (filled - model.numeric.mean[i]) / model.numeric.scale[i];
    z += scaled * model.numeric.coef[i];
  });

  model.categorical.names.forEach((name, i) => {
    const raw = input[name];
    const value = raw === "" || raw === null || raw === undefined ? model.categorical.fill[i] : String(raw);
    const pos = model._catIndex[i].get(value);
    // handle_unknown='ignore': an unseen category contributes an all-zero block.
    if (pos !== undefined) z += model.categorical.coef[model._catOffsets[i] + pos];
  });

  const text = textContribution(model, input[model.text.column]);
  z += text.score;

  const probaPositive = 1 / (1 + Math.exp(-z));
  const recommended = probaPositive >= 0.5 ? 1 : 0;

  return {
    interest: recommended === 1 ? "recommended" : "not_recommended",
    recommended,
    // confidence in the predicted class, matching the old API's rounding
    confidence: Math.round((recommended === 1 ? probaPositive : 1 - probaPositive) * 1e4) / 1e4,
    probability: probaPositive,
    decision: z,
    topTerms: text.terms.slice(0, 6),
  };
}

/** Fetch public/model.json (served from the same origin — no backend involved). */
export async function loadModel(url = `${import.meta.env.BASE_URL}model.json`) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Không tải được model.json (HTTP ${res.status})`);
  return prepareModel(await res.json());
}
