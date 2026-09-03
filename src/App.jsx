import { useState, useEffect, useRef } from "react";
import { loadModel, predict } from "./model/infer.js";

const DIVISIONS = ["General", "General Petite", "Initmates"];
const DEPARTMENTS = ["Tops", "Dresses", "Bottoms", "Intimate", "Jackets", "Trend"];
const CLASSES = [
  "Dresses", "Knits", "Blouses", "Sweaters", "Pants", "Jeans", "Fine gauge", "Skirts",
  "Jackets", "Lounge", "Swim", "Outerwear", "Shorts", "Sleep", "Legwear", "Intimates",
  "Layering", "Trend", "Casual bottoms", "Chemises",
];

const initialForm = {
  Age: 35,
  Rating: 5,
  "Positive Feedback Count": 2,
  "Division Name": "General",
  "Department Name": "Dresses",
  "Class Name": "Dresses",
  "Review Text": "This dress is absolutely beautiful, fits perfectly and the fabric feels great. Highly recommend!",
};

const SAMPLES = [
  {
    label: "Đánh giá tích cực",
    form: initialForm,
  },
  {
    label: "Đánh giá tiêu cực",
    form: {
      Age: 42,
      Rating: 2,
      "Positive Feedback Count": 0,
      "Division Name": "General",
      "Department Name": "Tops",
      "Class Name": "Blouses",
      "Review Text": "The material is cheap and see through, it runs way too small and the color looks nothing like the picture. Very disappointed, returning it.",
    },
  },
  {
    label: "Đánh giá lưỡng lự",
    form: {
      Age: 29,
      Rating: 3,
      "Positive Feedback Count": 1,
      "Division Name": "General Petite",
      "Department Name": "Bottoms",
      "Class Name": "Jeans",
      "Review Text": "Nice denim and the fit is okay, but the length was a bit short for me and the price feels high for what you get.",
    },
  },
];

export default function App() {
  const [form, setForm] = useState(initialForm);
  const [model, setModel] = useState(null);
  const [modelError, setModelError] = useState("");
  const [result, setResult] = useState(null);

  const handleChange = (name, value) => setForm((prev) => ({ ...prev, [name]: value }));

  // The whole model is a 84 KB JSON of coefficients, so it is fetched once from this
  // site's own /model.json — there is no prediction server to call.
  useEffect(() => {
    let cancelled = false;
    loadModel()
      .then((m) => !cancelled && setModel(m))
      .catch((err) => !cancelled && setModelError(err.message));
    return () => {
      cancelled = true;
    };
  }, []);

  const runPredict = (source = form, loaded = model) => {
    if (!loaded) return;
    setResult(predict(loaded, source));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    runPredict();
  };

  const applySample = (sample) => {
    setForm(sample.form);
    runPredict(sample.form);
  };

  // Lets a screenshot/demo run land straight on a result: /?autodemo
  const didAutoDemo = useRef(false);
  useEffect(() => {
    if (model && !didAutoDemo.current && new URLSearchParams(window.location.search).has("autodemo")) {
      didAutoDemo.current = true;
      runPredict(form, model);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model]);

  const status = modelError ? "error" : !model ? "loading" : result ? "done" : "idle";

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>🛍️ Customer Interest Discovery</h1>
        <p>Dự đoán khách hàng có giới thiệu (recommend) sản phẩm không, từ hồ sơ + bình luận — chạy hoàn toàn trong trình duyệt</p>
      </header>

      <div className="card">
        <div className="sample-row">
          <span className="sample-label">Thử nhanh:</span>
          {SAMPLES.map((s) => (
            <button key={s.label} type="button" className="sample-btn" onClick={() => applySample(s)} disabled={!model}>
              {s.label}
            </button>
          ))}
        </div>

        <div className="layout">
          <form onSubmit={handleSubmit}>
            <div className="field-grid">
              <div className="field">
                <label htmlFor="Age">Tuổi khách hàng</label>
                <input
                  id="Age" type="number" min={10} max={100}
                  value={form.Age} onChange={(e) => handleChange("Age", e.target.value)} required
                />
              </div>
              <div className="field">
                <label htmlFor="Rating">Rating (số sao, 1–5)</label>
                <input
                  id="Rating" type="number" min={1} max={5}
                  value={form.Rating} onChange={(e) => handleChange("Rating", e.target.value)} required
                />
              </div>
              <div className="field">
                <label htmlFor="pfc">Positive Feedback Count</label>
                <input
                  id="pfc" type="number" min={0} max={1000}
                  value={form["Positive Feedback Count"]}
                  onChange={(e) => handleChange("Positive Feedback Count", e.target.value)} required
                />
              </div>
              <div className="field">
                <label htmlFor="division">Division Name</label>
                <select id="division" value={form["Division Name"]} onChange={(e) => handleChange("Division Name", e.target.value)}>
                  {DIVISIONS.map((d) => <option key={d} value={d}>{d}</option>)}
                </select>
              </div>
              <div className="field">
                <label htmlFor="department">Department Name</label>
                <select id="department" value={form["Department Name"]} onChange={(e) => handleChange("Department Name", e.target.value)}>
                  {DEPARTMENTS.map((d) => <option key={d} value={d}>{d}</option>)}
                </select>
              </div>
              <div className="field">
                <label htmlFor="class">Class Name</label>
                <select id="class" value={form["Class Name"]} onChange={(e) => handleChange("Class Name", e.target.value)}>
                  {CLASSES.map((d) => <option key={d} value={d}>{d}</option>)}
                </select>
              </div>
              <div className="field span-2">
                <label htmlFor="review">Review Text (bình luận của khách hàng)</label>
                <textarea
                  id="review" rows={4}
                  value={form["Review Text"]}
                  onChange={(e) => handleChange("Review Text", e.target.value)}
                  placeholder="Nhập nội dung đánh giá sản phẩm..."
                />
              </div>
            </div>
            <button className="submit-btn" type="submit" disabled={!model}>
              {model ? "Dự đoán mức độ hài lòng (Predict)" : "Đang tải model..."}
            </button>
          </form>

          <div className={`result-panel ${status === "idle" || status === "loading" ? "idle" : ""}`}>
            {status === "loading" && <p>Đang tải model.json (~84 KB)...</p>}
            {status === "error" && <div className="error-box">{modelError}</div>}
            {status === "idle" && <p>Kết quả dự đoán sẽ hiển thị ở đây sau khi bạn nhấn "Dự đoán".</p>}
            {status === "done" && result && (
              <>
                <span className={`result-badge ${result.recommended === 1 ? "positive" : "negative"}`}>
                  {result.recommended === 1 ? "✅ Recommended (khách sẽ giới thiệu)" : "❌ Not recommended"}
                </span>
                <div>
                  <div className="confidence-bar-track">
                    <div className="confidence-bar-fill" style={{ width: `${Math.round(result.confidence * 100)}%` }} />
                  </div>
                  <div className="result-meta">Độ tin cậy (confidence): {(result.confidence * 100).toFixed(1)}%</div>
                </div>

                {result.topTerms.length > 0 && (
                  <div className="terms">
                    <div className="result-meta">Từ/cụm từ ảnh hưởng mạnh nhất trong bình luận:</div>
                    <div className="term-chips">
                      {result.topTerms.map((t) => (
                        <span key={t.term} className={`term-chip ${t.contribution >= 0 ? "pos" : "neg"}`}>
                          {t.term}
                          <em>{t.contribution >= 0 ? "+" : "−"}{Math.abs(t.contribution).toFixed(2)}</em>
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                <div className="result-meta">
                  Logistic Regression (tabular + TF-IDF 1–2 gram), 2032 chiều · test F1 ≈ 0.965 · ROC-AUC ≈ 0.979
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      <footer className="app-footer">
        Không có backend — model được nạp từ <code>/model.json</code> và suy luận chạy bằng JavaScript ngay trên máy bạn.
        Dữ liệu bạn nhập không rời khỏi trình duyệt.
      </footer>
    </div>
  );
}
