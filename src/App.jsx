import { useState, useEffect, useRef } from "react";
import { loadModel, predict } from "./model/infer.js";
import { loadDeepModel, predictDeep } from "./model/deepInfer.js";

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
    label: "Váy đầm · tích cực",
    form: initialForm,
  },
  {
    label: "Áo · tiêu cực",
    form: {
      Age: 42,
      Rating: 2,
      "Positive Feedback Count": 0,
      "Division Name": "General",
      "Department Name": "Tops",
      "Class Name": "Blouses",
      "Review Text": "The material of this blouse is cheap and see through, it runs way too small and the color looks nothing like the picture. Very disappointed, returning it.",
    },
  },
  {
    label: "Quần jeans · lưỡng lự",
    form: {
      Age: 29,
      Rating: 3,
      "Positive Feedback Count": 1,
      "Division Name": "General Petite",
      "Department Name": "Bottoms",
      "Class Name": "Jeans",
      "Review Text": "Nice denim and the fit through the hips is okay, but the leg length was a bit short for me and the price feels high for what you get.",
    },
  },
  {
    label: "Áo khoác · tích cực",
    form: {
      Age: 51,
      Rating: 5,
      "Positive Feedback Count": 6,
      "Division Name": "General",
      "Department Name": "Jackets",
      "Class Name": "Outerwear",
      "Review Text": "This coat is warm without being bulky, the wool feels substantial and it layers beautifully over a sweater. I wore it all winter.",
    },
  },
];

/** Hai bài toán, hai mô hình — người dùng chọn bằng tab. */
const TASKS = {
  interest: {
    title: "Sở thích ngành hàng",
    subtitle: "Deep MLP 5 tầng · phân loại 6 lớp",
  },
  recommend: {
    title: "Khách có giới thiệu?",
    subtitle: "Logistic Regression · phân loại nhị phân",
  },
};

const pct = (v) => `${(v * 100).toFixed(1)}%`;

export default function App() {
  const [form, setForm] = useState(initialForm);
  const [task, setTask] = useState("interest");
  const [model, setModel] = useState(null);
  const [deepModel, setDeepModel] = useState(null);
  const [modelError, setModelError] = useState("");
  const [result, setResult] = useState(null);
  const [deepResult, setDeepResult] = useState(null);

  const handleChange = (name, value) => setForm((prev) => ({ ...prev, [name]: value }));

  // Cả hai bundle đều là JSON hệ số nằm cùng origin — không có server dự đoán nào cả.
  useEffect(() => {
    let cancelled = false;
    Promise.all([loadModel(), loadDeepModel()])
      .then(([m, d]) => {
        if (cancelled) return;
        setModel(m);
        setDeepModel(d);
      })
      .catch((err) => !cancelled && setModelError(err.message));
    return () => {
      cancelled = true;
    };
  }, []);

  const runPredict = (source = form, loaded = model, loadedDeep = deepModel) => {
    if (loaded) setResult(predict(loaded, source));
    if (loadedDeep) setDeepResult(predictDeep(loadedDeep, source));
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
    if (model && deepModel && !didAutoDemo.current && new URLSearchParams(window.location.search).has("autodemo")) {
      didAutoDemo.current = true;
      runPredict(form, model, deepModel);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, deepModel]);

  const ready = model && deepModel;
  const shown = task === "interest" ? deepResult : result;
  const status = modelError ? "error" : !ready ? "loading" : shown ? "done" : "idle";

  const deepMetrics = deepModel?.metrics;
  const truth = form["Department Name"];

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>🛍️ Customer Interest Discovery</h1>
        <p>
          Từ hồ sơ và bình luận của khách, suy ra <strong>ngành hàng họ quan tâm</strong> bằng mạng
          nơ-ron sâu 5 tầng tự cài đặt bằng NumPy — chạy hoàn toàn trong trình duyệt, không backend.
        </p>
      </header>

      <div className="card">
        <div className="task-tabs" role="tablist">
          {Object.entries(TASKS).map(([key, t]) => (
            <button
              key={key}
              type="button"
              role="tab"
              aria-selected={task === key}
              className={`task-tab ${task === key ? "active" : ""}`}
              onClick={() => setTask(key)}
            >
              {t.title}
              <small>{t.subtitle}</small>
            </button>
          ))}
        </div>

        <div className="sample-row">
          <span className="sample-label">Thử nhanh:</span>
          {SAMPLES.map((s) => (
            <button key={s.label} type="button" className="sample-btn" onClick={() => applySample(s)} disabled={!ready}>
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
                <label htmlFor="department">
                  Department Name
                  {task === "interest" && <span className="field-tag">nhãn cần dự đoán</span>}
                </label>
                <select id="department" value={form["Department Name"]} onChange={(e) => handleChange("Department Name", e.target.value)}>
                  {DEPARTMENTS.map((d) => <option key={d} value={d}>{d}</option>)}
                </select>
              </div>
              <div className="field">
                <label htmlFor="class">
                  Class Name
                  {task === "interest" && <span className="field-tag leak">rò rỉ nhãn — bị loại</span>}
                </label>
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
            {task === "interest" && (
              <p className="form-note">
                Mạng chỉ nhận <strong>Age, Rating, Positive Feedback Count, Division Name</strong> và
                TF-IDF của bình luận. <strong>Department Name</strong> ở trên là đáp án đúng để bạn
                đối chiếu, còn <strong>Class Name</strong> xác định Department một–một (Jeans → Bottoms)
                nên bắt buộc phải loại khỏi đầu vào — giữ lại là rò rỉ nhãn.
              </p>
            )}
            <button className="submit-btn" type="submit" disabled={!ready}>
              {ready
                ? (task === "interest" ? "Dự đoán ngành hàng quan tâm" : "Dự đoán mức độ hài lòng")
                : "Đang tải model..."}
            </button>
          </form>

          <div className={`result-panel ${status === "idle" || status === "loading" ? "idle" : ""}`}>
            {status === "loading" && <p>Đang tải model.json (~85 KB) và model_deep.json (~740 KB)...</p>}
            {status === "error" && <div className="error-box">{modelError}</div>}
            {status === "idle" && <p>Kết quả dự đoán sẽ hiển thị ở đây sau khi bạn nhấn nút.</p>}

            {status === "done" && task === "interest" && deepResult && (
              <>
                <span className={`result-badge ${deepResult.label === truth ? "positive" : "negative"}`}>
                  🎯 {deepResult.labelVi}
                  {deepResult.label === truth ? " · khớp nhãn thật" : ` · nhãn thật: ${truth}`}
                </span>

                <div className="class-bars">
                  {deepResult.ranking.map((r) => (
                    <div key={r.label} className={`class-row ${r.i === deepResult.classIndex ? "top" : ""}`}>
                      <span className="class-name">{r.labelVi}</span>
                      <div className="class-track">
                        <div className="class-fill" style={{ width: `${Math.max(1, r.p * 100)}%` }} />
                      </div>
                      <span className="class-pct">{pct(r.p)}</span>
                    </div>
                  ))}
                </div>

                <div className="result-meta">
                  Á quân: <strong>{deepResult.runnerUp.labelVi}</strong> ({pct(deepResult.runnerUp.p)}) — trên
                  tập test, Top-2 accuracy đạt {pct(deepMetrics.top_2_acc)} so với Top-1 {pct(deepMetrics.accuracy)},
                  nên gợi ý hai ngành hàng thay vì một là chiến lược đáng cân nhắc.
                </div>

                {deepResult.topTerms.length > 0 && (
                  <div className="terms">
                    <div className="result-meta">
                      N-gram đẩy dự đoán về phía <strong>{deepResult.labelVi}</strong> (gradient × TF-IDF):
                    </div>
                    <div className="term-chips">
                      {deepResult.topTerms.map((t) => (
                        <span key={t.term} className={`term-chip ${t.contribution >= 0 ? "pos" : "neg"}`}>
                          {t.term}
                          <em>{t.contribution >= 0 ? "+" : "−"}{Math.abs(t.contribution).toFixed(2)}</em>
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                <div className="result-meta">
                  {deepModel.model_name} · {deepModel.architecture.join(" → ")} ·{" "}
                  {deepModel.n_params.toLocaleString("vi-VN")} tham số · test accuracy{" "}
                  {pct(deepMetrics.accuracy)}, balanced accuracy {pct(deepMetrics.balanced_acc)},
                  macro-F1 {deepMetrics.macro_f1.toFixed(3)} (đoán lớp đa số chỉ đạt{" "}
                  {pct(deepModel.majority_baseline)}).
                </div>
              </>
            )}

            {status === "done" && task === "recommend" && result && (
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

        {deepModel && (
          <div className="bench">
            <div className="bench-title">Đối sánh Classical ML vs Deep Learning — phân loại 6 ngành hàng (tập Test)</div>
            <table className="bench-table">
              <thead>
                <tr>
                  <th>Mô hình</th>
                  <th>Accuracy</th>
                  <th>Balanced Acc</th>
                  <th>Macro-F1</th>
                  <th>Top-2 Acc</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ...Object.entries(deepModel.baseline_metrics).map(([name, m]) => ({
                    name,
                    deep: false,
                    acc: m.Accuracy,
                    bal: m["Balanced Acc"],
                    f1: m["Macro-F1"],
                    top2: m["Top-2 Acc"],
                  })),
                  {
                    name: "Deep MLP-5 (NumPy from scratch)",
                    deep: true,
                    acc: deepMetrics.accuracy,
                    bal: deepMetrics.balanced_acc,
                    f1: deepMetrics.macro_f1,
                    top2: deepMetrics.top_2_acc,
                  },
                ]
                  .sort((a, b) => b.f1 - a.f1)
                  .map((r) => (
                    <tr key={r.name} className={r.deep ? "deep-row" : ""}>
                      <td>{r.deep ? "🧠 " : ""}{r.name}</td>
                      <td>{r.acc.toFixed(4)}</td>
                      <td>{r.bal.toFixed(4)}</td>
                      <td>{r.f1.toFixed(4)}</td>
                      <td>{r.top2.toFixed(4)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
            <p className="bench-note">
              Balanced accuracy và macro-F1 mới là thước đo đúng ở đây: dữ liệu lệch 88:1 giữa lớp lớn
              nhất (Tops, 44.6%) và lớp hiếm nhất (Trend, 0.5%), nên accuracy thuần luôn thiên vị mô hình
              chỉ đoán lớp đa số. Mạng sâu được huấn luyện với trọng số lớp theo phương án{" "}
              <code>{deepModel.training.class_weight_scheme}</code>, chọn bằng macro-F1 trên tập validation.
            </p>
          </div>
        )}
      </div>

      <footer className="app-footer">
        Không có backend — hai mô hình được nạp từ <code>/model.json</code> và <code>/model_deep.json</code>,
        suy luận chạy bằng JavaScript ngay trên máy bạn. Dữ liệu bạn nhập không rời khỏi trình duyệt.
      </footer>
    </div>
  );
}
