import { useState, useEffect, useRef } from "react";
import { loadModel, predict } from "./model/infer.js";
import { loadDeepModel, predictDeep } from "./model/deepInfer.js";
import { loadCnnModel, predictCnn } from "./model/cnnInfer.js";

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

/** Ba bài toán, ba mô hình — người dùng chọn bằng tab. */
const TASKS = {
  cnn: {
    title: "CNN 1D trên văn bản",
    subtitle: "Assignment 04 · chỉ dùng bình luận",
  },
  interest: {
    title: "Sở thích ngành hàng",
    subtitle: "Assignment 03 · Deep MLP 5 tầng",
  },
  recommend: {
    title: "Khách có giới thiệu?",
    subtitle: "Logistic Regression · nhị phân",
  },
};

const pct = (v) => `${(v * 100).toFixed(1)}%`;

export default function App() {
  const [form, setForm] = useState(initialForm);
  const [task, setTask] = useState("cnn");
  const [model, setModel] = useState(null);
  const [deepModel, setDeepModel] = useState(null);
  const [cnnModel, setCnnModel] = useState(null);
  const [modelError, setModelError] = useState("");
  const [result, setResult] = useState(null);
  const [deepResult, setDeepResult] = useState(null);
  const [cnnResult, setCnnResult] = useState(null);

  const handleChange = (name, value) => setForm((prev) => ({ ...prev, [name]: value }));

  // Cả ba bundle đều là JSON hệ số nằm cùng origin — không có server dự đoán nào cả.
  useEffect(() => {
    let cancelled = false;
    Promise.all([loadModel(), loadDeepModel(), loadCnnModel()])
      .then(([m, d, c]) => {
        if (cancelled) return;
        setModel(m);
        setDeepModel(d);
        setCnnModel(c);
      })
      .catch((err) => !cancelled && setModelError(err.message));
    return () => {
      cancelled = true;
    };
  }, []);

  const runPredict = (
    source = form,
    loaded = model,
    loadedDeep = deepModel,
    loadedCnn = cnnModel,
  ) => {
    if (loaded) setResult(predict(loaded, source));
    if (loadedDeep) setDeepResult(predictDeep(loadedDeep, source));
    // CNN chỉ nhận trường văn bản — không dùng bất kỳ đặc trưng bảng nào.
    if (loadedCnn) setCnnResult(predictCnn(loadedCnn, source["Review Text"]));
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
    if (model && deepModel && cnnModel && !didAutoDemo.current
        && new URLSearchParams(window.location.search).has("autodemo")) {
      didAutoDemo.current = true;
      runPredict(form, model, deepModel, cnnModel);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, deepModel, cnnModel]);

  const ready = model && deepModel && cnnModel;
  const shown = task === "cnn" ? cnnResult : task === "interest" ? deepResult : result;
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
            {task === "cnn" && (
              <p className="form-note">
                CNN 1D <strong>chỉ đọc ô Review Text</strong> — mọi trường còn lại bị bỏ qua hoàn toàn.
                Bình luận được tách token, tra từ điển {cnnModel?.tokenizer.vocab.length.toLocaleString("vi-VN")} từ,
                rồi cho kernel dài 3 trượt qua như một bộ dò n-gram.{" "}
                <strong>Department Name</strong> ở trên chỉ là đáp án đúng để bạn đối chiếu.
              </p>
            )}
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
                ? (task === "cnn" ? "Dự đoán bằng CNN 1D"
                  : task === "interest" ? "Dự đoán ngành hàng quan tâm"
                  : "Dự đoán mức độ hài lòng")
                : "Đang tải model..."}
            </button>
          </form>

          <div className={`result-panel ${status === "idle" || status === "loading" ? "idle" : ""}`}>
            {status === "loading" && <p>Đang tải model.json, model_deep.json và model_cnn.json (~1,5 MB)...</p>}
            {status === "error" && <div className="error-box">{modelError}</div>}
            {status === "idle" && <p>Kết quả dự đoán sẽ hiển thị ở đây sau khi bạn nhấn nút.</p>}

            {status === "done" && task === "cnn" && cnnResult && (
              <>
                <span className={`result-badge ${cnnResult.label === truth ? "positive" : "negative"}`}>
                  🧩 {cnnResult.labelVi}
                  {cnnResult.label === truth ? " · khớp nhãn thật" : ` · nhãn thật: ${truth}`}
                </span>

                <div className="class-bars">
                  {cnnResult.ranking.map((r) => (
                    <div key={r.label} className={`class-row ${r.i === cnnResult.classIndex ? "top" : ""}`}>
                      <span className="class-name">{r.labelVi}</span>
                      <div className="class-track">
                        <div className="class-fill" style={{ width: `${Math.max(1, r.p * 100)}%` }} />
                      </div>
                      <span className="class-pct">{pct(r.p)}</span>
                    </div>
                  ))}
                </div>

                <div className="result-meta">
                  Bình luận được tách thành <strong>{cnnResult.nTokens} token</strong>
                  {cnnResult.nOov > 0 && <> ({cnnResult.nOov} token ngoài từ điển → <code>&lt;UNK&gt;</code>)</>}
                  {cnnResult.truncated && <> · đã cắt về {cnnModel.tokenizer.max_len} token đầu</>}
                  . Á quân: <strong>{cnnResult.runnerUp.labelVi}</strong> ({pct(cnnResult.runnerUp.p)}).
                </div>

                {cnnResult.topTokens.length > 0 && (
                  <div className="terms">
                    <div className="result-meta">
                      Che từng token rồi đo xác suất lớp <strong>{cnnResult.labelVi}</strong> tụt bao nhiêu
                      — số dương nghĩa là từ đó <em>củng cố</em> kết luận:
                    </div>
                    <div className="term-chips">
                      {cnnResult.topTokens.map((t) => (
                        <span key={`${t.token}-${t.position}`}
                              className={`term-chip ${t.drop >= 0 ? "pos" : "neg"}`}>
                          {t.token}
                          <em>{t.drop >= 0 ? "+" : "−"}{(Math.abs(t.drop) * 100).toFixed(1)}%</em>
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                <div className="result-meta">
                  {cnnModel.model_name} · {cnnModel.n_params.toLocaleString("vi-VN")} tham số ·
                  test Macro-F1 {cnnModel.metrics.macro_f1.toFixed(3)}, balanced accuracy{" "}
                  {pct(cnnModel.metrics.balanced_accuracy)}, accuracy {pct(cnnModel.metrics.accuracy)}.
                  Huấn luyện {cnnModel.training.epochs_run} epoch bằng NumPy thuần trong{" "}
                  {cnnModel.training.numpy_seconds}s (PyTorch {cnnModel.training.pytorch_seconds}s,
                  TensorFlow {cnnModel.training.tensorflow_seconds}s cho cùng kiến trúc).
                </div>
              </>
            )}

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

        {task === "cnn" && cnnModel && (
          <div className="bench">
            <div className="bench-title">
              Đối sánh CNN 1D vs TF-IDF — phân loại 6 ngành hàng CHỈ từ văn bản (tập Test)
            </div>
            <table className="bench-table">
              <thead>
                <tr>
                  <th>Mô hình</th>
                  <th>Accuracy</th>
                  <th>Balanced Acc</th>
                  <th>Macro-F1</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(cnnModel.comparison)
                  .map(([name, m]) => ({ name, ...m, isCnn: name.startsWith("CNN") }))
                  .sort((a, b) => b.macro_f1 - a.macro_f1)
                  .map((r) => (
                    <tr key={r.name} className={r.isCnn ? "deep-row" : ""}>
                      <td>{r.isCnn ? "🧩 " : ""}{r.name}</td>
                      <td>{r.accuracy.toFixed(4)}</td>
                      <td>{r.balanced_accuracy.toFixed(4)}</td>
                      <td>{r.macro_f1.toFixed(4)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
            <p className="bench-note">
              <strong>Thứ tự từ có thật sự quan trọng không?</strong> Xáo trộn ngẫu nhiên thứ tự từ trong
              mỗi bình luận của tập test làm Macro-F1 của CNN tụt từ{" "}
              {cnnModel.word_order_test.cnn_f1_original.toFixed(4)} xuống{" "}
              {cnnModel.word_order_test.cnn_f1_shuffled.toFixed(4)}, trong khi mô hình túi từ
              (TF-IDF unigram) <strong>không đổi một chữ số nào</strong> (
              {cnnModel.word_order_test.bow_f1_original.toFixed(4)} →{" "}
              {cnnModel.word_order_test.bow_f1_shuffled.toFixed(4)}) — đúng như lý thuyết, vì nó không
              nhìn thấy thứ tự. Đây là bằng chứng trực tiếp rằng tích chập đã học được các mẫu{" "}
              <em>n-gram cục bộ</em>, thứ mà dữ liệu bảng ở Bài 1 và Bài 2 không hề có.
            </p>
            <p className="bench-note">
              Dù vậy, TF-IDF + Logistic Regression vẫn nhỉnh hơn: nhãn ở đây là <em>loại sản phẩm</em>,
              phần lớn được quyết định bởi sự có mặt của từ khoá (<code>dress</code>, <code>jeans</code>,{" "}
              <code>blouse</code>) hơn là bởi trật tự. CNN chỉ thật sự vượt trội khi bài toán phụ thuộc
              vào cụm từ — ví dụ phân tích cảm xúc, nơi <code>not good</code> phải khác <code>very good</code>.
            </p>
          </div>
        )}

        {task !== "cnn" && deepModel && (
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
        Không có backend — ba mô hình được nạp từ <code>/model.json</code>, <code>/model_deep.json</code> và{" "}
        <code>/model_cnn.json</code> (trọng số đẩy thẳng lên GitHub, Vercel phục vụ như tệp tĩnh),
        suy luận chạy bằng JavaScript ngay trên máy bạn. Dữ liệu bạn nhập không rời khỏi trình duyệt.
      </footer>
    </div>
  );
}
