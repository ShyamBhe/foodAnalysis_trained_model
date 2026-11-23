
import { useRef, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  function selectFile(selectedFile) {
    if (!selectedFile) return;

    if (!selectedFile.type.startsWith("image/")) {
      setError("Please select a valid image.");
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("Image must be smaller than 10 MB.");
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setPredictions([]);
    setError("");
  }

  function handleFileChange(event) {
    selectFile(event.target.files?.[0]);
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    selectFile(event.dataTransfer.files?.[0]);
  }

  async function identifyFood() {
    if (!file) {
      setError("Please upload a food image first.");
      return;
    }

    setLoading(true);
    setError("");
    setPredictions([]);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      setPredictions(data.predictions);
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to the AI server. Make sure FastAPI is running on port 8000."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  function resetImage() {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview("");
    setPredictions([]);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  const bestPrediction = predictions[0];

  return (
    <div className="app">
      <header className="navbar">
        <a className="brand" href="/">
          <span className="brand-icon">✳</span>
          <span>FoodVision<span className="brand-ai"> AI</span></span>
        </a>

        <div className="nav-right">
          <span className="model-status">
            <span className="status-dot"></span>
            AI model
          </span>
          <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer">
            API Docs ↗
          </a>
        </div>
      </header>

      <main className="main-content">
        <section className="hero">
          <span className="eyebrow">FOOD RECOGNITION POWERED BY AI</span>
          <h1>
            Discover what's
            <br />
            <span>on your plate.</span>
          </h1>
          <p>
            Upload a food photo and let our AI model identify
            your dish from 101 food categories.
          </p>
        </section>

        <section className="workspace">
          <div className="upload-panel">
            <div className="section-heading">
              <div>
                <span className="step-label">STEP 01</span>
                <h2>Upload your image</h2>
              </div>
              {file && (
                <button className="text-button" onClick={resetImage}>
                  Remove image
                </button>
              )}
            </div>

            {!preview ? (
              <div
                className={`dropzone ${dragging ? "dragging" : ""}`}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={handleDrop}
                onClick={() => inputRef.current?.click()}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    inputRef.current?.click();
                  }
                }}
              >
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={handleFileChange}
                  hidden
                />

                <div className="upload-icon">↑</div>
                <h3>Drop your food image here</h3>
                <p>or click to browse your files</p>
                <span className="file-hint">JPG, PNG or WEBP · Maximum 10 MB</span>
                <button
                  className="browse-button"
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation();
                    inputRef.current?.click();
                  }}
                >
                  Choose an image
                </button>
              </div>
            ) : (
              <div className="preview-container">
                <img src={preview} alt="Uploaded food" />
                <div className="preview-label">
                  <span className="status-dot"></span>
                  Image ready for analysis
                </div>
              </div>
            )}

            {file && (
              <div className="file-info">
                <span>▧</span>
                <div>
                  <strong>{file.name}</strong>
                  <small>{(file.size / 1024 / 1024).toFixed(2)} MB</small>
                </div>
              </div>
            )}

            {error && <div className="error-message">{error}</div>}

            <button
              className="predict-button"
              onClick={identifyFood}
              disabled={!file || loading}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Analyzing image...
                </>
              ) : (
                <>✦ Identify food with AI <span>→</span></>
              )}
            </button>

            <p className="privacy-note">
              Your image is processed by your FoodVision AI model.
            </p>
          </div>

          <div className="results-panel">
            <div className="section-heading">
              <div>
                <span className="step-label">STEP 02</span>
                <h2>AI predictions</h2>
              </div>
              {predictions.length > 0 && (
                <span className="result-count">TOP 5 RESULTS</span>
              )}
            </div>

            {predictions.length === 0 ? (
              <div className="empty-results">
                <div className="brain-icon">✳</div>
                <h3>Waiting for your image</h3>
                <p>
                  Your food recognition results will appear here
                  after the AI analyzes your image.
                </p>
                <div className="empty-features">
                  <span>✦ 101 food categories</span>
                  <span>✦ Deep learning</span>
                  <span>✦ Real-time inference</span>
                </div>
              </div>
            ) : (
              <div className="prediction-results">
                {bestPrediction && (
                  <div className="best-result">
                    <span className="best-label">MOST LIKELY MATCH</span>
                    <h3>{bestPrediction.food}</h3>
                    <strong>{bestPrediction.confidence.toFixed(2)}%</strong>
                    <span>Model confidence</span>
                  </div>
                )}

                <h3 className="other-results-title">Other predictions</h3>

                {predictions.slice(1).map((prediction, index) => (
                  <div className="prediction-item" key={prediction.food}>
                    <div className="prediction-top">
                      <span className="rank">0{index + 2}</span>
                      <strong>{prediction.food}</strong>
                      <span className="percentage">
                        {prediction.confidence.toFixed(2)}%
                      </span>
                    </div>
                    <div className="progress-track">
                      <div
                        className="progress-fill"
                        style={{ width: `${prediction.confidence}%` }}
                      ></div>
                    </div>
                  </div>
                ))}

                <p className="confidence-note">
                  Percentages represent the model's softmax scores across
                  its 101 categories, not guaranteed probabilities of correctness.
                </p>
              </div>
            )}
          </div>
        </section>

        <section className="info-strip">
          <div><strong>101</strong><span>Food categories</span></div>
          <div><strong>MobileNetV3</strong><span>Deep learning architecture</span></div>
          <div><strong>PyTorch</strong><span>AI inference engine</span></div>
          <div><strong>FastAPI</strong><span>Prediction API</span></div>
        </section>
      </main>

      <footer>
        <span>FoodVision AI</span>
        <span>Built with PyTorch, FastAPI & React</span>
      </footer>
    </div>
  );
}

export default App;
