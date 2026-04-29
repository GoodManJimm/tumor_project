"use client";
import { useState, ChangeEvent, useRef } from "react";
import { GoogleGenerativeAI } from "@google/generative-ai";

// ============================================================================
// CONFIGURATION
// ============================================================================
const GEMINI_API_KEY = "AIzaSyCcPkq9G9Gbyb4XHY770V7qb9h6gHMIRXs"; 
const BACKEND_URL = "http://localhost:8000/predict";

// ============================================================================
// TUMOR INFORMATION DATABASE
// ============================================================================
const tumorInfo: Record<
  string,
  {
    emoji: string;
    severity: string;
    severityColor: string;
    description: string;
    symptoms: string[];
    recommendations: string[];
  }
> = {
  Meningioma: {
    emoji: "🧠",
    severity: "Usually Benign (WHO Grade I–II)",
    severityColor: "#15803d",
    description:
      "Meningiomas are tumors that arise from the meninges — the protective membranes surrounding the brain and spinal cord. They are the most common type of primary brain tumor and are typically slow-growing. The majority of meningiomas are benign (non-cancerous), though a small percentage can be atypical or malignant.",
    symptoms: [
      "Persistent headaches that worsen over time",
      "Vision changes such as blurriness or double vision",
      "Hearing loss or ringing in the ears",
      "Memory difficulties or cognitive changes",
      "Weakness in the arms or legs",
      "Seizures in some cases",
    ],
    recommendations: [
      "Consult a neurologist or neurosurgeon for a comprehensive evaluation",
      "Regular MRI monitoring may be recommended if the tumor is small and asymptomatic",
      "Surgical removal is the primary treatment for symptomatic or growing tumors",
      "Radiation therapy may be considered for tumors that cannot be fully removed",
      "Maintain regular follow-up appointments to monitor for recurrence",
      "Seek a second medical opinion before making treatment decisions",
    ],
  },
  Glioma: {
    emoji: "⚠️",
    severity: "Variable (WHO Grade I–IV)",
    severityColor: "#dc2626",
    description:
      "Gliomas originate from glial cells, which are the supportive cells of the central nervous system. They are the most common type of malignant brain tumor and can vary widely in aggressiveness — from slow-growing low-grade gliomas to highly aggressive glioblastomas (Grade IV). Early detection and treatment planning are critical for improving patient outcomes.",
    symptoms: [
      "Severe and persistent headaches, often worse in the morning",
      "Nausea and vomiting unrelated to other conditions",
      "Seizures, which may be the first noticeable symptom",
      "Personality or behavioural changes",
      "Speech difficulties or language comprehension problems",
      "Progressive weakness or numbness on one side of the body",
    ],
    recommendations: [
      "Seek immediate consultation with a neuro-oncologist",
      "A biopsy or advanced imaging (MRI with contrast) is typically needed to determine the tumor grade",
      "Treatment often involves a combination of surgery, radiation therapy, and chemotherapy",
      "Genetic and molecular testing (e.g., IDH mutation, MGMT methylation) can help guide treatment",
      "Consider enrolling in clinical trials for access to newer therapies",
      "Engage palliative care early for symptom management and quality of life support",
    ],
  },
  "Pituitary Tumor": {
    emoji: "🔬",
    severity: "Usually Benign (Adenoma)",
    severityColor: "#b45309",
    description:
      "Pituitary tumors develop in the pituitary gland, a small pea-sized gland located at the base of the brain. This gland plays a vital role in regulating hormones that control growth, metabolism, reproduction, and stress response. Most pituitary tumors are benign adenomas, but they can cause significant health issues by disrupting hormone production or pressing on nearby structures such as the optic nerves.",
    symptoms: [
      "Unexplained weight gain or loss",
      "Changes in menstrual cycles or sexual function",
      "Excessive thirst and frequent urination",
      "Visual field loss, particularly peripheral vision",
      "Fatigue, weakness, and low energy levels",
      "Abnormal growth of hands, feet, or facial features (acromegaly)",
    ],
    recommendations: [
      "Consult an endocrinologist for hormonal evaluation and blood tests",
      "An MRI focused on the pituitary region is the standard diagnostic imaging",
      "Medication can effectively shrink certain types of pituitary tumors (e.g., prolactinomas)",
      "Transsphenoidal surgery (through the nose) is a common and minimally invasive treatment",
      "Hormone replacement therapy may be needed after treatment",
      "Regular long-term monitoring of hormone levels and tumor size is essential",
    ],
  },
  Normal: {
    emoji: "✅",
    severity: "Healthy / No Tumor Detected",
    severityColor: "#059669",
    description:
      "The model has classified this scan as 'Normal', indicating the absence of detectable tumors. The brain tissue, ventricles, and overall anatomical structures appear healthy and free from the pathological signs associated with Meningioma, Glioma, or Pituitary tumors. This is the baseline classification for a healthy control scan.",
    symptoms: [
      "No tumor-related symptoms indicated by this scan.",
      "Patient is expected to have normal neurological structure.",
    ],
    recommendations: [
      "No immediate oncological intervention is required based on this scan.",
      "Continue to maintain a healthy lifestyle and routine medical check-ups.",
      "Note: If the patient is experiencing severe neurological symptoms (like chronic headaches or seizures), consult a doctor regardless of these results, as there are other conditions a tumor-specific AI model does not screen for.",
    ],
  },
};

// ============================================================================
// GEMINI IMAGE VALIDATION (QUOTA-SAFE VERSION)
// ============================================================================
async function validateWithGemini(file: File): Promise<{
  isValid: boolean;
  message: string;
}> {
  try {
    const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash-lite" });

    // Convert file to base64
    const base64 = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });

    const imageParts = [
      {
        inlineData: {
          data: base64.split(",")[1],
          mimeType: file.type,
        },
      },
    ];

    const prompt = `Is this image a brain MRI scan or brain CT scan? Answer ONLY "yes" or "no". Nothing else.`;

    const result = await model.generateContent([prompt, ...imageParts]);
    const response = await result.response;
    const text = response.text().trim().toLowerCase();

    console.log("Gemini response:", text);

    if (text.includes("yes")) {
      return {
        isValid: true,
        message:
          "✅ Brain MRI scan confirmed. Proceeding with classification...",
      };
    } else {
      return {
        isValid: false,
        message:
          "❌ This image does not appear to be a brain MRI or CT scan. Please upload a valid brain MRI image.",
      };
    }
  } catch (error: any) {
    console.error("Gemini validation error:", error);

    // If quota exceeded (429) or rate limited, skip validation and let image through
    const errorMessage = error?.message || "";
    if (
      errorMessage.includes("429") ||
      errorMessage.includes("quota") ||
      errorMessage.includes("rate")
    ) {
      return {
        isValid: true,
        message:
          "⚠️ Image validation skipped (API quota exceeded). Proceeding with classification — please ensure this is a brain MRI scan.",
      };
    }

    // For other errors (bad API key, network issues), also skip with warning
    return {
      isValid: true,
      message:
        "⚠️ Image validation temporarily unavailable. Proceeding with classification — please ensure this is a brain MRI scan.",
    };
  }
}

// ============================================================================
// MAIN PAGE COMPONENT
// ============================================================================
export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>("");
  const [result, setResult] = useState<string>("");
  const [confidence, setConfidence] = useState<string>("");
  const [allProbs, setAllProbs] = useState<Record<string, string>>({});
  const [pipeline, setPipeline] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [validationStatus, setValidationStatus] = useState<
    "idle" | "validating" | "valid" | "invalid" | "skipped"
  >("idle");
  const [validationMessage, setValidationMessage] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setResult("");
      setConfidence("");
      setAllProbs({});
      setError("");
      setValidationStatus("idle");
      setValidationMessage("");

      if (selectedFile.name.toLowerCase().endsWith(".mat")) {
        setPreview("");
      } else {
        setPreview(URL.createObjectURL(selectedFile));
      }
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult("");
    setValidationStatus("idle");
    setValidationMessage("");

    // ====================================================
    // STEP 1: Gemini validation (skip for .mat files)
    // ====================================================
    const isMatFile = file.name.toLowerCase().endsWith(".mat");

    if (!isMatFile) {
      setValidationStatus("validating");
      setValidationMessage(
        "🔍 Verifying image with Gemini AI... Checking if this is a valid brain MRI scan."
      );

      const validation = await validateWithGemini(file);

      if (!validation.isValid) {
        setValidationStatus("invalid");
        setValidationMessage(validation.message);
        setLoading(false);
        return; // STOP — do not send to backend
      }

      // Check if validation was skipped due to quota
      if (validation.message.includes("⚠️")) {
        setValidationStatus("skipped");
      } else {
        setValidationStatus("valid");
      }
      setValidationMessage(validation.message);
    } else {
      setValidationStatus("valid");
      setValidationMessage(
        "✅ .mat file detected — raw MRI data format, skipping image validation."
      );
    }

    // ====================================================
    // STEP 2: Send to backend for tumor classification
    // ====================================================
    setValidationMessage(
      (prev) => prev + "\n⏳ Running tumor classification..."
    );

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(BACKEND_URL, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (data.status === "success") {
        setResult(data.prediction);
        setConfidence(data.confidence);
        setAllProbs(data.all_probabilities || {});
        setPipeline(data.pipeline_used || "");
      } else {
        setError(data.error || "Unknown error occurred.");
      }
    } catch {
      setError(
        "Unable to connect to the backend server. Please ensure the server is running on http://localhost:8000."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview("");
    setResult("");
    setConfidence("");
    setAllProbs({});
    setPipeline("");
    setError("");
    setValidationStatus("idle");
    setValidationMessage("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const currentTumor = result ? tumorInfo[result] : null;

  // Helper to get validation section colors
  const getValidationColors = () => {
    switch (validationStatus) {
      case "validating":
        return { bg: "#eff6ff", border: "#bfdbfe", text: "#1d4ed8" };
      case "valid":
        return { bg: "#f0fdf4", border: "#bbf7d0", text: "#15803d" };
      case "skipped":
        return { bg: "#fffbeb", border: "#fde68a", text: "#92400e" };
      case "invalid":
        return { bg: "#fef2f2", border: "#fecaca", text: "#dc2626" };
      default:
        return { bg: "#ffffff", border: "#e5e7eb", text: "#000000" };
    }
  };

  const validationColors = getValidationColors();

  return (
    <div style={styles.pageWrapper}>
      {/* ============ HEADER ============ */}
      <header style={styles.header}>
        <div style={styles.headerContent}>
          <h1 style={styles.title}>🧠 Brain Tumor Classification System</h1>
          <p style={styles.subtitle}>
            AI-Powered MRI Analysis using DenseNet-121 Deep Learning
          </p>
        </div>
      </header>

      <main style={styles.main}>
        {/* ============ ABOUT SECTION ============ */}
        <section style={styles.aboutSection}>
          <h2 style={styles.sectionTitle}>About This System</h2>
          <p style={styles.aboutText}>
            This web application uses a deep learning model trained on brain MRI
            scans to classify imaging data into four categories:{" "}
            <strong>Meningioma</strong>, <strong>Glioma</strong>,{" "}
            <strong>Pituitary Tumor</strong>, and <strong>Normal</strong> (Healthy). 
            The system is built on a DenseNet-121 convolutional neural network, 
            fine-tuned using transfer learning from ImageNet, and optimised 
            specifically for medical brain imaging data.
          </p>
          <div style={styles.featureGrid}>
            <div style={styles.featureCard}>
              <span style={styles.featureIcon}>📤</span>
              <h3 style={styles.featureTitle}>Upload</h3>
              <p style={styles.featureDesc}>
                Upload an MRI brain scan in <strong>.mat</strong> (recommended)
                or standard image format (PNG/JPG)
              </p>
            </div>
            <div style={styles.featureCard}>
              <span style={styles.featureIcon}>🛡️</span>
              <h3 style={styles.featureTitle}>Validate</h3>
              <p style={styles.featureDesc}>
                Gemini AI verifies the uploaded image is a genuine brain MRI scan
                before processing
              </p>
            </div>
            <div style={styles.featureCard}>
              <span style={styles.featureIcon}>⚙️</span>
              <h3 style={styles.featureTitle}>Preprocess</h3>
              <p style={styles.featureDesc}>
                Automatic Gaussian smoothing (σ=1) and dynamic thresholding to
                isolate brain tissue from background noise
              </p>
            </div>
            <div style={styles.featureCard}>
              <span style={styles.featureIcon}>🤖</span>
              <h3 style={styles.featureTitle}>Classify</h3>
              <p style={styles.featureDesc}>
                DenseNet-121 analyses the preprocessed scan and predicts the
                diagnostic class with a confidence score
              </p>
            </div>
          </div>
        </section>

        {/* ============ UPLOAD SECTION ============ */}
        <section style={styles.uploadSection}>
          <h2 style={styles.sectionTitle}>Upload MRI Scan</h2>

          <div
            style={styles.dropZone}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              onChange={handleFileChange}
              accept=".mat,.png,.jpg,.jpeg,.bmp,.tiff"
              style={{ display: "none" }}
            />

            {preview ? (
              <img src={preview} alt="MRI Preview" style={styles.previewImg} />
            ) : file ? (
              <div style={styles.matFileIndicator}>
                <span style={{ fontSize: "48px" }}>📁</span>
                <p style={styles.matFileName}>{file.name}</p>
                <p style={styles.matFileHint}>
                  .mat file selected (recommended format)
                </p>
              </div>
            ) : (
              <div style={styles.dropContent}>
                <span style={{ fontSize: "48px", opacity: 0.4 }}>🖼️</span>
                <p style={styles.dropText}>Click to select an MRI scan</p>
                <p style={styles.dropHint}>
                  Supported: .mat (recommended), .png, .jpg, .jpeg
                </p>
              </div>
            )}
          </div>

          <div style={styles.buttonGroup}>
            <button
              onClick={handleUpload}
              disabled={loading || !file}
              style={{
                ...styles.primaryButton,
                opacity: loading || !file ? 0.5 : 1,
                cursor: loading || !file ? "not-allowed" : "pointer",
              }}
            >
              {loading
                ? validationStatus === "validating"
                  ? "🔍 Validating Image..."
                  : "⏳ Classifying..."
                : "🔍 Classify Scan"}
            </button>

            {file && (
              <button onClick={handleReset} style={styles.secondaryButton}>
                ↩ Reset
              </button>
            )}
          </div>
        </section>

        {/* ============ VALIDATION STATUS ============ */}
        {validationStatus !== "idle" && (
          <section
            style={{
              ...styles.validationSection,
              backgroundColor: validationColors.bg,
              borderColor: validationColors.border,
            }}
          >
            {validationStatus === "validating" && (
              <div style={styles.validationLoading}>
                <div style={styles.spinnerWrapper}>
                  <div style={styles.spinner} />
                </div>
                <p
                  style={{
                    ...styles.validationText,
                    color: validationColors.text,
                  }}
                >
                  {validationMessage}
                </p>
              </div>
            )}
            {(validationStatus === "valid" ||
              validationStatus === "skipped") && (
              <p
                style={{
                  ...styles.validationText,
                  color: validationColors.text,
                  whiteSpace: "pre-line",
                }}
              >
                {validationMessage}
              </p>
            )}
            {validationStatus === "invalid" && (
              <div>
                <p
                  style={{
                    ...styles.validationText,
                    color: validationColors.text,
                  }}
                >
                  {validationMessage}
                </p>
                <div style={styles.invalidHelpBox}>
                  <p style={styles.invalidHelpTitle}>
                    What images are accepted?
                  </p>
                  <ul style={styles.invalidHelpList}>
                    <li>Brain MRI scans (axial, sagittal, or coronal views)</li>
                    <li>Brain CT scan slices</li>
                    <li>
                      Grayscale medical imaging output from MRI/CT scanners
                    </li>
                    <li>
                      .mat files from medical imaging datasets (always accepted)
                    </li>
                  </ul>
                  <p style={styles.invalidHelpTitle}>
                    What images are rejected?
                  </p>
                  <ul style={styles.invalidHelpList}>
                    <li>
                      Diagrams, flowcharts, DFDs, UML, or any schematic drawings
                    </li>
                    <li>Photos of people, animals, objects, or scenes</li>
                    <li>Screenshots, documents, charts, or presentations</li>
                    <li>Illustrations or artistic renderings of brains</li>
                    <li>X-rays of non-brain body parts</li>
                  </ul>
                </div>
              </div>
            )}
          </section>
        )}

        {/* ============ ERROR DISPLAY ============ */}
        {error && (
          <section style={styles.errorSection}>
            <p style={styles.errorText}>❌ {error}</p>
          </section>
        )}

        {/* ============ RESULT SECTION ============ */}
        {result && currentTumor && (
          <section style={styles.resultSection}>
            {/* Prediction Header */}
            <div style={styles.resultHeader}>
              <span style={{ fontSize: "40px" }}>{currentTumor.emoji}</span>
              <div>
                <h2 style={styles.resultTitle}>
                  Classification Result: {result}
                </h2>
                <p style={styles.resultConfidence}>
                  Confidence: <strong>{confidence}</strong>
                </p>
                <p
                  style={{
                    ...styles.resultSeverity,
                    color: currentTumor.severityColor,
                  }}
                >
                  {currentTumor.severity}
                </p>
              </div>
            </div>

            {/* Probability Breakdown */}
            {Object.keys(allProbs).length > 0 && (
              <div style={styles.probSection}>
                <h3 style={styles.subSectionTitle}>Probability Breakdown</h3>
                <div style={styles.probGrid}>
                  {Object.entries(allProbs).map(([cls, prob]) => {
                    const percentage = parseFloat(prob);
                    const isTop = cls === result;
                    return (
                      <div
                        key={cls}
                        style={{
                          ...styles.probCard,
                          borderColor: isTop ? "#0ea5e9" : "#e5e7eb",
                          backgroundColor: isTop ? "#f0f9ff" : "#ffffff",
                        }}
                      >
                        <p style={styles.probLabel}>{cls}</p>
                        <div style={styles.probBarBg}>
                          <div
                            style={{
                              ...styles.probBarFill,
                              width: `${Math.max(percentage, 2)}%`,
                              backgroundColor: isTop ? "#0ea5e9" : "#cbd5e1",
                            }}
                          />
                        </div>
                        <p
                          style={{
                            ...styles.probValue,
                            color: isTop ? "#0369a1" : "#64748b",
                          }}
                        >
                          {prob}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Pipeline Info */}
            {pipeline && (
              <div style={styles.pipelineInfo}>
                <p style={styles.pipelineText}>
                  📎 Preprocessing pipeline used: <strong>{pipeline}</strong>
                </p>
              </div>
            )}

            {/* Tumor Description */}
            <div style={styles.infoBlock}>
              <h3 style={styles.subSectionTitle}>📖 What is {result}?</h3>
              <p style={styles.infoText}>{currentTumor.description}</p>
            </div>

            {/* Symptoms */}
            <div style={styles.infoBlock}>
              <h3 style={styles.subSectionTitle}>🩺 Common Symptoms / Indicators</h3>
              <ul style={styles.infoList}>
                {currentTumor.symptoms.map((s, i) => (
                  <li key={i} style={styles.infoListItem}>
                    {s}
                  </li>
                ))}
              </ul>
            </div>

            {/* Recommendations */}
            <div style={styles.infoBlock}>
              <h3 style={styles.subSectionTitle}>💡 Recommended Next Steps</h3>
              <ul style={styles.infoList}>
                {currentTumor.recommendations.map((r, i) => (
                  <li key={i} style={styles.infoListItem}>
                    {r}
                  </li>
                ))}
              </ul>
            </div>

            {/* Medical Disclaimer */}
            <div style={styles.disclaimer}>
              <p style={styles.disclaimerText}>
                <strong>⚠️ Medical Disclaimer:</strong> This system is a
                research tool designed to assist in preliminary screening only.
                It is <strong>not</strong> a substitute for professional medical
                diagnosis. The classification results should be reviewed and
                confirmed by a qualified healthcare professional. Always consult
                a doctor for medical advice, diagnosis, and treatment.
              </p>
            </div>
          </section>
        )}
      </main>

      {/* ============ FOOTER ============ */}
      <footer style={styles.footer}>
        <p style={styles.footerText}>
          Brain Tumor Classification System — Final Year Project
        </p>
        <p style={styles.footerSubtext}>
          Built with DenseNet-121 · Gemini AI · Transfer Learning · FastAPI ·
          Next.js
        </p>
      </footer>
    </div>
  );
}

// ============================================================================
// STYLES
// ============================================================================
const styles: Record<string, React.CSSProperties> = {
  pageWrapper: {
    minHeight: "100vh",
    backgroundColor: "#f8fafc",
    display: "flex",
    flexDirection: "column",
  },
  header: {
    background: "linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%)",
    padding: "40px 20px",
    textAlign: "center",
  },
  headerContent: {
    maxWidth: "800px",
    margin: "0 auto",
  },
  title: {
    color: "#ffffff",
    fontSize: "32px",
    fontWeight: 700,
    margin: 0,
  },
  subtitle: {
    color: "#94a3b8",
    fontSize: "16px",
    marginTop: "8px",
  },
  main: {
    maxWidth: "900px",
    margin: "0 auto",
    padding: "30px 20px",
    flex: 1,
    width: "100%",
    boxSizing: "border-box",
  },
  aboutSection: {
    backgroundColor: "#ffffff",
    borderRadius: "16px",
    padding: "30px",
    marginBottom: "24px",
    boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
  },
  sectionTitle: {
    fontSize: "22px",
    fontWeight: 700,
    color: "#1e293b",
    marginTop: 0,
    marginBottom: "12px",
  },
  aboutText: {
    color: "#475569",
    lineHeight: 1.7,
    fontSize: "15px",
  },
  featureGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
    gap: "16px",
    marginTop: "20px",
  },
  featureCard: {
    backgroundColor: "#f8fafc",
    borderRadius: "12px",
    padding: "20px",
    textAlign: "center",
    border: "1px solid #e2e8f0",
  },
  featureIcon: {
    fontSize: "32px",
  },
  featureTitle: {
    fontSize: "15px",
    fontWeight: 700,
    color: "#1e293b",
    margin: "8px 0 4px",
  },
  featureDesc: {
    fontSize: "13px",
    color: "#64748b",
    lineHeight: 1.5,
    margin: 0,
  },
  uploadSection: {
    backgroundColor: "#ffffff",
    borderRadius: "16px",
    padding: "30px",
    marginBottom: "24px",
    boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
    textAlign: "center",
  },
  dropZone: {
    border: "2px dashed #cbd5e1",
    borderRadius: "12px",
    padding: "40px 20px",
    cursor: "pointer",
    transition: "border-color 0.2s",
    backgroundColor: "#fafbfc",
    marginBottom: "20px",
  },
  dropContent: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "8px",
  },
  dropText: {
    fontSize: "16px",
    color: "#475569",
    fontWeight: 600,
    margin: 0,
  },
  dropHint: {
    fontSize: "13px",
    color: "#94a3b8",
    margin: 0,
  },
  previewImg: {
    maxWidth: "280px",
    maxHeight: "280px",
    borderRadius: "12px",
    objectFit: "contain",
  },
  matFileIndicator: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "8px",
  },
  matFileName: {
    fontSize: "16px",
    fontWeight: 600,
    color: "#1e293b",
    margin: 0,
  },
  matFileHint: {
    fontSize: "13px",
    color: "#22c55e",
    margin: 0,
    fontWeight: 500,
  },
  buttonGroup: {
    display: "flex",
    justifyContent: "center",
    gap: "12px",
  },
  primaryButton: {
    padding: "14px 32px",
    fontSize: "16px",
    fontWeight: 700,
    color: "#ffffff",
    backgroundColor: "#0ea5e9",
    border: "none",
    borderRadius: "10px",
    transition: "all 0.2s",
  },
  secondaryButton: {
    padding: "14px 24px",
    fontSize: "16px",
    fontWeight: 600,
    color: "#475569",
    backgroundColor: "#f1f5f9",
    border: "1px solid #e2e8f0",
    borderRadius: "10px",
    cursor: "pointer",
  },
  validationSection: {
    borderRadius: "12px",
    padding: "16px 20px",
    marginBottom: "24px",
    border: "1px solid",
  },
  validationLoading: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
  },
  validationText: {
    margin: 0,
    fontSize: "15px",
    fontWeight: 600,
  },
  spinnerWrapper: {
    flexShrink: 0,
  },
  spinner: {
    width: "20px",
    height: "20px",
    border: "3px solid #bfdbfe",
    borderTopColor: "#2563eb",
    borderRadius: "50%",
    animation: "spin 0.8s linear infinite",
  },
  invalidHelpBox: {
    marginTop: "12px",
    padding: "12px 16px",
    backgroundColor: "#fff5f5",
    borderRadius: "8px",
  },
  invalidHelpTitle: {
    fontSize: "13px",
    fontWeight: 700,
    color: "#991b1b",
    margin: "8px 0 4px 0",
  },
  invalidHelpList: {
    paddingLeft: "18px",
    margin: "0 0 8px 0",
    fontSize: "13px",
    color: "#7f1d1d",
    lineHeight: 1.6,
  },
  errorSection: {
    backgroundColor: "#fef2f2",
    border: "1px solid #fecaca",
    borderRadius: "12px",
    padding: "16px 20px",
    marginBottom: "24px",
  },
  errorText: {
    color: "#dc2626",
    margin: 0,
    fontSize: "15px",
  },
  resultSection: {
    backgroundColor: "#ffffff",
    borderRadius: "16px",
    padding: "30px",
    marginBottom: "24px",
    boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
  },
  resultHeader: {
    display: "flex",
    alignItems: "center",
    gap: "16px",
    paddingBottom: "20px",
    borderBottom: "1px solid #e2e8f0",
    marginBottom: "24px",
  },
  resultTitle: {
    fontSize: "24px",
    fontWeight: 700,
    color: "#0f172a",
    margin: 0,
  },
  resultConfidence: {
    fontSize: "15px",
    color: "#475569",
    margin: "4px 0",
  },
  resultSeverity: {
    fontSize: "14px",
    fontWeight: 700,
    margin: 0,
  },
  probSection: {
    marginBottom: "24px",
  },
  subSectionTitle: {
    fontSize: "17px",
    fontWeight: 700,
    color: "#1e293b",
    marginBottom: "12px",
  },
  probGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
    gap: "12px",
  },
  probCard: {
    borderRadius: "10px",
    padding: "14px",
    border: "2px solid",
  },
  probLabel: {
    fontSize: "14px",
    fontWeight: 600,
    color: "#334155",
    margin: "0 0 8px 0",
  },
  probBarBg: {
    height: "8px",
    backgroundColor: "#f1f5f9",
    borderRadius: "4px",
    overflow: "hidden",
  },
  probBarFill: {
    height: "100%",
    borderRadius: "4px",
    transition: "width 0.6s ease",
  },
  probValue: {
    fontSize: "18px",
    fontWeight: 700,
    marginTop: "8px",
    marginBottom: 0,
  },
  pipelineInfo: {
    backgroundColor: "#f8fafc",
    borderRadius: "8px",
    padding: "10px 16px",
    marginBottom: "24px",
  },
  pipelineText: {
    fontSize: "13px",
    color: "#64748b",
    margin: 0,
  },
  infoBlock: {
    marginBottom: "24px",
  },
  infoText: {
    color: "#475569",
    lineHeight: 1.7,
    fontSize: "15px",
  },
  infoList: {
    paddingLeft: "20px",
    margin: 0,
  },
  infoListItem: {
    color: "#475569",
    lineHeight: 1.8,
    fontSize: "15px",
  },
  disclaimer: {
    backgroundColor: "#fffbeb",
    border: "1px solid #fde68a",
    borderRadius: "10px",
    padding: "16px 20px",
    marginTop: "8px",
  },
  disclaimerText: {
    color: "#92400e",
    fontSize: "13px",
    lineHeight: 1.7,
    margin: 0,
  },
  footer: {
    backgroundColor: "#0f172a",
    padding: "24px 20px",
    textAlign: "center",
  },
  footerText: {
    color: "#94a3b8",
    fontSize: "14px",
    margin: 0,
  },
  footerSubtext: {
    color: "#475569",
    fontSize: "12px",
    marginTop: "4px",
  },
};