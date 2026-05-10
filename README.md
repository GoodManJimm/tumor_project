# Brain Tumor Classification AI System

An automated 4-class brain tumour classification and localisation system using a fine-tuned DenseNet-121 architecture with a dedicated segmentation head. The system classifies MRI scans into four categories and visually highlights the predicted tumour region directly on the scan.

## 🧠 What It Does

- **Classifies** brain MRI scans into one of four categories: Meningioma, Glioma, Pituitary Tumour, or Normal healthy brain
- **Localises** the predicted tumour region by generating a pixel-level red overlay on the MRI image (for positive tumour predictions)
- **Preprocesses** automatically using Gaussian filtering (σ=1) and a per-image dynamic threshold (T = 0.75μ + 0.25σ)
- **Validates** uploads using Gemini AI to reject non-MRI files before inference
- **Displays** the preprocessed MRI and tumour overlay side by side in the web interface

## 📊 Performance

| Dataset | Accuracy | F1-Score | Specificity | |
|---|---|---|---|---|
| Balanced Augmented (16,992 images) | 97.16% | 97.17% | 99.06% | ← **this is best_model.pth** |
| Imbalanced (9,860 images) | 99.05% | 97.47% | 99.74% | |

Validated using 5-fold cross-validation with StratifiedGroupKFold to prevent data leakage across augmented variants.

## 🌟 Features

- **4-Class Classification** — Meningioma, Glioma, Pituitary Tumour, Normal
- **Tumour Region Localisation** — segmentation head produces a visual red overlay showing predicted tumour location
- **Smart File Loader** — handles `.mat` v7 (SciPy) and v7.3 HDF5 (h5py) formats automatically
- **Dynamic Preprocessing** — per-image Gaussian smoothing + adaptive thresholding
- **Gemini AI Validation** — rejects non-MRI uploads before they reach the model
- **Full Confidence Breakdown** — returns probability scores for all four classes
- **REST API** — FastAPI backend with health check endpoint

## 📁 Project Structure

```
tumor_project/
├── main.py                    # FastAPI backend — DenseNetWithSegmentation inference
├── frontend/                  # Next.js web application
│   ├── app/
│   │   └── page.tsx          # Main UI — upload, results, overlay display
│   ├── public/
│   └── package.json
└── .gitignore
```

> ⚠️ **Model file not included in this repository** — see [Model Download](#-model-download) below.

## 📥 Model Download

The trained model file (`best_model.pth`) exceeds GitHub's 25MB file size limit and is hosted externally.

**Download link:** https://drive.google.com/file/d/1-P26MfSFhs3DrNVRjoDtodcmL4DZSbLb/view?usp=sharing

> This is the **balanced augmented model** trained on 16,992 images (2,832 originals × 6 augmentation factor, 708 images per class). Validated with 5-fold StratifiedGroupKFold CV. Mean F1-Score: 97.17%.

After downloading, place the file in the project root:

```
tumor_project/
└── best_model.pth    ← place here
```

The model is a `DenseNetWithSegmentation` architecture — it will **not** work with a standard DenseNet-121 checkpoint. Make sure you download the correct file from the link above.

## 🚀 Quick Start

### Prerequisites

- Python ≥ 3.9
- Node.js ≥ 18
- pip

### Backend Setup (FastAPI)

```bash
cd tumor_project

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn torch torchvision numpy scipy pillow h5py python-multipart opencv-python matplotlib

# Start the server
python main.py
```

Backend runs at `http://localhost:8000`

### Frontend Setup (Next.js)

```bash
# Open a new terminal
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:3000`

## 🔬 Technical Pipeline

### Preprocessing Steps

1. **Smart Loading** — detects file format automatically (HDF5 v7.3 via h5py, legacy v5 via SciPy, or standard image formats)
2. **HDF5 Transpose** — applies `.T` to HDF5 arrays to correct column-major storage order
3. **Gaussian Smoothing** — applies σ=1 blur to reduce scanner noise
4. **Dynamic Thresholding** — computes `T = 0.75 × μ + 0.25 × σ` per image to isolate brain tissue from background
5. **Normalisation** — scales to [0–255] uint8
6. **Model Inference** — DenseNetWithSegmentation produces classification probabilities + segmentation mask simultaneously

### Model Architecture

The model uses a **DenseNetWithSegmentation** architecture — not a standard DenseNet-121. It has two output heads:

- **Classification head** — `Linear(1024→512) → ReLU → Dropout(0.3) → Linear(512→4)` — outputs tumour type + confidence
- **Segmentation head** — series of Conv2d + Upsample layers progressively upsampling `7×7 → 224×224` — outputs pixel-level tumour probability map

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Upload MRI and get prediction + overlay images |
| `GET` | `/health` | Server health check |

**Response Example:**

```json
{
  "prediction": "Glioma",
  "confidence": "96.83%",
  "all_probabilities": {
    "Meningioma": "1.23%",
    "Glioma": "96.83%",
    "Pituitary Tumor": "0.12%",
    "Normal": "1.82%"
  },
  "has_tumor": true,
  "pipeline_used": "MAT Pipeline",
  "preprocessed_image": "<base64 PNG>",
  "tumor_overlay": "<base64 PNG>",
  "status": "success"
}
```

The `preprocessed_image` and `tumor_overlay` fields are base64-encoded PNG strings rendered directly by the frontend:

```jsx
<img src={`data:image/png;base64,${preprocessedImage}`} />
<img src={`data:image/png;base64,${tumorOverlay}`} />
```

## ⚠️ Known Limitations

- **Tumour overlay may not render for all positive predictions** — the segmentation head was trained on ground truth masks available for only 3,064 of the original images. For tumour presentations that differ significantly from the training distribution, the overlay may be faint or absent. This does not affect the classification output.
- **Normal class performance on imbalanced data was inflated** — the original 9,860-image dataset was 68.9% Normal, making classification trivially easy due to contrast modality differences between sources. The balanced 16,992-image result (97.17% F1) is the more clinically honest figure.
- **Non-contrast vs contrast MRI** — Normal images (IXI/Kaggle) are non-contrast; tumour images (Figshare) are contrast-enhanced. Future work should validate on contrast-enhanced healthy scans.

## 🛠️ Troubleshooting

### Model Not Loading

```bash
# Check file exists in project root
ls -l best_model.pth

# If you see a KeyError or size mismatch, you may have a plain DenseNet-121 checkpoint
# This project requires DenseNetWithSegmentation — download the correct model from the link above
```

### Missing Modules

```bash
pip install opencv-python        # for cv2
pip install python-multipart     # for FastAPI file uploads
pip install h5py                 # for HDF5 .mat files
```

### Full Clean Reinstall

```bash
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn torch torchvision numpy scipy pillow h5py python-multipart opencv-python matplotlib
```

### GPU / MPS Issues (Mac)

If you encounter MPS-related errors, force CPU in `main.py`:

```python
device = torch.device("cpu")
```

### CORS Errors

Ensure both servers are running simultaneously — backend on port 8000 and frontend on port 3000. The backend already allows all origins via `CORSMiddleware`.

## 📖 Citation

If you use this work, please cite:

```
Jimmy, "Automated Brain Tumour Classification and Localisation Using DenseNet-121
with Dynamic Thresholding and Segmentation Head"
```
