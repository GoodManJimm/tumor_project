# Brain Tumor Classification System

A comprehensive medical image analysis platform for classifying brain tumors (Meningioma, Glioma, Pituitary Tumor, Normal) using deep learning.

---

## 🏥 Overview

This system combines:
- **Backend API**: FastAPI service with trained DenseNet-121 model for image classification
- **Frontend UI**: Next.js application for user interface

---

## 🚀 Quick Start

### Step 1: Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Mac/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install Python dependencies
pip install fastapi uvicorn torch numpy scipy pillow h5py
```

### Step 2: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### Step 3: Start the Application

**Option A: Start Backend Only**
```bash
python main.py
```
Backend will run at: `http://localhost:8000`

**Option B: Start Full Application**
Open **TWO** terminal windows:

**Terminal 1 - Backend (API Server)**
```bash
python main.py
```
Or for production:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend (Web Interface)**
```bash
cd frontend
npm run dev
```
Frontend will run at: `http://localhost:3000`

---

## 📁 Project Structure

```
tumour_project/
├── main.py                 # FastAPI backend server
├── best_model_4.pth       # Trained model weights (4-class classifier)
├── venv/                  # Python virtual environment
├── frontend/              # Next.js web interface
│   ├── app/
│   ├── public/
│   └── ...
└── README.md             # This file
```

---

## 🌐 Available URLs

When both servers are running:

- **API Documentation**: http://localhost:8000/docs
- **Backend API**: http://localhost:8000
- **Web Interface**: http://localhost:3000
- **Health Check**: http://localhost:8000/health

---

## 📊 Model Capabilities

- **Classes**: 4 tumor types + Normal tissue
- **Supported Formats**: 
  - MATLAB `.mat` files (v7, v7.3)
  - Standard images (`.png`, `.jpg`)
- **Architecture**: DenseNet-121 with custom head
- **Smart Preprocessing**: Dynamic thresholding + Gaussian smoothing

---

## 🔍 API Usage

### Test Health Check
```bash
curl http://localhost:8000/health
```

### Upload Image for Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@<your_image_file>"
```

---

## ⚙️ Configuration

Edit `main.py` if you need to change:
- Model path: `MODEL_PATH`
- Preprocessing parameters: `GAUSSIAN_SIGMA`, thresholds

---

## 🛠️ Troubleshooting

### Model Not Found
Ensure `best_model_4.pth` exists in the project root.

### Device Detection
The system auto-detects:
- MPS (Apple Silicon)
- GPU (CUDA)
- CPU (default fallback)


## 📄 License

For research and educational purposes.