 Tumor Classification AI System

A sophisticated deep learning-based tumor classification system that analyzes medical imaging data (MRI scans) to identify four types of brain conditions with 98% accuracy using DenseNet-121.

## 🌟 Features

- **4-Class Classification**: Accurately distinguishes between:
  - Meningioma
  - Glioma
  - Pituitary Tumor
  - Normal Brain Tissue

- **Advanced Preprocessing Pipeline**:
  - Smart file loading (.mat v7/v7.3 and standard formats)
  - Gaussian smoothing for noise reduction
  - Dynamic thresholding for brain tissue isolation
  
- **REST API**: FastAPI-based backend with health checks
- **Modern Web Interface**: Next.js frontend with real-time visualization

## 📁 Project Structure

```
tumor_project/
├── main.py                    # FastAPI backend server
├── best_model.pth            # Pre-trained DenseNet-121 model
├── frontend/                 # Next.js web application
│   ├── app/                 # React components
│   ├── public/              # Static assets
│   └── package.json        # Frontend dependencies
└── .gitignore               # Git ignore rules
```

## 🚀 Quick Start

### Prerequisites
- Python ≥ 3.9
- Node.js ≥ 18
- pip (package manager)
- best_model.pth file in project root

### Installation & Setup

#### Backend (FastAPI)
```bash
# Create and activate virtual environment
cd tumor_project
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Python dependencies
pip install fastapi uvicorn torch torchvision numpy scipy pillow h5py python-multipart

# Start the API server
python main.py
```

The backend will start at `http://localhost:8000`

#### Frontend (Next.js)
```bash
# Terminal 2 - Open a new terminal
cd frontend
npm install
npm run dev
```

The frontend will start at `http://localhost:5173`

## 🔬 Technical Pipeline

### Preprocessing Steps
1. **Smart Loading**: Automatically detects .mat files (v7 and v7.3 formats) and standard image formats
2. **Gaussian Smoothing**: Applies blur (σ=1) to reduce sensor noise
3. **Dynamic Thresholding**: Uses formula `0.75 × mean + 0.25 × std` to isolate brain tissue from background
4. **Normalization**: Scales images to [0-255] range for optimal model inference
5. **Model Inference**: DenseNet-121 processes image and outputs class probabilities

### API Endpoints
- `POST /predict` - Upload image and get prediction
- `GET /health` - Health check endpoint

**Request Format:**
```
Upload: Image file (.mat, .jpg, .png)
Content-Type: multipart/form-data
```

**Response Example:**
```json
{
  "prediction": "Glioma",
  "confidence": "98.45%",
  "all_probabilities": {
    "Meningioma": "1.23%",
    "Glioma": "98.45%",
    "Pituitary Tumor": "0.12%",
    "Normal": "0.20%"
  },
  "pipeline_used": "4-Class MAT Pipeline (Smart Loader)",
  "status": "success"
}
```

## 🛠️ Troubleshooting

### Model Not Initialized
```bash
# Check if best_model.pth exists
ls -l best_model.pth

# Verify path in main.py
# Should be: MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")
```

### Module Import Errors
```bash
# Clean rebuild
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn torch torchvision numpy scipy pillow h5py python-multipart
```

### GPU/MPS Issues (Mac)
Edit `main.py` and change:
```python
# Instead of this:
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Use this:
device = torch.device("cpu")
CORS Errors
