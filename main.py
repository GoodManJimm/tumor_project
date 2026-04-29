import os
import uvicorn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import scipy.io as sio
import tempfile
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================================
# 1. CONFIGURATION & CLASS MAP
# ============================================================================
GAUSSIAN_SIGMA = 1
THRESHOLD_MEAN_WEIGHT = 0.75
THRESHOLD_STD_WEIGHT = 0.25

# Ensure these indices match training script's LabelEncoder
class_names = {
    0: "Meningioma",
    1: "Glioma",
    2: "Pituitary Tumor",
    3: "Normal"  # Added the 4th class
}

# ============================================================================
# 2. MODEL LOADING (Updated for 4 Classes)
# ============================================================================
def load_my_model(path):
    if not os.path.exists(path):
        print(f"❌ Cannot find model at: {path}")
        return None

    # Load DenseNet-121
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    
    # Updated to 4 output neurons
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 4)  # Changed from 3 to 4
    )

    try:
        state_dict = torch.load(path, map_location=device)
        # Handle cases where model is saved as a full dict or just weights
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"✅ 4-Class Model loaded successfully on: {device}")
        return model
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        return None

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")
model = load_my_model(MODEL_PATH)

# ============================================================================
# 3. SMART PREPROCESSING PIPELINE
# ============================================================================
model_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_mat_file(file_bytes):
    """Robustly handles both v7 (Healthy) and v7.3 (Figshare) .mat files."""
    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # Step 1: Smart Load Logic
        try:
            # Try Scipy (v7 files - IXI/Normal)
            mat_data = sio.loadmat(tmp_path)
            raw_img = mat_data['cjdata'][0, 0]['image'].astype(np.float32)
        except (NotImplementedError, ValueError):
            # Fallback to h5py (v7.3 files - Original Figshare)
            with h5py.File(tmp_path, 'r') as f:
                raw_img = np.array(f['cjdata']['image'], dtype=np.float32).T

        # Step 2: Gaussian Smoothing
        img_smooth = gaussian_filter(raw_img, sigma=GAUSSIAN_SIGMA)

        # Step 3: Dynamic Thresholding (Chapter 5 Innovation!)
        tissue_pixels = img_smooth[img_smooth > 0]
        if len(tissue_pixels) > 0:
            dynamic_threshold = (THRESHOLD_MEAN_WEIGHT * np.mean(tissue_pixels) +
                                 THRESHOLD_STD_WEIGHT * np.std(tissue_pixels))
            img_smooth[img_smooth < dynamic_threshold] = 0

        # Step 4: Normalisation (0-255)
        img_max, img_min = img_smooth.max(), img_smooth.min()
        if img_max - img_min > 0:
            img_norm = ((img_smooth - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_norm = np.zeros_like(img_smooth, dtype=np.uint8)

        # Step 5: Final Transform
        img_rgb = Image.fromarray(img_norm).convert('RGB')
        return model_transform(img_rgb).unsqueeze(0).to(device)

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def preprocess_standard_image(image_bytes):
    img_pil = Image.open(BytesIO(image_bytes)).convert('L')
    img_array = np.array(img_pil, dtype=np.float32)
    img_smooth = gaussian_filter(img_array, sigma=GAUSSIAN_SIGMA)
    
    tissue_pixels = img_smooth[img_smooth > 0]
    if len(tissue_pixels) > 0:
        dynamic_threshold = (THRESHOLD_MEAN_WEIGHT * np.mean(tissue_pixels) +
                             THRESHOLD_STD_WEIGHT * np.std(tissue_pixels))
        img_smooth[img_smooth < dynamic_threshold] = 0

    img_max, img_min = img_smooth.max(), img_smooth.min()
    img_norm = ((img_smooth - img_min) / (img_max - img_min) * 255).astype(np.uint8) if img_max > img_min else np.zeros_like(img_smooth, dtype=np.uint8)
    
    img_rgb = Image.fromarray(img_norm).convert('RGB')
    return model_transform(img_rgb).unsqueeze(0).to(device)

# ============================================================================
# 4. PREDICTION ENDPOINT
# ============================================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not initialised", "status": "error"}

    try:
        contents = await file.read()
        filename = file.filename.lower() if file.filename else ""

        if filename.endswith('.mat'):
            input_tensor = preprocess_mat_file(contents)
            pipeline = "4-Class MAT Pipeline (Smart Loader)"
        else:
            input_tensor = preprocess_standard_image(contents)
            pipeline = "Standard Image Pipeline (Approximate)"

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        return {
            "prediction": class_names.get(predicted_class.item(), "Unknown"),
            "confidence": f"{confidence.item() * 100:.2f}%",
            "all_probabilities": {
                class_names[i]: f"{probabilities[0][i].item() * 100:.2f}%"
                for i in range(len(class_names))
            },
            "pipeline_used": pipeline,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}

@app.get("/health")
async def health_check():
    return {"status": "running", "classes": 4, "device": str(device)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)