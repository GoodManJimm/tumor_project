import os
import uvicorn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import scipy.io as sio
import tempfile
import base64
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as mpl_cm

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
GAUSSIAN_SIGMA        = 1
THRESHOLD_MEAN_WEIGHT = 0.75
THRESHOLD_STD_WEIGHT  = 0.25

class_names = {
    0: "Meningioma",
    1: "Glioma",
    2: "Pituitary Tumor",
    3: "Normal"
}

# ============================================================================
# 2. DENSENET-121 WITH SEGMENTATION HEAD
#    Must match EXACTLY what you trained in train_augmented.py
# ============================================================================
class DenseNetWithSegmentation(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        base = models.densenet121(weights=None)

        # Freeze everything (matches training setup)
        for param in base.parameters():
            param.requires_grad = False

        self.features = base.features  # output: (B, 1024, 7, 7)

        # Classification head — must match training script exactly
        self.classifier = nn.Sequential(
            nn.Linear(base.classifier.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # Segmentation head — must match training script exactly
        self.seg_head = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 7→14
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 14→28
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 28→56
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 56→112
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 112→224
            nn.Sigmoid()
        )

    def forward(self, x):
        features   = F.relu(self.features(x), inplace=True)
        pooled     = F.adaptive_avg_pool2d(features, (1, 1))
        cls_output = self.classifier(torch.flatten(pooled, 1))
        seg_output = self.seg_head(features)   # (B, 1, 224, 224)
        return cls_output, seg_output


# ============================================================================
# 3. MODEL LOADING
#    Place your best_model_fold_X.pth in the same folder as main.py
#    and rename it to best_model.pth
# ============================================================================
def load_model(path):
    if not os.path.exists(path):
        print(f"❌ Cannot find model at: {path}")
        return None
    try:
        model = DenseNetWithSegmentation(num_classes=4)
        state_dict = torch.load(path, map_location=device, weights_only=True)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"✅ DenseNetWithSegmentation loaded on: {device}")
        return model
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        return None

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model_fold_2.pth")
model = load_model(MODEL_PATH)

# ============================================================================
# 4. PREPROCESSING PIPELINE
#    Gaussian σ=1 → Dynamic Threshold (T = 0.75μ + 0.25σ) → Normalise → Transform
# ============================================================================
model_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess(raw_img: np.ndarray):
    """
    Applies the full preprocessing pipeline to a raw float32 image array.
    Returns:
      input_tensor  : (1, 3, 224, 224) tensor ready for the model
      display_img   : (224, 224) uint8 numpy array — normalised for display
    """
    # Step 1 — Gaussian smoothing
    img_smooth = gaussian_filter(raw_img.astype(np.float32), sigma=GAUSSIAN_SIGMA)

    # Step 2 — Dynamic thresholding: T = 0.75μ + 0.25σ (per image)
    tissue_pixels = img_smooth[img_smooth > 0]
    if len(tissue_pixels) > 0:
        dynamic_threshold = (THRESHOLD_MEAN_WEIGHT * np.mean(tissue_pixels) +
                             THRESHOLD_STD_WEIGHT  * np.std(tissue_pixels))
        img_smooth[img_smooth < dynamic_threshold] = 0

    # Step 3 — Normalise to 0–255 uint8
    img_max, img_min = img_smooth.max(), img_smooth.min()
    if img_max - img_min > 0:
        img_norm = ((img_smooth - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img_norm = np.zeros_like(img_smooth, dtype=np.uint8)

    # Step 4 — Resize to 224×224 for display
    display_img = cv2.resize(img_norm, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Step 5 — Convert to PIL RGB and apply ImageNet transforms
    img_rgb      = Image.fromarray(display_img).convert('RGB')
    input_tensor = model_transform(img_rgb).unsqueeze(0).to(device)

    return input_tensor, display_img


def load_raw_image(file_bytes: bytes, filename: str) -> np.ndarray:
    """
    Loads raw MRI image array from either .mat or standard image format.
    HDF5 .mat files (Figshare originals) require .T transpose.
    Scipy .mat files (augmented/healthy) do not.
    """
    if filename.endswith('.mat'):
        with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            # Try HDF5 first (original Figshare v7.3)
            try:
                with h5py.File(tmp_path, 'r') as f:
                    return np.array(f['cjdata']['image'], dtype=np.float32).T
            except (OSError, KeyError):
                # Fall back to scipy (augmented/healthy v5)
                mat = sio.loadmat(tmp_path)
                return mat['cjdata'][0, 0]['image'].astype(np.float32)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        # Standard image (PNG, JPG etc.) — convert to grayscale float
        img_pil = Image.open(BytesIO(file_bytes)).convert('L')
        return np.array(img_pil, dtype=np.float32)


# ============================================================================
# 5. MASK OVERLAY HELPERS
# ============================================================================
def seg_mask_to_overlay(display_img_uint8: np.ndarray,
                        seg_mask: np.ndarray,
                        threshold: float = 0.3,
                        alpha: float = 0.45) -> np.ndarray:
    """
    Blends the segmentation mask over the MRI image.

    display_img_uint8 : (224, 224)   grayscale uint8
    seg_mask          : (224, 224)   float32  0–1  from sigmoid
    threshold         : pixels above this value are treated as tumor
    alpha             : opacity of the red overlay

    Returns           : (224, 224, 3) RGB uint8 with red tumor overlay
    """
    rgb      = cv2.cvtColor(display_img_uint8, cv2.COLOR_GRAY2RGB).astype(np.float32)
    mask_bin = (seg_mask > threshold).astype(np.float32)

    # Red fill
    red_layer            = np.zeros_like(rgb)
    red_layer[:, :, 0]   = 220
    red_layer[:, :, 1]   = 50
    red_layer[:, :, 2]   = 50

    mask_3ch = np.stack([mask_bin, mask_bin, mask_bin], axis=-1)
    blended  = rgb * (1 - alpha * mask_3ch) + red_layer * (alpha * mask_3ch)
    blended  = blended.astype(np.uint8)

    # Draw contour border on top
    mask_uint8  = (mask_bin * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (255, 60, 60), 2)

    return blended


def array_to_base64(img_array: np.ndarray, is_rgb: bool = True) -> str:
    """Converts a numpy image array to a base64-encoded PNG string."""
    if is_rgb:
        pil_img = Image.fromarray(img_array.astype(np.uint8), 'RGB')
    else:
        pil_img = Image.fromarray(img_array.astype(np.uint8), 'L')
    buffer = BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# ============================================================================
# 6. PREDICTION ENDPOINT
# ============================================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not initialised", "status": "error"}

    try:
        contents = await file.read()
        filename = (file.filename or "").lower()

        # ── Load raw image ────────────────────────────────────────────────
        raw_img  = load_raw_image(contents, filename)
        pipeline = "MAT Pipeline" if filename.endswith('.mat') else "Image Pipeline"

        # ── Preprocess ────────────────────────────────────────────────────
        input_tensor, display_img = preprocess(raw_img)

        # ── Run model — get both classification AND segmentation output ───
        input_tensor.requires_grad_(False)
        with torch.no_grad():
            cls_output, seg_output = model(input_tensor)
            probabilities           = F.softmax(cls_output, dim=1)
            confidence, pred_idx    = torch.max(probabilities, 1)

        predicted_class = pred_idx.item()
        predicted_label = class_names.get(predicted_class, "Unknown")

        # ── Build response images ─────────────────────────────────────────
        # Always return the preprocessed MRI as base64
        preprocessed_b64 = array_to_base64(
            cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB), is_rgb=True
        )

        # Only generate tumor overlay if prediction is NOT Normal
        overlay_b64  = None
        has_tumor    = predicted_label != "Normal"

        if has_tumor:
            # Extract seg mask: (1,1,224,224) → (224,224)
            seg_mask = seg_output.squeeze().cpu().numpy()  # values 0–1

            # Generate overlay image
            overlay_rgb = seg_mask_to_overlay(display_img, seg_mask,
                                              threshold=0.3, alpha=0.45)
            overlay_b64 = array_to_base64(overlay_rgb, is_rgb=True)

        # ── Return response ───────────────────────────────────────────────
        return {
            "prediction":        predicted_label,
            "confidence":        f"{confidence.item() * 100:.2f}%",
            "all_probabilities": {
                class_names[i]: f"{probabilities[0][i].item() * 100:.2f}%"
                for i in range(len(class_names))
            },
            "pipeline_used":     pipeline,
            "has_tumor":         has_tumor,

            # Base64-encoded PNG images for the frontend to display
            # Frontend: <img src={`data:image/png;base64,${preprocessed_image}`} />
            "preprocessed_image": preprocessed_b64,
            "tumor_overlay":      overlay_b64,   # null if Normal

            "status": "success"
        }

    except Exception as e:
        return {"error": str(e), "status": "error"}


@app.get("/health")
async def health_check():
    return {
        "status":  "running",
        "model":   "DenseNetWithSegmentation",
        "classes": 4,
        "device":  str(device)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
