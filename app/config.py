import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# App specific directories
APP_DIR = BASE_DIR / "dAIgnoQ" / "app"
MODELS_DIR = BASE_DIR / "dAIgnoQ" / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
DATA_DIR = BASE_DIR / "dAIgnoQ" / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
RUNTIME_DATA_DIR = DATA_DIR / "runtime"

# Ensure generalized runtime directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Relative paths from BASE_DIR
REL_CHECKPOINTS_DIR = "dAIgnoQ/models/checkpoints"
REL_DATA_DIR = "dAIgnoQ/data"

# Model specific paths
RESNET50_PATH = CHECKPOINTS_DIR / "resnet50_finetuned.pth"
QSVM_MODEL_PATH = CHECKPOINTS_DIR / "qsvm_model.pkl"
QSVM_PTH_PATH = CHECKPOINTS_DIR / "qsvm_model.pth"
QSVM_PIPELINE_PATH = CHECKPOINTS_DIR / "qsvm_pipeline.pth"
GAN_GENERATOR_PATH = CHECKPOINTS_DIR / "generator_final.pth"
GAN_DISCRIMINATOR_PATH = CHECKPOINTS_DIR / "discriminator_final.pth"

# Default parameters
IMG_SIZE = (224, 224)
GAN_IMG_SIZE = 64
DIFFUSION_IMG_SIZE = 128
BATCH_SIZE = 32
DEVICE = "cuda" if os.environ.get("USE_CUDA") == "1" else "cpu"

# Classification labels
CLASS_LABELS = {0: "Healthy", 1: "Glaucoma"}
DYNAMIC_CLASS_LABELS = {0: "Negative", 1: "Positive"}

# Gemini API configuration
GEMINI_MODEL_NAME = "gemini-1.5-flash"

# Generalized run-time config for disease-agnostic workflows
CURRENT_DISEASE = os.environ.get("DAIGNOQ_DISEASE", "rare_disease")
