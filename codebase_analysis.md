# dAIgnoQ: Quantum-AI Medical Diagnosis Platform — Codebase Analysis

## Executive Summary

**dAIgnoQ** is a sophisticated **Quantum-Classical Hybrid Medical Imaging Platform** designed for medical image analysis (specifically glaucoma detection in retinal fundus images). The project combines classical deep learning (ResNet50 CNNs), quantum machine learning algorithms (Quantum Support Vector Machines and Variational Quantum Circuits via PennyLane), and advanced generative models (GANs and DDPM Diffusion) into a unified framework. The application is delivered as an interactive **Streamlit web interface** that enables clinicians and researchers to upload retinal images for diagnosis, compare predictions from multiple models simultaneously, and generate synthetic medical images for data augmentation. The codebase demonstrates cutting-edge hybrid quantum-classical computing applied to medical imaging with explainable AI (XAI) features including Grad-CAM visualization and SHAP feature attribution analysis.

**Technology Stack:** Python 3.8+, PyTorch, TorchVision, PennyLane (Quantum), Diffusers, Google Generative AI, Streamlit, Scikit-Learn  
**Architecture Style:** Modular, component-based with clear separation between UI (Streamlit), model inference, ensemble fusion, generative models, and explainability utilities  
**Project Status:** Production-ready hackathon project with trained models, multiple inference options, and comprehensive UI

---

## Project Overview

### Purpose and Business Domain

dAIgnoQ addresses critical challenges in **automated medical image analysis** by:

1. **Hybrid Classification:** Combining classical neural networks with quantum machine learning to potentially achieve higher diagnostic accuracy and robustness than classical methods alone
2. **Data Augmentation:** Generating synthetic retinal fundus images via GANs and Diffusion Models to expand training datasets for future research
3. **Explainability:** Providing clinicians with visual explanations (Grad-CAM heatmaps, SHAP feature importance) and AI-generated analysis reports to understand model decisions
4. **Ensemble Diagnosis:** Fusing predictions from multiple models with configurable strategies to improve decision robustness and reduce false positives

### Key Use Cases

- **Clinical Screening:** Automated preliminary glaucoma detection from retinal fundus images
- **Model Comparison:** Side-by-side evaluation of classical vs. quantum models on the same image
- **Research & Training:** Generation of synthetic retinal images for training augmentation without privacy concerns
- **Explainable AI:** Understanding which regions of retinal images influence model predictions
- **Ensemble Decision Support:** Combining multiple model predictions with confidence weighting for more reliable diagnosis

### Clinical Context

The project focuses on **glaucoma detection** in retinal fundus images. Glaucoma is a serious eye condition that can lead to irreversible blindness if not detected early. The platform provides:
- Binary classification: Healthy vs. Glaucoma
- Risk assessment indicators (LOW, LOW-MODERATE, MODERATE, HIGH)
- Visual explanation overlays to highlight affected regions
- AI-generated clinical insights via Google Gemini API

---

## Technology Stack

### Core Languages & Frameworks

| Component | Technology | Version/Notes |
|-----------|-----------|---------------|
| **Language** | Python | 3.8+ |
| **Web Framework** | Streamlit | Interactive dashboard UI |
| **Deep Learning** | PyTorch | Neural network models, tensors |
| **Vision Models** | TorchVision | ResNet50/ResNet18 pre-trained weights |
| **Quantum Computing** | PennyLane | VQC and QSVM implementations on default.qubit |
| **Generative Models** | Diffusers | DDPM diffusion sampling pipeline |
| **ML Toolkit** | Scikit-Learn | SVM, PCA dimensionality reduction |
| **Explainability** | Captum | Grad-CAM implementation |
| **Feature Attribution** | SHAP | DeepExplainer for feature importance |
| **AI API** | Google Generative AI | Gemini Vision model for report generation |
| **Image Processing** | OpenCV, Pillow, NumPy | Image I/O, preprocessing, visualization |
| **Data Serialization** | joblib, pickle, torch.save | Model checkpoint persistence |

### Key Dependencies

```
streamlit              # UI framework
torch, torchvision    # Deep learning
pennylane             # Quantum machine learning
diffusers             # Diffusion models
transformers          # Pre-trained model zoo
accelerate            # GPU optimization
captum                # Interpretability
shap                  # Feature attribution
scikit-learn          # Classical ML utilities
joblib                # Model serialization
google-generativeai   # Gemini API
opencv-python        # Image processing
matplotlib            # Visualization
numpy, pillow         # Array/image operations
```

---

## Architecture

### High-Level System Architecture

The platform follows a **modular, layered architecture** as documented in the README:

```
Patient Data (Retinal Images)
         │
         ▼
┌─────────────────────┐
│  Data Preprocessing  │   Resize, Normalize, Augment
└────────┬────────────┘
         │
    ┌────┴─────┐
    ▼          ▼
┌────────┐ ┌──────────────┐
│ GAN /  │ │ Augmented     │
│Diffusion│ │ Dataset       │
│Generator│ │               │
└────────┘ └───┬──────┬───┘
               │      │
          ┌────┘      └────┐
          ▼                ▼
   ┌────────────┐   ┌────────────┐
   │ Classical   │   │ Quantum    │
   │ CNN         │   │ ML         │
   │ (ResNet50)  │   │ (VQC/QSVM) │
   └──────┬─────┘   └─────┬──────┘
          │                │
          ▼                ▼
   ┌──────────────────────────┐
   │   Ensemble / Decision     │
   │   Fusion Module           │
   │  (Weighted Avg / Voting)  │
   └────────────┬─────────────┘
                │
          ┌─────┴──────┐
          ▼            ▼
   ┌───────────┐ ┌───────────┐
   │ Grad-CAM  │ │ Gemini AI │
   │ SHAP      │ │ Report    │
   └─────┬─────┘ └─────┬─────┘
         └──────┬──────┘
                ▼
   ┌──────────────────────┐
   │  Clinician Dashboard  │
   │  (Streamlit UI)       │
   └──────────────────────┘
```

### Architectural Layers

1. **Presentation Layer (Streamlit UI)**
   - `app/main.py`: 3-tab interactive dashboard
   - `app/components/sidebar.py`: Configuration panel
   - Visual risk indicators, prediction cards, confidence bars

2. **Classification Layer (Model Inference)**
   - `app/utils/classifier.py`: Unified `MedicalImageClassifier` for PyTorch, QSVM, and hybrid models
   - Handles model loading, preprocessing, and prediction
   - Integrates explainability methods

3. **Ensemble Fusion Layer**
   - `app/utils/ensemble.py`: `EnsembleClassifier` for decision fusion
   - Three fusion strategies: weighted average, max confidence, majority voting
   - Generates fusion reports with model agreement analysis

4. **Generative Models Layer**
   - `app/utils/generation_utils.py`: GAN and Diffusion image generation
   - Supports synthetic data augmentation for research

5. **Quantum Computing Layer**
   - `app/utils/quantum_utils.py`: PennyLane circuits for VQC and QSVM
   - Quantum state computation, kernel matrices, angle embeddings

6. **Explainability Layer**
   - `app/utils/xai_utils.py`: Grad-CAM, SHAP, Gemini AI report generation
   - Visual and textual explanations for predictions

### Key Architectural Patterns

**Model Strategy Pattern:** Different model types (PyTorch CNN, QSVM, Hybrid VQC) abstracted under `MedicalImageClassifier` with type-specific logic in `predict()` and `load_model()` methods.

**Ensemble Meta-Classifier Pattern:** `EnsembleClassifier` fuses predictions from multiple base models using pluggable fusion strategies (Strategy Pattern).

**Hooks & Callbacks:** Gradient hooks in Grad-CAM generation for layer-wise activation/gradient extraction.

**Session State Management:** Streamlit session state persists loaded models and configurations across user interactions.

---

## Project Structure

### Directory Organization

```
dAIgnoQ/
├── app/                                # Main application code
│   ├── main.py                         # Streamlit dashboard (3 tabs: classify, ensemble, generate)
│   ├── config.py                       # Paths, device, labels, model configuration
│   ├── __init__.py                     # Package marker
│   ├── components/
│   │   ├── __init__.py
│   │   └── sidebar.py                  # Sidebar UI: model status, Gemini API key, device selection
│   └── utils/
│       ├── __init__.py
│       ├── classifier.py               # MedicalImageClassifier: unified inference interface
│       ├── ensemble.py                 # EnsembleClassifier: decision fusion + reporting
│       ├── architectures.py            # HybridModel, Generator, Discriminator, DiffusionUNet
│       ├── generation_utils.py         # GAN & Diffusion image generation
│       ├── quantum_utils.py            # PennyLane: VQC circuits, QSVM kernels, quantum states
│       └── xai_utils.py                # Grad-CAM, SHAP, Gemini explanations
├── data/
│   ├── G1020/                          # Glaucoma dataset (OpenEDS standard)
│   │   ├── train/                      # Training images
│   │   └── test/                       # Test images
│   └── Gan_Train_Dataset/              # GAN training data
├── models/
│   └── checkpoints/                    # Trained model weights
│       ├── resnet50_finetuned.pth      # Fine-tuned ResNet50
│       ├── qsvm_model.pkl              # Scikit-learn QSVM classifier
│       ├── qsvm_pipeline.pth           # Complete QSVM pipeline (PCA + kernel + SVM)
│       ├── generator_final.pth         # GAN generator weights
│       ├── discriminator_final.pth     # GAN discriminator weights
│       └── [diffusion weights]         # DDPM model (if available)
├── notebooks/
│   ├── training/
│   │   ├── CNN_WITH_VQC.ipynb          # Hybrid ResNet18+VQC training
│   │   └── CNN_SEP_FINETUNING_QSVM.ipynb  # ResNet50 + QSVM training
│   └── generation/
│       ├── GAN.ipynb                   # GAN training
│       ├── Diffusion.ipynb             # DDPM diffusion model training
│       └── DDPM_Image_Generation.ipynb # Diffusion sampling pipeline
├── .planning/                          # Project planning & documentation
├── requirements.txt                    # Python package dependencies
├── run_app.py                          # Entry point: launches Streamlit
├── __init__.py                         # Package marker
└── README.md                           # Project documentation
```

### Entry Points

1. **`run_app.py`** (Recommended)
   - Simple launcher that sets PYTHONPATH and invokes `streamlit run`
   - Handles path resolution for module imports

2. **Direct Streamlit**
   ```bash
   streamlit run dAIgnoQ/app/main.py
   ```

### Configuration Files

- **`app/config.py`**: Central configuration for paths, image sizes, device selection, class labels, API settings
- **`requirements.txt`**: Package dependencies with pinned versions

---

## Core Modules & Components

### 1. **main.py** — Streamlit Dashboard (Main Entry Point)

**Responsibility:** Interactive web interface with 3 tabs for classification, ensemble analysis, and generative modeling.

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `main()` | Initializes page config, sidebar, and 3-tab layout |
| `load_resnet()` | Loads ResNet50 classifier from checkpoint |
| `load_qsvm()` | Loads QSVM pipeline or fallback QSVM model |
| `render_risk_indicator()` | Renders visual risk level (LOW/MODERATE/HIGH) |
| `render_prediction_card()` | Styled prediction result with confidence bar |
| `classify_and_display()` | Single-model classification + Grad-CAM + SHAP + Gemini report |

**Session State Variables:**
- `classifier_resnet`: ResNet50 classifier instance
- `classifier_qsvm`: QSVM classifier instance
- `models_loaded`: Dict tracking which models are loaded
- `img_size`: User-selected image resize dimensions
- `gemini_api_key`: User-provided Gemini API key

**UI Tabs:**

1. **Tab 1: Single-Model Classification**
   - Select model (ResNet50 or QSVM)
   - Upload image
   - View prediction, risk assessment, Grad-CAM, SHAP, Gemini report

2. **Tab 2: Ensemble Diagnosis**
   - Load both ResNet50 and QSVM
   - Select fusion strategy
   - View fused prediction with model agreement analysis
   - Display XAI from best-performing model

3. **Tab 3: Synthetic Image Generation**
   - GAN image synthesis (adjustable count, 1-8 images)
   - Diffusion model synthesis (adjustable inference steps)

---

### 2. **classifier.py** — MedicalImageClassifier

**Responsibility:** Unified inference interface for multiple model types (PyTorch CNN, QSVM, Hybrid VQC).

**Key Class: `MedicalImageClassifier`**

| Method | Purpose |
|--------|---------|
| `load_model(path, type)` | Load PyTorch, QSVM, or Hybrid model from checkpoint |
| `load_qsvm_pipeline(path)` | Load full QSVM pipeline with PCA, kernel, SVM components |
| `predict(image)` | Inference: returns (class_idx, confidence) |
| `get_gradcam(image)` | Generate Grad-CAM overlay for visualization |
| `setup_gemini(api_key)` | Configure Gemini API for report generation |
| `get_explanation(image, pred, conf)` | Get Gemini AI-generated analysis report |

**Model Type Support:**
- `'pytorch'`: Standard PyTorch models (ResNet50)
- `'qsvm'`: Simple QSVM classifier (joblib.load)
- `'qsvm_pipeline'`: Full pipeline with PCA + quantum kernel + SVM
- `'hybrid'`: Hybrid CNN-VQC model (ResNetVQC)

**Preprocessing Pipeline:**
- Resize to 224×224
- Convert to tensor
- Normalize with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

---

### 3. **ensemble.py** — EnsembleClassifier

**Responsibility:** Fuse predictions from multiple models using configurable strategies.

**Key Class: `EnsembleClassifier`**

| Method | Purpose |
|--------|---------|
| `predict(image)` | Run all models, fuse predictions, return (class, confidence, individual_results) |
| `_weighted_average(results)` | Weight each model's confidence by learned weights |
| `_max_confidence(results)` | Pick prediction from highest-confidence model |
| `_majority_voting(results)` | Majority vote with agreement-ratio-weighted confidence |
| `get_report(class, conf, results)` | Generate markdown report with model results + agreement analysis |

**Fusion Strategies:**

1. **Weighted Average** (default)
   - Each model votes with `weight × confidence`
   - Final class = argmax of weighted class scores
   - Balanced multi-model input

2. **Max Confidence**
   - Select prediction from model with highest confidence
   - Fast, uses single best model

3. **Majority Voting**
   - Count votes per class
   - Confidence = agreement ratio × average confidence of agreeing models
   - Robust to disagreement

---

### 4. **architectures.py** — Model Definitions

**Responsibility:** PyTorch model architectures for all classifier types.

**Key Classes:**

| Class | Purpose |
|-------|---------|
| `HybridModel` | ResNet18 backbone + compression layer + Variational Quantum Circuit |
| `ResNetVQC` | Alias for HybridModel |
| `Generator` | GAN generator with FC layers: latent_dim → 3×64×64 images |
| `Discriminator` | GAN discriminator: 3×64×64 images → binary validity score |
| `DiffusionUNet` | UNet for DDPM diffusion with attention and down/up blocks |

**HybridModel Architecture:**
```
Input Image (3×224×224)
    ↓
ResNet18 backbone [no pretrained weights]
    ↓
Global avg pool → (512,)
    ↓
Linear compression → (n_qubits,)
    ↓
Variational Quantum Circuit (PennyLane TorchLayer)
    ↓
Linear classifier → (2,) [Healthy/Glaucoma]
```

---

### 5. **generation_utils.py** — Synthetic Image Generation

**Responsibility:** GAN and Diffusion-based image synthesis for data augmentation.

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `generate_gan_image(path, device, latent_dim, img_size)` | Load Generator, sample latent vector, generate 64×64 image |
| `generate_diffusion_image(path, device, img_size, steps)` | Load UNet, run DDPM reverse-diffusion loop, generate 128×128 image |

**GAN Generation Pipeline:**
1. Sample latent vector from N(0,1)
2. Pass through Generator (FC layers)
3. Reshape to (3, 64, 64)
4. Denormalize from [-1, 1] to [0, 255]
5. Convert to PIL Image

**Diffusion Pipeline:**
1. Initialize UNet with trained weights (or random for demo)
2. Sample pure noise (1, 3, 128, 128)
3. Iteratively denoise with DDPMScheduler for N steps
4. Convert final sample to PIL Image

---

### 6. **quantum_utils.py** — Quantum Machine Learning Primitives

**Responsibility:** PennyLane quantum circuits, kernels, and state computation.

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `get_quantum_device(n_qubits)` | Initialize PennyLane 'default.qubit' device |
| `vqc_circuit(inputs, weights, n_qubits)` | Variational Quantum Circuit: angle embedding + entangling + measurement |
| `qstate(x, n_qubits, device)` | Compute quantum state from angles (return full state vector) |
| `compute_states(X_angles, n_qubits, device)` | Batch compute quantum states for QSVM kernel |
| `kernel_from_states(A, B)` | Quantum kernel: K = \|⟨ψ_A\|ψ_B⟩\|² |
| `scale_to_angles(X, mins, ranges)` | Normalize features to [-π/2, π/2] angle space |

**Quantum Circuits:**

1. **Variational Quantum Circuit (VQC)**
   - Angle embedding: map classical features to qubit rotations
   - Strongly entangling layers: parameterized gates
   - Pauli-Z measurement on each qubit
   - Used in hybrid CNN-VQC model

2. **Quantum Kernel (QSVM)**
   - Angle embedding with CNOT entanglement
   - Full quantum state output
   - Kernel matrix from state inner products

---

### 7. **xai_utils.py** — Explainability Module

**Responsibility:** Grad-CAM visualization, SHAP feature attribution, Gemini AI reports.

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `generate_gradcam(model, image, target_layer, device)` | Compute Grad-CAM heatmap and overlay on image |
| `generate_shap_explanation(model, image, device)` | Compute SHAP values and generate feature attribution image |
| `setup_gemini(api_key)` | Configure Google Generative AI client |
| `get_gemini_explanation(image, prediction, confidence)` | Query Gemini Vision for clinical insights |

**Grad-CAM Implementation:**
1. Forward pass with hook to capture target layer activations
2. Compute gradients of predicted class w.r.t. target layer
3. Global average pool gradients → importance weights
4. Weight feature maps and average → heatmap
5. Normalize and resize to image dimensions
6. Apply Jet colormap and blend with original image (40% heatmap, 60% image)

**SHAP Implementation:**
1. Wrap model with `shap.DeepExplainer`
2. Compute SHAP values for prediction class
3. Normalize and apply Jet colormap
4. Return visualization as numpy array

**Gemini Report:**
- Prompt-engineered to explain the prediction and image features
- Uses Gemini 1.5 Flash vision model
- Returns markdown-formatted clinical insights

---

### 8. **sidebar.py** — Configuration Interface

**Responsibility:** Streamlit sidebar for model management, API configuration, and settings.

**Key Components:**

| Control | Purpose |
|---------|---------|
| Model Status | Displays which models are loaded (✅ indicators) |
| Gemini API Key | Text input for Google API key (password field) |
| Image Size Slider | Adjust classification image dimensions (64-512px) |
| Device Selector | Choose CPU or CUDA for inference |
| About Section | Project branding and team info |

**State Management:**
- Updates `st.session_state` for all settings
- Calls `setup_gemini()` on classifiers when API key is entered
- Returns dict with all sidebar values for main.py

---

### 9. **config.py** — Application Configuration

**Responsibility:** Centralized paths, parameters, and constants.

**Key Configurations:**

```python
# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINTS_DIR = BASE_DIR / "dAIgnoQ" / "models" / "checkpoints"
DATA_DIR = BASE_DIR / "dAIgnoQ" / "data"

# Model paths
RESNET50_PATH = CHECKPOINTS_DIR / "resnet50_finetuned.pth"
QSVM_MODEL_PATH = CHECKPOINTS_DIR / "qsvm_model.pkl"
QSVM_PIPELINE_PATH = CHECKPOINTS_DIR / "qsvm_pipeline.pth"
GAN_GENERATOR_PATH = CHECKPOINTS_DIR / "generator_final.pth"

# Image dimensions
IMG_SIZE = (224, 224)               # Classification
GAN_IMG_SIZE = 64                   # GAN generation
DIFFUSION_IMG_SIZE = 128            # Diffusion generation

# Other settings
DEVICE = "cuda" if USE_CUDA else "cpu"
CLASS_LABELS = {0: "Healthy", 1: "Glaucoma"}
GEMINI_MODEL_NAME = "gemini-1.5-flash"
```

---

## Dependencies Analysis

### External Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| **streamlit** | Web UI framework | Latest |
| **torch, torchvision** | Deep learning, ResNet models | Latest |
| **pennylane** | Quantum circuits (VQC, QSVM) | Latest |
| **diffusers** | DDPM diffusion models | Latest |
| **transformers** | Pre-trained model hub | Latest |
| **captum** | Grad-CAM implementation | Latest |
| **shap** | SHAP feature attribution | Latest |
| **scikit-learn** | SVM, PCA utilities | Latest |
| **google-generativeai** | Gemini Vision API | Latest |
| **opencv-python** | Image processing (colormap, resize) | Latest |
| **matplotlib** | Visualization colormaps (Jet) | Latest |
| **numpy, pillow** | Array/image operations | Latest |
| **joblib** | Model serialization for QSVM | Latest |

### Internal Module Dependencies

```
main.py
  ├── config (paths, labels, device)
  ├── sidebar (sidebar UI rendering)
  ├── classifier (single-model inference)
  ├── ensemble (multi-model fusion)
  ├── generation_utils (GAN + Diffusion)
  ├── xai_utils (Grad-CAM, SHAP, Gemini)
  └── PIL, torch, torchvision (imaging, deep learning)

classifier.py
  ├── config (paths, image sizes, labels)
  ├── architectures (model definitions)
  ├── xai_utils (Grad-CAM, Gemini)
  ├── quantum_utils (QSVM pipeline)
  ├── torch, torchvision (model loading)
  └── joblib (QSVM model I/O)

ensemble.py
  ├── numpy (array operations)
  ├── config (class labels)
  └── PIL (image type hints)

architectures.py
  ├── torch, torch.nn (model building)
  ├── torchvision.models (ResNet backbones)
  ├── diffusers.UNet2DModel (diffusion UNet)
  └── pennylane (Variational Quantum Circuit)

generation_utils.py
  ├── torch (tensor operations)
  ├── numpy (array operations)
  ├── PIL (image conversion)
  ├── architectures (Generator, DiffusionUNet)
  ├── config (model paths, image sizes)
  ├── diffusers.DDPMScheduler (diffusion sampling)
  └── torchvision (preprocessing)

quantum_utils.py
  ├── numpy (state computation)
  ├── torch (tensor operations)
  └── pennylane (quantum circuits, QML)

xai_utils.py
  ├── numpy, cv2, torch (array, image, tensor ops)
  ├── matplotlib.cm (colormaps)
  ├── PIL (image I/O)
  ├── google.generativeai (Gemini API)
  ├── captum (Grad-CAM)
  └── shap (feature attribution)

sidebar.py
  ├── streamlit (UI components)
  └── config (device, image size)
```

---

## Code Conventions & Patterns

### Naming Conventions

**Files:**
- `snake_case.py` for modules (e.g., `classifier.py`, `generation_utils.py`)
- Descriptive names reflecting content (e.g., `xai_utils.py` for explainability)

**Classes:**
- `PascalCase` for class names (e.g., `MedicalImageClassifier`, `EnsembleClassifier`, `HybridModel`)
- Suffixed with purpose (`...Classifier`, `...Model`, `...UNet`)

**Functions:**
- `snake_case` for function names (e.g., `load_model()`, `predict()`, `generate_gradcam()`)
- Verb-noun pattern: `{verb}_{noun}()` or `get_{thing}()`, `generate_{thing}()`

**Variables:**
- `snake_case` for local variables and attributes
- Descriptive names: `classifier_resnet`, `final_confidence`, `individual_results`
- Prefixes for types: `X_` for arrays, `clf_` for classifiers, `q_` for quantum objects

### Code Organization Patterns

**Type-Based Polymorphism:** `MedicalImageClassifier.predict()` handles different model types via `self.model_type` branching rather than separate classes.

**Model State Dictionary:** Config stored in `config.py` rather than scattered constants throughout code.

**Session State for Stateful UI:** Streamlit session state persists model instances across user interactions.

**Hook-Based Activation Capture:** Grad-CAM uses PyTorch hooks to capture intermediate layer activations without modifying model code.

**Fusion Strategy Pattern:** `EnsembleClassifier` uses named strategies (`_weighted_average`, `_max_confidence`, `_majority_voting`) dispatched via string selection.

**Optional Dependencies:** Checks for `PENNYLANE_AVAILABLE`, `SHAP_AVAILABLE`, `CAPTUM_AVAILABLE` to gracefully handle missing imports.

### Preprocessing & Feature Normalization

**Image Preprocessing (consistent across all models):**
```python
transforms.Compose([
    transforms.Resize(config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```
- Fixed input size: 224×224
- ImageNet normalization (pre-trained ResNet standard)

**Quantum Angle Embedding:**
- Features scaled from feature space to [-π/2, π/2]
- PCA preprocessing before angle embedding (reduces dimensionality)

**QSVM Kernel Computation:**
- Quantum kernel K = |⟨ψ_A|ψ_B⟩|² (fidelity-based kernel)
- Computed between training and test states

### Error Handling

- **Fallback mechanisms:** QSVM tries pipeline load first, falls back to simple QSVM if fails
- **Try-except blocks:** Model loading has specific exception handling with user feedback
- **Graceful degradation:** XAI features (Grad-CAM, SHAP) disable gracefully if dependencies missing
- **API error handling:** Gemini exceptions caught and reported to user

---

## Key Findings & Technical Insights

### Architecture Strengths

1. **Modularity:** Clear separation of concerns (classification, fusion, generation, XAI) makes code maintainable and testable
2. **Flexibility:** Polymorphic `MedicalImageClassifier` supports multiple model types without client code changes
3. **Scalability:** Ensemble framework supports adding new models without modifying fusion logic
4. **Explainability:** Multiple XAI approaches (Grad-CAM, SHAP, Gemini) provide complementary explanations
5. **Quantum Integration:** Seamless integration of quantum circuits via PennyLane with classical fallbacks

### Technical Innovations

1. **Hybrid Quantum-Classical ML:** Combines ResNet feature extraction with quantum kernel methods (QSVM) and variational quantum circuits (VQC)
2. **Ensemble Decision Fusion:** Multiple strategies for combining classical and quantum predictions
3. **Dual Generative Models:** Both GAN and Diffusion-based approaches to synthetic image generation
4. **Multi-Modal Explainability:** Visual (Grad-CAM, SHAP) + textual (Gemini) explanations
5. **Interactive Dashboard:** Real-time model comparison and synthesis in web interface

### Areas of Complexity

1. **Quantum State Computation:** QSVM requires full state vector computation and kernel matrix generation (computationally expensive for many samples)
2. **Model Checkpoints:** Multiple checkpoint formats (PyTorch .pth, scikit-learn .pkl) require careful path management
3. **Preprocessing Consistency:** Different image sizes for classification (224×224), GAN (64×64), Diffusion (128×128) require careful dimension tracking
4. **API Dependencies:** Google Gemini API is external dependency with quota/reliability implications
5. **Device Management:** Code supports CPU and CUDA but tensor device placement must be explicit throughout

### Design Decisions

1. **Streamlit for UI:** Chosen for rapid prototyping, minimal boilerplate for data apps. Trade-off: less customization than React/Vue
2. **Stateless REST vs. Streamlit Session State:** Streamlit's session state persists models across interactions, avoiding reload overhead
3. **PennyLane for Quantum:** Abstraction over circuit definition (vs. Qiskit) for cleaner integration with PyTorch via TorchLayer
4. **Joblib for Model Persistence:** Standard for scikit-learn models; used for QSVM alongside PyTorch .pth files
5. **Dual Model Formats:** ResNet saved as PyTorch model; QSVM as joblib (sklearn native); trade-off in consistency for framework compatibility

### Potential Technical Debt

1. **Hardcoded Image Sizes:** Different sizes for classification vs. generation duplicated across codebase
2. **Path Resolution Complexity:** Multiple path conventions (relative, BASE_DIR-based) in config.py could be simplified
3. **Missing Unit Tests:** No test files visible; would benefit from pytest suite for model loading, inference, fusion strategies
4. **Error Messages:** Some exceptions too generic (e.g., "Error generating Gemini explanation"); could be more specific
5. **Quantum Overhead:** QSVM kernel computation is synchronous and blocking; could benefit from async handling for large batches
6. **Documentation:** Inline docstrings minimal in utils modules; would benefit from expanded parameter/return documentation

---

## Development Guide

### Prerequisites

- Python 3.8+
- pip or conda
- GPU (optional, for CUDA acceleration)
- Google Gemini API key (for report generation)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd dAIgnoQ
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Gemini API (optional):**
   - Obtain API key from [Google AI Studio](https://aistudio.google.com/)
   - Enter in app sidebar when running (stored in session state, not persisted)

### Running the Application

**Option 1 (Recommended):** Using the launcher
```bash
python run_app.py
```

**Option 2:** Direct Streamlit
```bash
streamlit run dAIgnoQ/app/main.py
```

**Option 3:** Using Python module
```bash
python -m streamlit run dAIgnoQ/app/main.py
```

The application will open at `http://localhost:8501` by default.

### Key Configuration Points

**Device Selection:** Set `USE_CUDA` environment variable or select in sidebar
```bash
USE_CUDA=1 python run_app.py  # Force CUDA
```

**Image Sizes (in `config.py`):**
```python
IMG_SIZE = (224, 224)           # Classification
GAN_IMG_SIZE = 64               # GAN generation
DIFFUSION_IMG_SIZE = 128        # Diffusion generation
```

**Model Paths (in `config.py`):**
Update paths if checkpoint files are in different location

### Development Workflows

#### Adding a New Model Type

1. Create model architecture in `architectures.py`
2. Add loading logic to `MedicalImageClassifier.load_model()` with new `model_type` branch
3. Add prediction logic to `MedicalImageClassifier.predict()` with conditional on `self.model_type`
4. Add model option to UI dropdown in `main.py`

#### Adding a New Fusion Strategy

1. Implement method in `EnsembleClassifier`: `def _new_strategy(self, results):`
2. Add strategy name to `STRATEGIES` class variable
3. Add dispatch in `_fuse()` method
4. Update UI dropdown in ensemble tab

#### Implementing a New XAI Method

1. Create function in `xai_utils.py` (e.g., `def generate_new_explanation(...)`)
2. Import and call in `classifier.py` or directly in `main.py`
3. Render output (image or markdown) in appropriate tab/card

#### Training Custom Models

Use Jupyter notebooks in `notebooks/training/` and `notebooks/generation/`:
- `CNN_WITH_VQC.ipynb`: Hybrid model training
- `CNN_SEP_FINETUNING_QSVM.ipynb`: QSVM training
- `GAN.ipynb`: GAN training
- `Diffusion.ipynb`: Diffusion model training

Save trained weights to `models/checkpoints/` with appropriate naming convention.

### Testing Approach

**Manual Testing:**
1. Load each model type individually and verify predictions
2. Test ensemble with different fusion strategies
3. Test XAI outputs (Grad-CAM, SHAP)
4. Test generative models (GAN, Diffusion)
5. Test Gemini API integration with various images

**Suggested Test Cases (for automated testing):**
- Model loading from missing checkpoint files
- Prediction on images of different aspect ratios
- Ensemble with single model loaded (graceful fallback)
- Division by zero in normalization
- Quantum state computation with edge cases (all zeros, NaNs)

### Building & Deployment

**Local Build:**
```bash
streamlit run dAIgnoQ/app/main.py
```

**Docker (suggested, not included):**
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "dAIgnoQ/app/main.py"]
```

**Streamlit Cloud Deployment:**
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository, select `dAIgnoQ/app/main.py` as main file
4. Deploy

---

## Notes for New Developers

### Quick Start Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Understand the three-tab UI structure: Classification | Ensemble | Generation
- [ ] Familiarize with `config.py` for paths and settings
- [ ] Read the `README.md` for project context
- [ ] Trace the flow: upload image → preprocess → run inference → render results
- [ ] Explore quantum primitives in `quantum_utils.py` to understand QSVM/VQC

### Most Important Files to Understand First

1. **`config.py`** — Start here for all paths, constants, and configuration
2. **`main.py`** — Entry point; understand the three-tab structure and session state management
3. **`classifier.py`** — Core inference logic; understand how different model types are handled
4. **`ensemble.py`** — Decision fusion; understand how predictions are combined
5. **`architectures.py`** — Model definitions; understand each model type used
6. **`xai_utils.py`** — Explainability; understand how insights are generated

### Common Gotchas & Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Import errors (`ModuleNotFoundError: dAIgnoQ`) | PYTHONPATH not set | Run via `run_app.py` or set `PYTHONPATH` |
| Model loading fails | Wrong checkpoint path | Verify paths in `config.py` match actual files |
| CUDA out of memory | GPU RAM exceeded | Switch to CPU in sidebar or reduce batch size |
| SHAP visualization missing | `shap` package not installed | Install with `pip install shap` |
| Gemini API returns empty | API key invalid or quota exceeded | Check API key, verify billing enabled |
| Quantum circuit errors | PennyLane not installed | Install with `pip install pennylane` |
| Image preprocessing errors | RGB conversion fails | Handle grayscale by converting to RGB first |
| Diffusion generation slow | Many inference steps specified | Reduce `diff_steps` slider in UI |

### Key Concepts to Understand

**Quantum Kernel SVM (QSVM):**
- Features are embedded into quantum states
- Quantum states combined via CNOT entanglement
- Kernel computed from state overlaps (inner products)
- Kernel fed to classical SVM for classification

**Variational Quantum Circuit (VQC):**
- Parametrized quantum circuit as trainable layer in neural network
- ResNet extracts classical features
- Features compressed and fed into VQC
- VQC outputs measured on all qubits
- Final classification via classical dense layer

**Ensemble Decision Fusion:**
- **Weighted Average:** Confidence-weighted voting per class
- **Max Confidence:** Trust the model with highest confidence
- **Majority Voting:** Democratic vote with confidence = agreement ratio

**Explainability Methods:**
- **Grad-CAM:** Shows which image regions influenced the prediction (red = high influence)
- **SHAP:** Feature-level attribution for pixel importance
- **Gemini:** AI-generated clinical narrative explaining the prediction

### Where to Look for Specific Functionality

| Functionality | File | Key Class/Function |
|---------------|------|-------------------|
| Image classification | `classifier.py` | `MedicalImageClassifier.predict()` |
| Risk indicators | `main.py` | `render_risk_indicator()` |
| Prediction styling | `main.py` | `render_prediction_card()` |
| Multi-model fusion | `ensemble.py` | `EnsembleClassifier` |
| Grad-CAM visualization | `xai_utils.py` | `generate_gradcam()` |
| SHAP explanation | `xai_utils.py` | `generate_shap_explanation()` |
| Gemini AI report | `xai_utils.py` | `get_gemini_explanation()` |
| GAN synthesis | `generation_utils.py` | `generate_gan_image()` |
| Diffusion synthesis | `generation_utils.py` | `generate_diffusion_image()` |
| Quantum circuits | `quantum_utils.py` | `vqc_circuit()`, `qstate()` |
| Model loading | `classifier.py` | `load_model()`, `load_qsvm_pipeline()` |
| Sidebar UI | `sidebar.py` | `render_sidebar()` |
| Configuration | `config.py` | (constants and paths) |

### Debugging Tips

1. **Enable Streamlit verbose mode:**
   ```bash
   streamlit run dAIgnoQ/app/main.py --logger.level=debug
   ```

2. **Add print statements in callbacks:**
   ```python
   st.write(f"Debug: model_type={clf.model_type}, confidence={conf:.3f}")
   ```

3. **Inspect session state:**
   ```python
   st.sidebar.write(st.session_state)
   ```

4. **Check model gradients (for development):**
   ```python
   for name, param in model.named_parameters():
       if param.grad is not None:
           st.write(f"{name}: grad_mean={param.grad.mean():.6f}")
   ```

5. **Profile quantum computation:**
   ```python
   import time
   start = time.time()
   states = compute_states(X_angles, n_qubits, device)
   print(f"Quantum state computation took {time.time()-start:.3f}s")
   ```

### Advanced Topics

**Quantum Circuit Customization:**
- Modify `vqc_circuit()` in `quantum_utils.py` to change entanglement pattern
- Change number of qubits, layers, embedding type
- Experiment with different PennyLane gates

**Model Ensemble Customization:**
- Implement custom fusion in `EnsembleClassifier._fuse()`
- Add learned weights per model instead of equal weights
- Implement confidence calibration post-hoc

**XAI Extension:**
- Add Integrated Gradients via Captum
- Implement Attention Rollout for transformer models
- Add saliency maps via gradient-based approaches

---

## Summary

dAIgnoQ is a sophisticated **hybrid quantum-classical medical imaging platform** that demonstrates cutting-edge applications of quantum machine learning in healthcare. The codebase is **well-organized** with clear module separation, implements **multiple state-of-the-art techniques** (quantum kernels, variational circuits, ensemble fusion, explainable AI), and provides a **professional user interface** for clinical use. The architecture supports easy extension with new models, fusion strategies, and explainability methods. While there are opportunities for increased test coverage and documentation, the project successfully achieves its goal of combining quantum and classical ML for enhanced diagnostic accuracy with interpretable, clinician-friendly explanations.

**Team:** Overfit Squad (Tanishq Zade, Ayush Chintalwar, Advait Raktate, Divyansh Dubey)
