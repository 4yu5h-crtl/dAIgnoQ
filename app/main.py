import streamlit as st
import os
import sys
from pathlib import Path
import torch
from PIL import Image

# Ensure the project root is in sys.path so 'dAIgnoQ' can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dAIgnoQ.app import config
from dAIgnoQ.app.components.sidebar import render_sidebar
from dAIgnoQ.app.components.dataset_uploader import render_dataset_uploader
from dAIgnoQ.app.utils.classifier import MedicalImageClassifier
from dAIgnoQ.app.utils.ensemble import EnsembleClassifier
from dAIgnoQ.app.utils.generation_utils import generate_gan_image, generate_diffusion_image
from dAIgnoQ.app.utils.generative_trainer import GenerativeModelTrainer
from dAIgnoQ.app.utils.training_pipeline import ClassicalTrainer, QuantumTrainer
from dAIgnoQ.app.utils.training_db import init_training_db, insert_generation_run, insert_training_run
from dAIgnoQ.app.utils.xai_utils import generate_shap_explanation, SHAP_AVAILABLE

# Page configuration
st.set_page_config(
    page_title="dAIgnoQ: Quantum-AI Medical Diagnosis",
    page_icon="🏥",
    layout="wide"
)

# ── Session State Initialization ──────────────────────────────────────────────
if 'classifier_resnet' not in st.session_state:
    st.session_state.classifier_resnet = MedicalImageClassifier(device=config.DEVICE)
if 'classifier_qsvm' not in st.session_state:
    st.session_state.classifier_qsvm = MedicalImageClassifier(device=config.DEVICE)
if 'img_size' not in st.session_state:
    st.session_state.img_size = config.IMG_SIZE
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = {}
if "training_db_initialized" not in st.session_state:
    init_training_db()
    st.session_state.training_db_initialized = True


def load_resnet():
    """Load the ResNet50 classifier."""
    try:
        model_path = st.session_state.get("user_classical_checkpoint", config.RESNET50_PATH)
        st.session_state.classifier_resnet.load_model(model_path, model_type='pytorch')
        st.session_state.models_loaded['ResNet50'] = True
        return True
    except Exception as e:
        st.error(f"Error loading ResNet50: {e}")
        return False


def load_qsvm():
    """Load the QSVM pipeline."""
    try:
        pipeline_path = st.session_state.get("user_qsvm_pipeline_checkpoint", str(config.QSVM_PIPELINE_PATH))
        st.session_state.classifier_qsvm.load_qsvm_pipeline(str(pipeline_path))
        st.session_state.models_loaded['QSVM'] = True
        return True
    except Exception as e:
        # Fallback to simple QSVM
        try:
            st.session_state.classifier_qsvm.load_model(
                st.session_state.get("user_qsvm_model_checkpoint", config.QSVM_MODEL_PATH),
                model_type='qsvm'
            )
            st.session_state.models_loaded['QSVM'] = True
            return True
        except Exception as e2:
            st.error(f"Error loading QSVM: {e2}")
            return False


def render_risk_indicator(confidence: float, prediction: str):
    """Render a visual risk level indicator."""
    if prediction == "Healthy":
        risk_level = "LOW"
        risk_color = "#28a745"
        risk_bg = "#d4edda"
        risk_emoji = "🟢"
    else:
        if confidence > 0.85:
            risk_level = "HIGH"
            risk_color = "#dc3545"
            risk_bg = "#f8d7da"
            risk_emoji = "🔴"
        elif confidence > 0.6:
            risk_level = "MODERATE"
            risk_color = "#fd7e14"
            risk_bg = "#fff3cd"
            risk_emoji = "🟡"
        else:
            risk_level = "LOW-MODERATE"
            risk_color = "#ffc107"
            risk_bg = "#fff3cd"
            risk_emoji = "🟡"

    st.markdown(f"""
    <div style="
        background: {risk_bg};
        border-left: 5px solid {risk_color};
        padding: 16px 20px;
        border-radius: 8px;
        margin: 8px 0;
    ">
        <div style="font-size: 14px; color: #555; margin-bottom: 4px;">Risk Assessment</div>
        <div style="font-size: 28px; font-weight: 700; color: {risk_color};">
            {risk_emoji} {risk_level}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_prediction_card(prediction: str, confidence: float):
    """Render a styled prediction result card."""
    is_healthy = prediction == "Healthy"
    bg = "#d4edda" if is_healthy else "#f8d7da"
    border = "#28a745" if is_healthy else "#dc3545"
    text_col = "#155724" if is_healthy else "#721c24"
    icon = "✅" if is_healthy else "⚠️"

    # Confidence bar
    bar_color = "#28a745" if confidence > 0.7 else ("#fd7e14" if confidence > 0.5 else "#dc3545")

    st.markdown(f"""
    <div style="
        background: {bg};
        border: 2px solid {border};
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin: 12px 0;
    ">
        <div style="font-size: 16px; color: {text_col}; margin-bottom: 6px;">Diagnosis</div>
        <div style="font-size: 32px; font-weight: 700; color: {text_col};">
            {icon} {prediction}
        </div>
        <div style="margin-top: 16px;">
            <div style="
                background: #e9ecef;
                border-radius: 10px;
                height: 14px;
                overflow: hidden;
            ">
                <div style="
                    background: {bar_color};
                    width: {confidence*100:.1f}%;
                    height: 100%;
                    border-radius: 10px;
                    transition: width 0.5s ease;
                "></div>
            </div>
            <div style="font-size: 14px; color: {text_col}; margin-top: 6px;">
                Confidence: {confidence:.1%}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def classify_and_display(classifier, image, model_name, sidebar_params):
    """Run classification on a single model and display results."""
    class_idx, confidence = classifier.predict(image)
    label = config.CLASS_LABELS.get(class_idx, "Unknown")
    gradcam_img = None
    shap_img = None
    report = None

    col_result, col_risk = st.columns([2, 1])
    with col_result:
        render_prediction_card(label, confidence)
    with col_risk:
        render_risk_indicator(confidence, label)

    # ── Visual Explanations ──────────────────────────────────────
    st.subheader("🔬 Visual Explanations")
    xai_cols = st.columns(2)

    with xai_cols[0]:
        st.markdown("**Grad-CAM Activation Map**")
        gradcam_img = classifier.get_gradcam(image)
        if gradcam_img is not None:
            st.image(gradcam_img, caption="Regions influencing the prediction (red = high importance)", use_column_width=True)
        else:
            st.info("Grad-CAM not available for this model type.")

    with xai_cols[1]:
        st.markdown("**SHAP Feature Attribution**")
        if SHAP_AVAILABLE and classifier.model_type in ['pytorch', 'hybrid']:
            try:
                shap_img = generate_shap_explanation(classifier.model, image, classifier.device)
                st.image(shap_img, caption="SHAP feature importance overlay", use_column_width=True)
            except Exception as e:
                st.info(f"SHAP analysis unavailable: {e}")
        else:
            st.info("SHAP requires PyTorch model and the `shap` package.")

    # ── AI Report ────────────────────────────────────────────────
    st.subheader("🤖 AI-Powered Analysis Report")
    if sidebar_params['gemini_key']:
        with st.spinner("Generating AI analysis..."):
            report = classifier.get_explanation(image, label, confidence)
            st.markdown(report)
    else:
        st.info("💡 Enter your Gemini API key in the sidebar for a detailed AI-generated report.")

    st.session_state.last_analysis = {
        "model_name": model_name,
        "prediction": label,
        "confidence": confidence,
        "gradcam": gradcam_img,
        "shap": shap_img,
        "report": report,
    }

    return class_idx, confidence, label


def main():
    # ── Header ───────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <h1 style="
            font-size: 2.8rem;
            background: linear-gradient(135deg, #1f77b4, #7b2ff7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
        ">🏥 dAIgnoQ</h1>
        <p style="color: #888; font-size: 1.1rem; margin: 0;">
            Quantum-Classical Hybrid Medical Imaging Platform
        </p>
        <p style="color: #aaa; font-size: 0.85rem; margin-top: 2px;">
            Generative Augmentation · Quantum ML · Explainable AI
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    sidebar_params = render_sidebar()

    # ── Tabs ─────────────────────────────────────────────────────
    tab_dataset, tab_generate, tab_training, tab_classify, tab_ensemble, tab_xai = st.tabs([
        "📂 Dataset Preparation",
        "🎨 Synthetic Image Generation",
        "🏋️ Model Training",
        "🔍 Single-Model Classification",
        "🔀 Ensemble Diagnosis",
        "🔬 XAI Review",
    ])

    # ══════════════════════════════════════════════════════════════
    # TAB 0 — Dataset Preparation (Generalized rare-disease workflow)
    # ══════════════════════════════════════════════════════════════
    with tab_dataset:
        dataset_state = render_dataset_uploader()
        if dataset_state.get("dataset_ready"):
            st.success("Dataset preparation complete. You can proceed to training workflows next.")
        else:
            st.info("Prepare dataset here first. Classification tabs can still be used with pre-trained models.")

    # ══════════════════════════════════════════════════════════════
    # TAB 1 — Synthetic Image Generation
    # ══════════════════════════════════════════════════════════════
    with tab_generate:
        st.header("🎨 Synthetic Medical Image Generation")
        st.write("Generate synthetic medical images using trained generative models for research and data augmentation.")
        generation_mode = st.radio(
            "Generation mode",
            ["Use pre-trained checkpoints", "Train generator on uploaded dataset", "Auto (intelligent selection)"],
            horizontal=True,
            key="generation_mode",
        )
        has_prepared_data = st.session_state.get("dataset_splits") is not None
        auto_rec = st.session_state.get("auto_recommendation")
        recommended_model = auto_rec.get("generator_model") if auto_rec else None
        if generation_mode in ("Train generator on uploaded dataset", "Auto (intelligent selection)") and not has_prepared_data:
            st.warning("Prepare dataset first in the Dataset Preparation tab.")
        if generation_mode == "Auto (intelligent selection)" and not auto_rec:
            st.warning("Run 'Analyze Dataset & Auto-Recommend' in Dataset Preparation tab first.")
        if generation_mode == "Auto (intelligent selection)" and auto_rec:
            st.success(
                f"Auto plan: use **{recommended_model.upper()}**, "
                f"{'apply' if auto_rec.get('use_augmentation') else 'skip'} augmentation, "
                f"target synthetic count {auto_rec.get('synthetic_target_count')}"
            )

        gen_col1, gen_col2 = st.columns(2)

        with gen_col1:
            st.subheader("GAN Generator")
            st.caption("Uses trained GAN to synthesize retinal fundus images")
            gan_enabled = (
                has_prepared_data
                and (
                    generation_mode == "Train generator on uploaded dataset"
                    or (generation_mode == "Auto (intelligent selection)" and recommended_model == "gan")
                )
            )
            if gan_enabled:
                gan_default_epochs = int((auto_rec or {}).get("train_defaults", {}).get("epochs", 1))
                gan_default_batch = int((auto_rec or {}).get("train_defaults", {}).get("batch_size", 8))
                gan_default_lr = float((auto_rec or {}).get("train_defaults", {}).get("learning_rate", 2e-4))
                gan_epochs = st.slider("GAN epochs", 1, 50, gan_default_epochs, key="gan_train_epochs")
                gan_batch = st.slider("GAN batch size", 2, 64, gan_default_batch, key="gan_train_batch")
                gan_lr = st.number_input(
                    "GAN learning rate",
                    min_value=1e-6,
                    max_value=1e-2,
                    value=gan_default_lr,
                    format="%.6f",
                )
                if st.button("Train GAN", key="train_gan"):
                    try:
                        with st.spinner("Training GAN on uploaded dataset..."):
                            train_dataset = st.session_state["dataset_splits"].get(
                                "train_augmented",
                                st.session_state["dataset_splits"]["train"],
                            )
                            trainer = GenerativeModelTrainer(model_type="gan", device=sidebar_params["device"])
                            result = trainer.train(
                                dataset=train_dataset,
                                epochs=gan_epochs,
                                batch_size=gan_batch,
                                learning_rate=float(gan_lr),
                                img_size=config.GAN_IMG_SIZE,
                            )
                        st.session_state.user_gan_checkpoint = result["checkpoint_path"]
                        insert_training_run(
                            model_family="gan",
                            dataset_size=len(train_dataset),
                            checkpoint_path=result["checkpoint_path"],
                            train_metric=result.get("g_loss"),
                            val_metric=result.get("d_loss"),
                            augmentation_applied=st.session_state["dataset_splits"].get("train_augmented") is not None,
                        )
                        st.success(f"GAN training complete. Checkpoint: {result['checkpoint_path']}")
                        g_loss = result.get("g_loss")
                        d_loss = result.get("d_loss")
                        if g_loss is not None and d_loss is not None:
                            st.caption(f"Final losses - G: {g_loss:.4f}, D: {d_loss:.4f}")
                    except Exception as e:
                        st.error(f"GAN training failed: {e}")
            elif generation_mode == "Auto (intelligent selection)" and recommended_model != "gan":
                st.info("Auto mode selected Diffusion for this dataset; GAN training is skipped.")

            num_gan = st.slider("Number of images", 1, 8, 1, key="gan_count")
            if st.button("🎨 Generate with GAN", key="gen_gan"):
                with st.spinner("Generating GAN images..."):
                    gan_ckpt = st.session_state.get("user_gan_checkpoint", config.GAN_GENERATOR_PATH)
                    cols = st.columns(min(num_gan, 4))
                    insert_generation_run("gan", str(gan_ckpt), num_gan)
                    for i in range(num_gan):
                        try:
                            gan_img = generate_gan_image(
                                gan_ckpt,
                                device=sidebar_params['device'],
                                img_size=config.GAN_IMG_SIZE
                            )
                            cols[i % len(cols)].image(gan_img, caption=f"GAN #{i+1}", use_column_width=True)
                        except Exception as e:
                            cols[i % len(cols)].error(f"Error: {e}")

        with gen_col2:
            st.subheader("Diffusion Model")
            st.caption("DDPM reverse-diffusion sampling for high-fidelity synthesis")
            diffusion_enabled = (
                has_prepared_data
                and (
                    generation_mode == "Train generator on uploaded dataset"
                    or (generation_mode == "Auto (intelligent selection)" and recommended_model == "diffusion")
                )
            )
            if diffusion_enabled:
                diff_default_epochs = int((auto_rec or {}).get("train_defaults", {}).get("epochs", 1))
                diff_default_batch = int((auto_rec or {}).get("train_defaults", {}).get("batch_size", 4))
                diff_default_lr = float((auto_rec or {}).get("train_defaults", {}).get("learning_rate", 1e-4))
                diff_epochs = st.slider("Diffusion epochs", 1, 20, diff_default_epochs, key="diff_train_epochs")
                diff_batch = st.slider("Diffusion batch size", 1, 32, diff_default_batch, key="diff_train_batch")
                diff_lr = st.number_input(
                    "Diffusion learning rate",
                    min_value=1e-6,
                    max_value=1e-2,
                    value=diff_default_lr,
                    format="%.6f",
                )
                if st.button("Train Diffusion", key="train_diffusion"):
                    try:
                        with st.spinner("Training diffusion model on uploaded dataset..."):
                            train_dataset = st.session_state["dataset_splits"].get(
                                "train_augmented",
                                st.session_state["dataset_splits"]["train"],
                            )
                            trainer = GenerativeModelTrainer(model_type="diffusion", device=sidebar_params["device"])
                            result = trainer.train(
                                dataset=train_dataset,
                                epochs=diff_epochs,
                                batch_size=diff_batch,
                                learning_rate=float(diff_lr),
                                img_size=config.DIFFUSION_IMG_SIZE,
                            )
                        st.session_state.user_diffusion_checkpoint = result["checkpoint_path"]
                        insert_training_run(
                            model_family="diffusion",
                            dataset_size=len(train_dataset),
                            checkpoint_path=result["checkpoint_path"],
                            train_metric=result.get("loss"),
                            augmentation_applied=st.session_state["dataset_splits"].get("train_augmented") is not None,
                        )
                        st.success(f"Diffusion training complete. Checkpoint: {result['checkpoint_path']}")
                        final_loss = result.get("loss")
                        if final_loss is not None:
                            st.caption(f"Final loss: {final_loss:.4f}")
                    except Exception as e:
                        st.error(f"Diffusion training failed: {e}")
            elif generation_mode == "Auto (intelligent selection)" and recommended_model != "diffusion":
                st.info("Auto mode selected GAN for this dataset; diffusion training is skipped.")

            diff_steps = st.slider("Inference steps", 10, 100, 50, key="diff_steps",
                                   help="More steps = higher quality but slower")
            if st.button("🎨 Generate with Diffusion", key="gen_diffusion"):
                with st.spinner(f"Running {diff_steps}-step diffusion sampling..."):
                    try:
                        insert_generation_run(
                            "diffusion",
                            str(st.session_state.get("user_diffusion_checkpoint", "")),
                            1,
                        )
                        diff_img = generate_diffusion_image(
                            st.session_state.get("user_diffusion_checkpoint"),
                            device=sidebar_params['device'],
                            img_size=config.DIFFUSION_IMG_SIZE,
                            num_inference_steps=diff_steps
                        )
                        st.image(diff_img, caption="Synthetic Image (Diffusion)", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error: {e}")

    # ══════════════════════════════════════════════════════════════
    # TAB 2 — Model Training
    # ══════════════════════════════════════════════════════════════
    with tab_training:
        st.header("Model Training (Classical + Quantum)")
        if st.session_state.get("dataset_splits") is None:
            st.warning("Prepare and split a dataset first in the Dataset Preparation tab.")
        else:
            train_dataset = st.session_state["dataset_splits"].get(
                "train_augmented",
                st.session_state["dataset_splits"]["train"],
            )
            val_dataset = st.session_state["dataset_splits"]["val"]

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Classical Model (ResNet50)")
                cls_epochs = st.slider("Classical epochs", 1, 30, 2, key="cls_epochs")
                cls_batch = st.slider("Classical batch size", 2, 64, 8, key="cls_batch")
                cls_lr = st.number_input(
                    "Classical learning rate",
                    min_value=1e-6,
                    max_value=1e-2,
                    value=1e-4,
                    format="%.6f",
                    key="cls_lr",
                )
                if st.button("Train Classical", key="train_classical_btn"):
                    try:
                        with st.spinner("Training classical model..."):
                            trainer = ClassicalTrainer(device=sidebar_params["device"])
                            result = trainer.train(
                                train_dataset=train_dataset,
                                val_dataset=val_dataset,
                                epochs=cls_epochs,
                                batch_size=cls_batch,
                                learning_rate=float(cls_lr),
                                img_size=st.session_state.get("img_size", config.IMG_SIZE),
                            )
                        st.session_state.user_classical_checkpoint = result["checkpoint_path"]
                        insert_training_run(
                            model_family="classical",
                            dataset_size=len(train_dataset),
                            checkpoint_path=result["checkpoint_path"],
                            train_metric=result.get("train_accuracy"),
                            val_metric=result.get("val_accuracy"),
                            augmentation_applied=st.session_state["dataset_splits"].get("train_augmented") is not None,
                        )
                        st.success(f"Saved checkpoint: {result['checkpoint_path']}")
                        st.caption(
                            f"Train acc: {result['train_accuracy']:.3f}"
                            + (
                                f" | Val acc: {result['val_accuracy']:.3f}"
                                if result.get("val_accuracy") is not None
                                else ""
                            )
                        )
                    except Exception as e:
                        st.error(f"Classical training failed: {e}")

            with col_b:
                st.subheader("Quantum Model (QSVM pipeline)")
                qsvm_batch = st.slider("Quantum batch size", 2, 64, 8, key="qsvm_batch")
                qsvm_pca = st.slider("PCA components", 2, 32, 12, key="qsvm_pca")
                qsvm_qubits = st.slider("Qubits", 2, 8, 4, key="qsvm_qubits")
                if st.button("Train Quantum (QSVM)", key="train_quantum_btn"):
                    try:
                        with st.spinner("Training quantum model..."):
                            trainer = QuantumTrainer(device=sidebar_params["device"])
                            result = trainer.train_qsvm(
                                train_dataset=train_dataset,
                                val_dataset=val_dataset,
                                pca_components=qsvm_pca,
                                n_qubits=qsvm_qubits,
                                batch_size=qsvm_batch,
                            )
                        st.session_state.user_qsvm_pipeline_checkpoint = result["checkpoint_path"]
                        insert_training_run(
                            model_family="quantum_qsvm",
                            dataset_size=len(train_dataset),
                            checkpoint_path=result["checkpoint_path"],
                            val_metric=result.get("val_accuracy"),
                            augmentation_applied=st.session_state["dataset_splits"].get("train_augmented") is not None,
                        )
                        st.success(f"Saved checkpoint: {result['checkpoint_path']}")
                        if result.get("val_accuracy") is not None:
                            st.caption(f"Val acc: {result['val_accuracy']:.3f}")
                    except Exception as e:
                        st.error(f"Quantum training failed: {e}")

            st.markdown("---")
            if st.button("Load Latest Trained Models for Inference", key="load_trained_models_btn"):
                loaded_a = load_resnet()
                loaded_b = load_qsvm()
                if loaded_a and loaded_b:
                    st.success("Both trained models loaded and ready for classification/ensemble tabs.")

    # ══════════════════════════════════════════════════════════════
    # TAB 1 — Single-Model Classification
    # ══════════════════════════════════════════════════════════════
    with tab_classify:
        st.header("Single-Model Classification")
        st.write("Upload a medical image and classify it with an individual model.")

        # Model loading
        model_choice = st.selectbox(
            "Select Model",
            ["ResNet50 (Classical CNN)", "QSVM (Quantum SVM)"],
            key="single_model_choice"
        )

        if st.button("Load Selected Model", key="load_single"):
            with st.spinner("Loading model..."):
                if "ResNet50" in model_choice:
                    load_resnet()
                else:
                    load_qsvm()

        # Show loaded status
        for name, loaded in st.session_state.models_loaded.items():
            if loaded:
                st.sidebar.success(f"✅ {name} loaded")

        uploaded = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png"], key="single_upload")

        if uploaded is not None:
            image = Image.open(uploaded).convert('RGB')

            col_img, col_info = st.columns([1, 1])
            with col_img:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            with col_info:
                st.markdown(f"""
                **Image Details**
                - Size: {image.size[0]} × {image.size[1]} px
                - Mode: {image.mode}
                """)

            if st.button("🔍 Analyze Image", key="classify_single"):
                if "ResNet50" in model_choice:
                    clf = st.session_state.classifier_resnet
                else:
                    clf = st.session_state.classifier_qsvm

                if clf.model is None:
                    st.warning("⚠️ Please load a model first!")
                else:
                    with st.spinner("Analyzing..."):
                        classify_and_display(clf, image, model_choice, sidebar_params)

    # ══════════════════════════════════════════════════════════════
    # TAB 2 — Ensemble Diagnosis
    # ══════════════════════════════════════════════════════════════
    with tab_ensemble:
        st.header("🔀 Ensemble Decision Fusion")
        st.write("Run multiple models simultaneously and fuse their predictions for a more robust diagnosis.")

        # Strategy selection
        strategy = st.selectbox(
            "Fusion Strategy",
            ["weighted_average", "max_confidence", "voting"],
            format_func=lambda x: {
                "weighted_average": "⚖️ Weighted Average",
                "max_confidence": "🎯 Max Confidence",
                "voting": "🗳️ Majority Voting"
            }.get(x, x),
            key="ensemble_strategy"
        )

        # Load both models
        load_col1, load_col2 = st.columns(2)
        with load_col1:
            if st.button("Load ResNet50", key="ens_load_resnet"):
                with st.spinner("Loading ResNet50..."):
                    load_resnet()
        with load_col2:
            if st.button("Load QSVM", key="ens_load_qsvm"):
                with st.spinner("Loading QSVM..."):
                    load_qsvm()

        # Model status indicators
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            if st.session_state.models_loaded.get('ResNet50'):
                st.success("✅ ResNet50 ready")
            else:
                st.warning("❌ ResNet50 not loaded")
        with status_col2:
            if st.session_state.models_loaded.get('QSVM'):
                st.success("✅ QSVM ready")
            else:
                st.warning("❌ QSVM not loaded")

        uploaded_ens = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png"], key="ensemble_upload")

        if uploaded_ens is not None:
            image = Image.open(uploaded_ens).convert('RGB')
            st.image(image, caption="Uploaded Image", width=350)

            if st.button("🔀 Run Ensemble Diagnosis", key="run_ensemble"):
                # Build classifiers dict from loaded models
                classifiers = {}
                if st.session_state.models_loaded.get('ResNet50'):
                    classifiers['ResNet50'] = st.session_state.classifier_resnet
                if st.session_state.models_loaded.get('QSVM'):
                    classifiers['QSVM'] = st.session_state.classifier_qsvm

                if len(classifiers) < 2:
                    st.warning("⚠️ Load at least 2 models for ensemble diagnosis. You can also use single-model classification.")
                    if len(classifiers) == 1:
                        st.info("Running single model as fallback...")
                        name, clf = list(classifiers.items())[0]
                        classify_and_display(clf, image, name, sidebar_params)
                else:
                    ensemble = EnsembleClassifier(classifiers, strategy=strategy)

                    with st.spinner("Running ensemble analysis..."):
                        final_class, final_confidence, individual_results = ensemble.predict(image)
                        final_label = config.CLASS_LABELS.get(final_class, "Unknown")

                    # Display fused result
                    st.markdown("---")
                    st.subheader("🎯 Fused Result")
                    render_prediction_card(final_label, final_confidence)
                    render_risk_indicator(final_confidence, final_label)

                    # Display ensemble report
                    report = ensemble.get_report(final_class, final_confidence, individual_results)
                    st.markdown(report)

                    # Show XAI from the highest-confidence model
                    st.markdown("---")
                    st.subheader("🔬 Visual Explanations (from best model)")
                    best_model_name = max(
                        individual_results,
                        key=lambda k: individual_results[k].get("confidence", 0)
                    )

                    if best_model_name == 'ResNet50':
                        best_clf = st.session_state.classifier_resnet
                    else:
                        best_clf = st.session_state.classifier_qsvm

                    xai_cols = st.columns(2)
                    gradcam_img = None
                    shap_img = None
                    with xai_cols[0]:
                        st.markdown("**Grad-CAM**")
                        gradcam_img = best_clf.get_gradcam(image)
                        if gradcam_img is not None:
                            st.image(gradcam_img, caption=f"Grad-CAM ({best_model_name})", use_column_width=True)
                        else:
                            st.info("Grad-CAM not available for this model.")

                    with xai_cols[1]:
                        st.markdown("**SHAP**")
                        if SHAP_AVAILABLE and best_clf.model_type in ['pytorch', 'hybrid']:
                            try:
                                shap_img = generate_shap_explanation(best_clf.model, image, best_clf.device)
                                st.image(shap_img, caption=f"SHAP ({best_model_name})", use_column_width=True)
                            except Exception as e:
                                st.info(f"SHAP unavailable: {e}")
                        else:
                            st.info("SHAP requires a PyTorch model.")

                    # Gemini report
                    if sidebar_params['gemini_key']:
                        st.subheader("🤖 AI Analysis Report")
                        with st.spinner("Generating AI report..."):
                            report_text = best_clf.get_explanation(image, final_label, final_confidence)
                            st.markdown(report_text)
                    else:
                        report_text = None

                    st.session_state.last_analysis = {
                        "model_name": f"Ensemble ({best_model_name} XAI source)",
                        "prediction": final_label,
                        "confidence": final_confidence,
                        "gradcam": gradcam_img,
                        "shap": shap_img,
                        "report": report_text,
                    }

    # ══════════════════════════════════════════════════════════════
    # TAB 5 — XAI Review
    # ══════════════════════════════════════════════════════════════
    with tab_xai:
        st.header("🔬 XAI Review")
        st.write("Review the latest prediction explanation from classification or ensemble analysis.")
        last = st.session_state.get("last_analysis")
        if not last:
            st.info("No analysis available yet. Run an image diagnosis first.")
        else:
            st.subheader(f"Source: {last.get('model_name', 'Unknown')}")
            st.write(f"Prediction: **{last.get('prediction', 'Unknown')}**")
            st.write(f"Confidence: **{float(last.get('confidence', 0.0)):.1%}**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Grad-CAM**")
                if last.get("gradcam") is not None:
                    st.image(last["gradcam"], use_column_width=True)
                else:
                    st.info("Grad-CAM not available.")
            with col2:
                st.markdown("**SHAP**")
                if last.get("shap") is not None:
                    st.image(last["shap"], use_column_width=True)
                else:
                    st.info("SHAP not available.")
            st.markdown("**AI Report**")
            if last.get("report"):
                st.markdown(last["report"])
            else:
                st.info("No Gemini report available. Provide API key to enable report generation.")

    # ── Custom Styles ────────────────────────────────────────────
    st.markdown("""
    <style>
        /* Global tweaks */
        .stApp {
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #f0f2f6;
        }
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
