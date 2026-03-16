from pathlib import Path
from typing import Dict, Any

import streamlit as st

from dAIgnoQ.app import config
from dAIgnoQ.app.utils.augmentation import DataAugmentor
from dAIgnoQ.app.utils.data_intelligence import DataIntelligenceEngine
from dAIgnoQ.app.utils.data_manager import DatasetManager


def render_dataset_uploader() -> Dict[str, Any]:
    """Render dataset upload/validation/split workflow for generalized diseases."""
    st.header("Dataset Upload & Preparation")
    st.write("Provide a local dataset folder to prepare data for rare-disease model training.")

    if "dataset_manager" not in st.session_state:
        st.session_state.dataset_manager = DatasetManager()

    manager: DatasetManager = st.session_state.dataset_manager

    dataset_path = st.text_input(
        "Dataset folder path",
        value=st.session_state.get("dataset_path", ""),
        help="Supported formats: (1) positive/negative subfolders, (2) images + labels.csv, (3) single folder.",
        key="dataset_path_input",
    ).strip()
    st.session_state.dataset_path = dataset_path

    validation_col, load_col = st.columns(2)
    format_type = "unknown"
    validation = None

    if dataset_path:
        format_type = manager.detect_format(dataset_path)
        st.caption(f"Detected format: `{format_type}`")

    if format_type == "single":
        positive_ratio = st.slider(
            "Positive label ratio (single-folder mode only)",
            min_value=0.05,
            max_value=0.95,
            value=float(st.session_state.get("single_positive_ratio", 0.5)),
            step=0.05,
            key="single_positive_ratio_slider",
        )
    else:
        positive_ratio = None

    with validation_col:
        if st.button("Validate Dataset", key="validate_dataset"):
            if not dataset_path:
                st.error("Please provide a dataset folder path.")
            else:
                validation = manager.validate_dataset(dataset_path)
                st.session_state.dataset_validation = validation

    validation = st.session_state.get("dataset_validation")
    if validation:
        if validation.get("valid"):
            st.success(validation.get("message", "Dataset looks valid."))
            stats = validation.get("stats", {})
            if stats:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total", int(stats.get("total", 0)))
                c2.metric("Positive", int(stats.get("positive", 0)))
                c3.metric("Negative", int(stats.get("negative", 0)))
        else:
            st.error(validation.get("message", "Dataset validation failed."))

    with load_col:
        if st.button("Load Dataset", key="load_dataset"):
            if not dataset_path:
                st.error("Please provide a dataset folder path.")
            else:
                try:
                    dataset, info = manager.load_dataset(
                        dataset_path,
                        positive_ratio=positive_ratio,
                    )
                    st.session_state.dataset_obj = dataset
                    st.session_state.dataset_info = info
                    st.success("Dataset loaded successfully.")
                except Exception as e:
                    st.error(f"Failed to load dataset: {e}")

    dataset_info = st.session_state.get("dataset_info")
    if dataset_info:
        st.markdown("### Loaded Dataset Summary")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Format", dataset_info.get("format", "unknown"))
        d2.metric("Total Images", int(dataset_info.get("total_images", 0)))
        d3.metric("Positive", int(dataset_info.get("positive_images", 0)))
        d4.metric("Negative", int(dataset_info.get("negative_images", 0)))

        st.markdown("### Train / Validation / Test Split")
        split_col1, split_col2, split_col3 = st.columns(3)
        train_ratio = split_col1.slider("Train %", 50, 90, 70, 1, key="train_ratio_slider")
        val_ratio = split_col2.slider("Val %", 5, 30, 15, 1, key="val_ratio_slider")
        test_ratio = 100 - train_ratio - val_ratio
        split_col3.metric("Test %", test_ratio)

        if test_ratio < 5:
            st.warning("Test split is very small. Consider reducing train/val ratios.")

        if st.button("Create Data Splits", key="create_data_splits"):
            dataset_obj = st.session_state.get("dataset_obj")
            if dataset_obj is None:
                st.error("Load dataset first.")
            else:
                try:
                    splits = manager.split_dataset(
                        dataset_obj,
                        train_ratio=train_ratio / 100.0,
                        val_ratio=val_ratio / 100.0,
                        test_ratio=test_ratio / 100.0,
                    )
                    st.session_state.dataset_splits = splits
                    st.success("Data splits created successfully.")
                except Exception as e:
                    st.error(f"Failed to split dataset: {e}")

    split_info = st.session_state.get("dataset_splits", {}).get("sizes")
    if split_info:
        st.markdown("### Split Sizes")
        s1, s2, s3 = st.columns(3)
        s1.metric("Train", int(split_info.get("train", 0)))
        s2.metric("Validation", int(split_info.get("val", 0)))
        s3.metric("Test", int(split_info.get("test", 0)))

        st.markdown("### Intelligent Strategy Recommendation")
        if st.button("Analyze Dataset & Auto-Recommend", key="run_intelligence"):
            try:
                rec = DataIntelligenceEngine.analyze_and_recommend(
                    st.session_state.get("dataset_info", {}),
                    st.session_state["dataset_splits"]["train"],
                )
                st.session_state.auto_recommendation = rec
                st.success("Auto recommendation generated.")
            except Exception as e:
                st.error(f"Failed to analyze dataset: {e}")

    rec = st.session_state.get("auto_recommendation")
    if rec:
        st.info(
            f"Recommended generator: **{rec['generator_model'].upper()}** | "
            f"Use augmentation: **{'Yes' if rec['use_augmentation'] else 'No'}** | "
            f"Synthetic target: **{rec['synthetic_target_count']}** images"
        )
        with st.expander("Why this recommendation?"):
            for reason in rec.get("reasons", []):
                st.write(f"- {reason}")
            q = rec.get("quality_metrics", {})
            st.write(
                f"Quality metrics — blur: `{q.get('mean_blur_score', 0):.2f}`, "
                f"diversity: `{q.get('mean_diversity', 0):.4f}`, "
                f"edge density: `{q.get('mean_edge_density', 0):.4f}`"
            )

        if rec.get("use_augmentation") and st.button("Apply Recommended Augmentation", key="apply_auto_aug"):
            try:
                augmented = DataAugmentor.augment_train_split(
                    st.session_state["dataset_splits"],
                    config=rec.get("augmentation_config", {}),
                    output_size=st.session_state.get("img_size", config.IMG_SIZE),
                )
                st.session_state.dataset_splits = augmented
                st.success("Recommended augmentation applied.")
            except Exception as e:
                st.error(f"Failed to apply recommended augmentation: {e}")

        st.markdown("### Optional Augmentation")
        a1, a2 = st.columns(2)
        rotation = a1.slider("Rotation (degrees)", 0, 30, 10, 1, key="aug_rotation")
        flip = a1.checkbox("Horizontal Flip", value=True, key="aug_flip")
        brightness = a2.slider("Brightness jitter", 0.0, 0.5, 0.1, 0.05, key="aug_brightness")
        contrast = a2.slider("Contrast jitter", 0.0, 0.5, 0.1, 0.05, key="aug_contrast")
        blur = a2.slider("Gaussian blur sigma max", 0.0, 2.0, 0.0, 0.1, key="aug_blur")

        if st.button("Apply Augmentation to Train Split", key="apply_augmentation"):
            try:
                aug_cfg = {
                    "rotation": rotation,
                    "flip": flip,
                    "brightness": brightness,
                    "contrast": contrast,
                    "blur": blur,
                }
                augmented = DataAugmentor.augment_train_split(
                    st.session_state["dataset_splits"],
                    config=aug_cfg,
                    output_size=st.session_state.get("img_size", config.IMG_SIZE),
                )
                st.session_state.dataset_splits = augmented
                st.success("Augmentation pipeline attached to train split.")
            except Exception as e:
                st.error(f"Failed to apply augmentation: {e}")

    return {
        "dataset_path": st.session_state.get("dataset_path", ""),
        "dataset_info": st.session_state.get("dataset_info"),
        "dataset_splits": st.session_state.get("dataset_splits"),
        "auto_recommendation": st.session_state.get("auto_recommendation"),
        "train_augmented": st.session_state.get("dataset_splits", {}).get("train_augmented") is not None,
        "dataset_ready": st.session_state.get("dataset_splits") is not None,
    }
