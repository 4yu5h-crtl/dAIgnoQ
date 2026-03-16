import streamlit as st
import os
from dAIgnoQ.app import config


def render_sidebar():
    """
    Renders the sidebar for configuration, model status, and API settings.
    """
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 10px 0;">
        <h2 style="margin: 0;">⚙️ Configuration</h2>
    </div>
    """, unsafe_allow_html=True)

    # ── Model Status ─────────────────────────────────────────────
    st.sidebar.subheader("📋 Model Status")

    loaded = st.session_state.get('models_loaded', {})
    if loaded:
        for name, status in loaded.items():
            if status:
                st.sidebar.success(f"✅ {name} loaded")
    else:
        st.sidebar.info("No models loaded yet. Use the main panel to load models.")

    # ── Gemini API ───────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Gemini AI")
    gemini_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state.get('gemini_api_key', ''),
        help="Get your key from https://aistudio.google.com/app/apikey"
    )

    if gemini_key:
        st.session_state.gemini_api_key = gemini_key
        # Configure on any classifier that exists
        for clf_key in ['classifier_resnet', 'classifier_qsvm']:
            if clf_key in st.session_state:
                st.session_state[clf_key].setup_gemini(gemini_key)

    # ── Image Parameters ─────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("🖼️ Image Parameters")
    img_size = st.sidebar.slider("Classification Image Size", 64, 512, config.IMG_SIZE[0], 64)
    st.session_state.img_size = (img_size, img_size)

    # ── Device ───────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("💻 Computation")
    device = st.sidebar.selectbox(
        "Device",
        ["cpu", "cuda"],
        index=0 if config.DEVICE == "cpu" else 1
    )
    st.session_state.device = device

    # ── About ────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="
        background: linear-gradient(135deg, #1f77b4, #7b2ff7);
        color: white;
        padding: 16px;
        border-radius: 10px;
        font-size: 0.85rem;
    ">
        <b>dAIgnoQ</b><br/>
        Quantum-Classical Hybrid<br/>
        Medical Imaging Platform<br/><br/>
        <em>Team Overfit Squad</em>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.caption("⚠️ For research/educational purposes only.")

    return {
        "gemini_key": gemini_key,
        "img_size": (img_size, img_size),
        "device": device
    }
