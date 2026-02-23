"""Streamlit URL Safety Predictor App."""
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
import re
import ipaddress
from urllib.parse import urlparse

from url_features import build_feature_vector
from url_rules import run_rule_checks, is_unsafe_by_rules

# Paths to project files
BASE_DIR = Path(__file__).resolve().parent
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns.joblib"
MODEL_PATH = BASE_DIR / "hybrid_model.joblib"
CSS_PATH = BASE_DIR / "styles.css"

# ML threshold: probability >= this -> Unsafe
ML_THRESHOLD = 0.5


@st.cache_resource
def load_model_and_columns():
    """Load feature columns and model once, cache for reuse."""
    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"Feature columns file not found: {FEATURE_COLUMNS_PATH}. "
            "Ensure feature_columns.joblib is in the same directory as app.py."
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Ensure hybrid_model.joblib is in the same directory as app.py."
        )
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    model = joblib.load(MODEL_PATH)
    # Handle various formats: list, Index, ndarray
    if hasattr(feature_columns, "tolist"):
        feature_columns = feature_columns.tolist()
    elif hasattr(feature_columns, "to_list"):
        feature_columns = feature_columns.to_list()
    elif not isinstance(feature_columns, list):
        feature_columns = list(feature_columns)
    return feature_columns, model


def apply_custom_css() -> None:
    """Load custom CSS from styles.css and inject it into the app."""
    if not CSS_PATH.exists():
        raise FileNotFoundError(
            f"CSS file not found: {CSS_PATH}. Ensure styles.css is in the same directory as app.py."
        )
    css = CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)


def get_risk_tier(risk_score: float) -> str:
    """Map risk score (0-1) to tier."""
    if risk_score <= 0.30:
        return "Low"
    if risk_score <= 0.60:
        return "Medium"
    return "High"


def is_valid_url(url: str) -> bool:
    """Check if the given string has a valid URL format."""
    url = url.strip()
    if not url:
        return False
    
    # Check if a protocol scheme is given
    has_scheme = bool(re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', url))
    
    if not has_scheme:
        has_dot = '.' in url
        
        is_ip = False
        try:
            host_to_check = url
            # Remove brackets and port if it's an IPv6 like [::1]:8080
            if host_to_check.startswith('['):
                host_to_check = host_to_check.split(']')[0].lstrip('[')
            
            if host_to_check.isdigit():
                ipaddress.ip_address(int(host_to_check))
            else:
                ipaddress.ip_address(host_to_check)
            is_ip = True
        except ValueError:
            is_ip = False

        if not has_dot and not is_ip and url.lower() != 'localhost':
            return False

        # Prepend https:// for valid schemeless inputs
        url = 'https://' + url

    try:
        parsed = urlparse(url)
        
        if parsed.scheme.lower() not in ('http', 'https'):
            return False
            
        if not parsed.netloc:
            return False
            
        hostname = parsed.hostname
        if not hostname:
            return False
            
        if '.' in hostname or hostname.lower() == 'localhost':
            return True
            
        try:
            ip_str = hostname.strip('[]')
            if ip_str.isdigit():
                ipaddress.ip_address(int(ip_str))
            else:
                ipaddress.ip_address(ip_str)
            return True
        except ValueError:
            return False
            
    except Exception:
        return False


def predict_url(
    url: str,
    feature_columns: list,
    model,
    rule_findings: dict | None = None,
    threshold: float = ML_THRESHOLD,
) -> tuple[str, float, list[str]]:
    """
    Predict if URL is safe or unsafe. Mark UNSAFE if ANY rule triggers.
    Returns (label, risk_score, unsafe_reasons).
    """
    # Rule-based override: ANY detection -> Unsafe (conservative)
    is_unsafe, unsafe_reasons = is_unsafe_by_rules(url)
    if is_unsafe:
        return "Unsafe", 1.0, unsafe_reasons

    X = build_feature_vector(feature_columns, url)
    X_df = pd.DataFrame([X], columns=feature_columns)

    risk_score: float
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_df)
        except (ValueError, TypeError):
            proba = model.predict_proba(X_df.values)
        # P(unsafe) = proba for class 1
        risk_score = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
    else:
        try:
            pred = model.predict(X_df)
        except (ValueError, TypeError):
            pred = model.predict(X_df.values)
        raw = int(pred[0]) if hasattr(pred[0], "__int__") else int(pred[0])
        risk_score = 1.0 if raw == 1 else 0.0

    label = "Unsafe" if risk_score >= threshold else "Safe"
    return label, risk_score, []


def main():
    st.set_page_config(
        page_title="URL Safety Predictor",
        page_icon=":link:",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    try:
        apply_custom_css()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    st.title("URL Safety Predictor")
    st.markdown(
        '<p class="subtitle">Paste a URL to check if it is safe or unsafe</p>',
        unsafe_allow_html=True,
    )

    # Session state for result
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "risk_score" not in st.session_state:
        st.session_state.risk_score = None
    if "rule_findings" not in st.session_state:
        st.session_state.rule_findings = None
    if "unsafe_reasons" not in st.session_state:
        st.session_state.unsafe_reasons = None
    if "url_input" not in st.session_state:
        st.session_state.url_input = ""

    # Load model (cached)
    try:
        feature_columns, model = load_model_and_columns()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    def handle_clear():
        st.session_state.prediction_result = None
        st.session_state.risk_score = None
        st.session_state.rule_findings = None
        st.session_state.unsafe_reasons = None
        st.session_state.url_input = ""

    url = st.text_input(
        "URL",
        key="url_input",
        placeholder="https://example.com or paste any link here...",
        label_visibility="collapsed",
    )

    url_entered = bool(url and url.strip())
    valid_url_flag = is_valid_url(url.strip()) if url_entered else False
    show_error = url_entered and not valid_url_flag

    if show_error:
        st.markdown(
            '<div style="color: var(--unsafe); font-size: 0.9rem; margin-top: -0.5rem; margin-bottom: 0.5rem;">'
            '❌ Please enter a valid URL (https://...)</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<style>div[data-testid="stTextInput"] > div:first-child { border-color: var(--unsafe) !important; box-shadow: 0 0 0 1px var(--unsafe) !important; }</style>',
            unsafe_allow_html=True
        )

    col1, col2 = st.columns(2)
    with col1:
        predict_clicked = st.button("Predict", disabled=not valid_url_flag, use_container_width=True)
    with col2:
        st.button("Clear", on_click=handle_clear, use_container_width=True)

    if predict_clicked and valid_url_flag:
        valid_url = url.strip()
        try:
            rule_findings = run_rule_checks(valid_url)
            label, risk_score, unsafe_reasons = predict_url(
                valid_url, feature_columns, model, rule_findings=rule_findings
            )
            st.session_state.prediction_result = label
            st.session_state.risk_score = risk_score
            st.session_state.rule_findings = rule_findings
            st.session_state.unsafe_reasons = unsafe_reasons
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.session_state.prediction_result = None
            st.session_state.risk_score = None
            st.session_state.rule_findings = None
            st.session_state.unsafe_reasons = None

    # Result display
    if st.session_state.prediction_result is not None:
        label = st.session_state.prediction_result
        risk_score = st.session_state.risk_score
        risk_pct = int((risk_score or 0) * 100)
        risk_tier = get_risk_tier(risk_score or 0)
        unsafe_reasons = st.session_state.unsafe_reasons or []
        # Use Suspicious styling for medium risk (borderline cases)
        if risk_tier == "Medium":
            display_label = "⚠ Suspicious"
            css_class = "result-suspicious"
        else:
            display_label = "🛡 Safe" if label == "Safe" else "⚠ Phishing"
            css_class = "result-safe" if label == "Safe" else "result-unsafe"

        # Structured result card
        if risk_pct <= 30:
            bar_color = "var(--safe)"
        elif risk_pct <= 60:
            bar_color = "#eab308"
        else:
            bar_color = "var(--unsafe)"

        meta_html = (
            f"<div style='margin-bottom: 8px;'>Risk score: <strong>{risk_pct}%</strong> ({risk_tier}) "
            f"&middot; Threshold: {int(ML_THRESHOLD * 100)}%</div>"
            f"<div style='background: #e5e7eb; border-radius: 999px; height: 8px; width: 100%;'>"
            f"<div style='background: {bar_color}; width: {risk_pct}%; height: 100%; border-radius: 999px; transition: width 0.5s ease;'></div>"
            f"</div>"
        )
        reasons_html = ""
        if label == "Unsafe" and unsafe_reasons:
            items = "".join(f"<li>{r}</li>" for r in unsafe_reasons)
            reasons_html = (
                '<div class="result-reasons"><strong>Detection reasons:</strong>'
                f"<ul>{items}</ul></div>"
            )

        st.markdown(
            f"""
            <div class="result-container {css_class}">
                <div class="result-header">Result: {display_label}</div>
                <div class="result-meta">{meta_html}</div>
                {reasons_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Additional rule-based findings (expander)
        findings = st.session_state.rule_findings
        if findings and any(
            v
            for k, v in findings.items()
            if k in ("sensitive_words", "brand_names", "misspelled_brands") and v
        ):
            with st.expander("Detailed analysis", expanded=len(unsafe_reasons) == 0):
                md_lines = []
                if findings.get("sensitive_words"):
                    md_lines.append("**🔎 Sensitive Words Detected**")
                    for w in findings["sensitive_words"]:
                        md_lines.append(f"- {w}")
                    md_lines.append("")
                if findings.get("brand_names"):
                    md_lines.append("**🏷 Brand Names Found**")
                    for b in findings["brand_names"]:
                        md_lines.append(f"- {b}")
                    md_lines.append("")
                if findings.get("misspelled_brands"):
                    md_lines.append("**⚠ Possible Misspelled Brands (Typosquatting)**")
                    for m, b in findings["misspelled_brands"]:
                        md_lines.append(f"- `{m}` → {b}")
                    md_lines.append("")

                if md_lines:
                    st.markdown("\n".join(md_lines))


if __name__ == "__main__":
    main()
