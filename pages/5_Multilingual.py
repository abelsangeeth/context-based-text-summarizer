"""Page 5 — Multilingual: Language detection + mT5 summarization + translation."""

import streamlit as st
import torch
from summarizer.models import load_model, get_model_label
from summarizer.pipeline import summarize
from input_handlers.lang_detector import detect_language, SUPPORTED_LANGUAGES, TRANSLATION_MODELS
from input_handlers.cleaner import clean_text as deep_clean
from evaluation.bert_scorer import compute_bertscore
from evaluation.aggregator import evaluate_all
from transformers import AutoTokenizer, T5ForConditionalGeneration, BartForConditionalGeneration

from components.stitch_theme import apply_stitch_theme, page_header, sidebar_brand, sidebar_nav, sidebar_footer

st.set_page_config(page_title="Multilingual — ContextFree-X", page_icon="🌍", layout="wide")
apply_stitch_theme()

with st.sidebar:
    sidebar_brand()
    sidebar_nav(active="multilingual")
    sidebar_footer()

page_header("Multilingual Summarization", "Auto-detect language, summarize with the best model, and translate to English.")


@st.cache_resource(show_spinner=False)
def load_translation_model(src_lang: str):
    """Load Helsinki-NLP translation model for source→English."""
    from transformers import MarianMTModel, MarianTokenizer
    model_name = TRANSLATION_MODELS.get(src_lang)
    if not model_name:
        return None, None
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return tokenizer, model


def translate_to_english(text: str, src_lang: str) -> str:
    """Translate text from source language to English."""
    tok, model = load_translation_model(src_lang)
    if tok is None:
        return "(Translation model not available for this language)"
    device = next(model.parameters()).device
    inputs = tok(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        output_ids = model.generate(inputs["input_ids"], max_length=512, num_beams=4)
    return tok.decode(output_ids[0], skip_special_tokens=True)


# --- Input ---
input_text = st.text_area(
    "Enter text in any supported language",
    height=220,
    placeholder="Paste text in English, French, German, Spanish, Chinese, Arabic, Tamil, Hindi...",
)

# --- Language detection ---
if input_text.strip() and len(input_text.split()) >= 5:
    lang_result = detect_language(input_text)
    lang_code = lang_result["code"]
    lang_name = lang_result["name"]
    flag = lang_result["flag"]
    rec_model = lang_result["recommended_model"]

    st.markdown(
        f'<div class="cfx-lang-pill"><span class="cfx-lang-dot"></span>'
        f'{flag} Detected: <strong>{lang_name}</strong> ({lang_code})</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    # Manual override
    lang_options = list(SUPPORTED_LANGUAGES.keys())
    lang_labels = [f"{SUPPORTED_LANGUAGES[k][1]} {SUPPORTED_LANGUAGES[k][0]}" for k in lang_options]

    override = st.selectbox(
        "Override language (if detection is wrong)",
        lang_options,
        index=lang_options.index(lang_code) if lang_code in lang_options else 0,
        format_func=lambda k: f"{SUPPORTED_LANGUAGES[k][1]} {SUPPORTED_LANGUAGES[k][0]}",
    )

    if override != lang_code:
        lang_code = override
        lang_name, flag, _ = SUPPORTED_LANGUAGES[lang_code]
        rec_model = SUPPORTED_LANGUAGES[lang_code][2]

    # Disclaimer for Tamil and Hindi
    if lang_code in ("ta", "hi"):
        st.warning(
            f"**Note:** {lang_name} summarization uses mT5 multilingual model. "
            f"Quality may be lower than English-specific models due to limited "
            f"training data for {lang_name}. Results should be reviewed for accuracy.",
            icon="⚠️",
        )

    # Model routing
    if lang_code == "en":
        model_choices = ["pegasus", "bart-cnn", "t5-base", "t5-large"]
        model_key = st.selectbox("Model", model_choices, format_func=get_model_label)
    else:
        model_key = "mt5"
        st.info(f"Using **{get_model_label('mt5')}** for {lang_name} text.")

    generate = st.button("Summarize", type="primary",
                         disabled=len(input_text.split()) < 10)

    if generate:
        cleaned = deep_clean(input_text)

        with st.spinner(f"Summarizing with {get_model_label(model_key)}..."):
            summary = summarize(cleaned, model_key)

        # Side-by-side: original language summary vs English translation
        col_orig, col_en = st.columns(2)

        with col_orig:
            st.markdown(f'<div class="cfx-section-title">{flag} {lang_name} Summary</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="cfx-output-card"><div class="cfx-summary-text">{summary}</div></div>',
                unsafe_allow_html=True,
            )
            st.caption(f"{len(summary.split())} words")

        with col_en:
            if lang_code != "en":
                st.markdown('<div class="cfx-section-title">🇬🇧 English Translation</div>', unsafe_allow_html=True)
                with st.spinner("Translating to English..."):
                    translated = translate_to_english(summary, lang_code)
                st.markdown(
                    f'<div class="cfx-output-card"><div class="cfx-summary-text">{translated}</div></div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"{len(translated.split())} words")
            else:
                st.markdown("#### Metrics")
                with st.spinner("Computing BERTScore..."):
                    bert = compute_bertscore([summary], [cleaned], lang=lang_code)
                for metric, val in bert.items():
                    st.metric(metric, f"{val:.4f}")

        # BERTScore with correct language parameter
        if lang_code != "en":
            with st.expander("Evaluation metrics"):
                with st.spinner("Computing metrics..."):
                    # BERTScore uses detected language
                    bert_lang = lang_code.split("-")[0]  # handle zh-cn -> zh
                    bert = compute_bertscore([summary], [cleaned], lang=bert_lang)
                for metric, val in bert.items():
                    st.metric(metric, f"{val:.4f}")
