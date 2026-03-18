"""Language detection with model routing for multilingual support."""

from langdetect import detect, DetectorFactory

DetectorFactory.seed = 42  # Reproducibility

SUPPORTED_LANGUAGES = {
    "en": ("English",  "🇬🇧", "pegasus"),
    "fr": ("French",   "🇫🇷", "mt5"),
    "de": ("German",   "🇩🇪", "mt5"),
    "es": ("Spanish",  "🇪🇸", "mt5"),
    "zh-cn": ("Chinese",  "🇨🇳", "mt5"),
    "ar": ("Arabic",   "🇸🇦", "mt5"),
    "ta": ("Tamil",    "🇮🇳", "mt5"),
    "hi": ("Hindi",    "🇮🇳", "mt5"),
}

# Helsinki NLP translation model mapping (source -> English)
TRANSLATION_MODELS = {
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "de": "Helsinki-NLP/opus-mt-de-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "zh-cn": "Helsinki-NLP/opus-mt-zh-en",
    "ar": "Helsinki-NLP/opus-mt-ar-en",
    "ta": "Helsinki-NLP/opus-mt-mul-en",
    "hi": "Helsinki-NLP/opus-mt-hi-en",
}


def detect_language(text: str) -> tuple[str, str, str, str]:
    """
    Detect the language of input text.
    Returns (lang_code, lang_name, flag_emoji, recommended_model_key).
    """
    code = detect(text)
    # Normalize Chinese variants
    if code.startswith("zh"):
        code = "zh-cn"
    if code in SUPPORTED_LANGUAGES:
        name, flag, model_key = SUPPORTED_LANGUAGES[code]
        return code, name, flag, model_key
    return code, "Unknown", "🌐", "mt5"
