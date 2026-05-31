"""
Vision-Augmented LLM Pipeline for Long-Form Document Summarization — Language Detector
═════════════════════════════════
Detects text language and routes to the best HuggingFace summarisation
model.  Uses ``langdetect`` with a fixed seed for reproducibility.
"""

from langdetect import detect as _detect, DetectorFactory, LangDetectException

# Reproducibility — must be set before any detection call
DetectorFactory.seed = 42


# ═══════════════════════════════════════════════════════════════════════════
#  Supported Languages Table
# ═══════════════════════════════════════════════════════════════════════════
#  code → (name, flag_emoji, recommended_huggingface_model)
SUPPORTED_LANGUAGES: dict[str, tuple[str, str, str]] = {
    "en": ("English",  "🇬🇧", "google/pegasus-cnn_dailymail"),
    "ta": ("Tamil",    "🇮🇳", "csebuetnlp/mT5_multilingual_XLSum"),
    "hi": ("Hindi",    "🇮🇳", "csebuetnlp/mT5_multilingual_XLSum"),
    "fr": ("French",   "🇫🇷", "csebuetnlp/mT5_multilingual_XLSum"),
    "de": ("German",   "🇩🇪", "csebuetnlp/mT5_multilingual_XLSum"),
    "es": ("Spanish",  "🇪🇸", "csebuetnlp/mT5_multilingual_XLSum"),
    "zh": ("Chinese",  "🇨🇳", "csebuetnlp/mT5_multilingual_XLSum"),
    "ar": ("Arabic",   "🇸🇦", "csebuetnlp/mT5_multilingual_XLSum"),
}

# Helsinki NLP translation model mapping (source → English)
TRANSLATION_MODELS: dict[str, str] = {
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "de": "Helsinki-NLP/opus-mt-de-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "zh": "Helsinki-NLP/opus-mt-zh-en",
    "ar": "Helsinki-NLP/opus-mt-ar-en",
    "ta": "Helsinki-NLP/opus-mt-mul-en",
    "hi": "Helsinki-NLP/opus-mt-hi-en",
}


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════
def detect_language(text: str) -> dict:
    """
    Detect the language of *text* and return a routing dict.

    Returns
    -------
    dict
        code              – ISO 639-1 code (e.g. "en", "ta")
        name              – human-readable name
        recommended_model – HuggingFace model identifier
        is_supported      – True if language is in SUPPORTED_LANGUAGES
        flag              – emoji flag
    """
    # Guard: text too short for reliable detection
    if not text or len(text.split()) < 20:
        return {
            "code": "unknown",
            "name": "Unknown",
            "recommended_model": "csebuetnlp/mT5_multilingual_XLSum",
            "is_supported": False,
            "flag": "🌐",
        }

    try:
        code = _detect(text)
    except LangDetectException:
        # Detection failure → default to English silently
        return {
            "code": "en",
            "name": "English",
            "recommended_model": "google/pegasus-cnn_dailymail",
            "is_supported": True,
            "flag": "🇬🇧",
        }

    # Normalise Chinese variants
    if code.startswith("zh"):
        code = "zh"

    if code in SUPPORTED_LANGUAGES:
        name, flag, model = SUPPORTED_LANGUAGES[code]
        return {
            "code": code,
            "name": name,
            "recommended_model": model,
            "is_supported": True,
            "flag": flag,
        }

    # Unsupported language — use mT5 as fallback
    return {
        "code": code,
        "name": code.upper(),
        "recommended_model": "csebuetnlp/mT5_multilingual_XLSum",
        "is_supported": False,
        "flag": "🌐",
    }


def needs_translation(lang_code: str, target_lang: str = "en") -> bool:
    """Return True if *lang_code* differs from *target_lang*."""
    return lang_code != target_lang
