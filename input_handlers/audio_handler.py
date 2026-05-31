"""
Vision-Augmented LLM Pipeline for Long-Form Document Summarization — Audio Transcription Handler
════════════════════════════════════════════
Uses OpenAI Whisper (local model, no API key needed) to transcribe
uploaded audio files into text for the summarisation pipeline.

Supported formats: mp3, wav, m4a, ogg, flac
"""

import os
import tempfile

import streamlit as st

from input_handlers.cleaner import clean_text, chunk_text


# ── Constraints ────────────────────────────────────────────────────────────
_MAX_FILE_SIZE_MB = 100
_SUPPORTED_EXTENSIONS = {"mp3", "wav", "m4a", "ogg", "flac"}


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════
def transcribe_audio(uploaded_file) -> dict:
    """
    Transcribe an uploaded audio file with Whisper.

    Parameters
    ----------
    uploaded_file : streamlit.UploadedFile
        Audio file (mp3, wav, m4a, ogg, flac).

    Returns
    -------
    dict
        type, source, duration_seconds, transcript, chunks,
        word_count, language_detected
    """
    filename = getattr(uploaded_file, "name", "audio_file")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # Validate format
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '.{ext}'. "
            f"Use {', '.join(sorted(_SUPPORTED_EXTENSIONS))}."
        )

    # Validate size
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)  # reset for potential re-read
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > _MAX_FILE_SIZE_MB:
        raise ValueError(
            f"Audio file too large ({size_mb:.1f} MB). Max {_MAX_FILE_SIZE_MB} MB."
        )

    # Write to temp file (Whisper needs a file path)
    tmp = tempfile.NamedTemporaryFile(
        suffix=f".{ext}", delete=False
    )
    try:
        tmp.write(file_bytes)
        tmp.close()

        # Get duration
        duration = get_audio_duration(tmp.name)

        # Transcribe
        model = get_whisper_model("base")
        result = model.transcribe(tmp.name)

        raw_text = result.get("text", "")
        lang_detected = result.get("language", "en")

        if not raw_text.strip():
            raise ValueError(
                "Transcription failed. Try a clearer audio file."
            )

    except ValueError:
        raise  # re-raise our own ValueErrors
    except Exception as exc:
        raise ValueError(
            f"Transcription failed: {exc}. Try a clearer audio file."
        ) from exc
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned, max_words=800, overlap=100)
    word_count = len(cleaned.split())

    return {
        "type": "audio",
        "source": filename,
        "duration_seconds": round(duration, 2) if duration else None,
        "transcript": cleaned,
        "chunks": chunks,
        "word_count": word_count,
        "language_detected": lang_detected,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════
def get_audio_duration(file_path: str) -> float:
    """Return duration in seconds using librosa."""
    try:
        import librosa
        duration = librosa.get_duration(path=file_path)
        return duration
    except Exception:
        return 0.0


@st.cache_resource
def get_whisper_model(size: str = "base"):
    """
    Load and cache a Whisper model.

    Sizes: "tiny", "base", "small", "medium"
    The model is cached via @st.cache_resource so it loads only once
    per Streamlit session.
    """
    import whisper
    return whisper.load_model(size)
