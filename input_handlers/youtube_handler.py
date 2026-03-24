"""
ContextFree-X — YouTube Transcript Handler
═══════════════════════════════════════════
Extracts transcripts from YouTube videos via the
``youtube-transcript-api`` package (no API key needed).

Compatible with youtube-transcript-api >= 1.0.0 (new instance-based API).

Supports:
  • youtube.com/watch?v=VIDEO_ID
  • youtu.be/VIDEO_ID
  • youtube.com/shorts/VIDEO_ID
"""

import re
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi

from input_handlers.cleaner import clean_text, chunk_text


# ── Maximum video duration (3 hours in seconds) ───────────────────────────
_MAX_DURATION_SECS = 3 * 60 * 60


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════
def extract_youtube(url: str) -> dict:
    """
    Fetch the transcript of a YouTube video.

    Parameters
    ----------
    url : str
        Any valid YouTube URL.

    Returns
    -------
    dict
        type, source, video_id, title, duration_seconds,
        transcript, chunks, word_count, language
    """
    if not is_valid_youtube_url(url):
        raise ValueError("Not a valid YouTube URL.")

    video_id = _extract_video_id(url)
    if not video_id:
        raise ValueError("Could not extract video ID from URL.")

    transcript_segments, lang = _get_transcript(video_id)

    # Join transcript segments into plain text (drop timestamps)
    raw_text = " ".join(seg.text for seg in transcript_segments)

    # Estimate duration from last segment
    duration_seconds = None
    if transcript_segments:
        last = transcript_segments[-1]
        duration_seconds = int(last.start + last.duration)

    if duration_seconds and duration_seconds > _MAX_DURATION_SECS:
        raise ValueError(
            "Video exceeds 3-hour limit. Please use a shorter video."
        )

    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned, max_words=800, overlap=100)
    word_count = len(cleaned.split())

    return {
        "type": "youtube",
        "source": url,
        "video_id": video_id,
        "title": None,  # transcript API doesn't provide title
        "duration_seconds": duration_seconds,
        "transcript": cleaned,
        "chunks": chunks,
        "word_count": word_count,
        "language": lang,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════
def _extract_video_id(url: str) -> str | None:
    """
    Extract the video ID from various YouTube URL formats.
    Returns the video ID string or None if extraction fails.
    """
    parsed = urlparse(url)

    # youtube.com/watch?v=VIDEO_ID
    if "youtube.com" in parsed.netloc:
        if parsed.path == "/watch":
            qs = parse_qs(parsed.query)
            return qs.get("v", [None])[0]
        # youtube.com/shorts/VIDEO_ID
        shorts_match = re.match(r"/shorts/([A-Za-z0-9_-]{11})", parsed.path)
        if shorts_match:
            return shorts_match.group(1)
        # youtube.com/embed/VIDEO_ID
        embed_match = re.match(r"/embed/([A-Za-z0-9_-]{11})", parsed.path)
        if embed_match:
            return embed_match.group(1)

    # youtu.be/VIDEO_ID
    if "youtu.be" in parsed.netloc:
        vid = parsed.path.lstrip("/")
        if vid:
            return vid.split("/")[0].split("?")[0]

    return None


def _get_transcript(
    video_id: str,
    preferred_lang: str = "en",
) -> tuple[list, str]:
    """
    Fetch transcript segments for *video_id*.

    Uses the youtube-transcript-api v1.x instance-based API:
      - ytt.fetch(video_id)  → FetchedTranscript (iterable of snippets)
      - ytt.list(video_id)   → TranscriptList (iterable of transcript info)

    Tries the preferred language first; falls back to any available
    language.

    Returns (list_of_segments, language_code).
    """
    ytt = YouTubeTranscriptApi()

    # Try fetching with preferred language
    try:
        result = ytt.fetch(video_id, languages=[preferred_lang])
        segments = list(result)
        return segments, result.language_code
    except Exception:
        pass

    # Fall back: list available transcripts and fetch the first one
    try:
        transcript_list = ytt.list(video_id)
        for transcript in transcript_list:
            try:
                result = transcript.fetch()
                segments = list(result)
                return segments, transcript.language_code
            except Exception:
                continue
    except Exception:
        pass

    # Nothing found
    raise ValueError(
        "This video has no transcript/subtitles available."
    )


def is_valid_youtube_url(url: str) -> bool:
    """Return True if *url* looks like a YouTube link."""
    if not url:
        return False
    return "youtube.com" in url or "youtu.be" in url
