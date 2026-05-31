"""
ContextFree-X FastAPI Bridge Server
====================================
Exposes all backend features as REST endpoints so the Stitch HTML/JS UI
can call them directly.  Run alongside Streamlit:

    uvicorn api_server:app --port 8502 --reload

Streamlit is NOT modified.
"""

from __future__ import annotations

import io
import json
import os
import sys
import traceback

# ── Make project root importable ─────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="ContextFree-X API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # open for local dev
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy imports (models load on first use) ───────────────────────────────────
from summarizer.models   import list_models, get_model_label, get_model_color
from summarizer.pipeline import summarize, summarize_long
from input_handlers.cleaner        import clean_text as deep_clean
from input_handlers.pdf_handler    import extract_pdf
from input_handlers.url_handler    import scrape_url
from input_handlers.youtube_handler import extract_youtube, is_valid_youtube_url
from input_handlers.arxiv_handler  import fetch_arxiv
from input_handlers.lang_detector  import detect_language, SUPPORTED_LANGUAGES, TRANSLATION_MODELS
from evaluation.aggregator import (
    evaluate_all, composite_score, pick_best_model,
    rank_models, generate_recommendation_text,
    abstractiveness_score, compression_ratio,
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

HISTORY_PATH = os.path.join(ROOT, "summary_history.json")


def _load_history() -> list:
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _metrics_for(summary: str, reference: str, source: str) -> dict:
    scores = evaluate_all([summary], [reference])
    scores["abstractiveness"] = abstractiveness_score(summary, source)
    scores["compression"]     = compression_ratio(summary, source)
    scores["composite"]       = composite_score(scores)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "models": list_models()}


# ── Model registry ────────────────────────────────────────────────────────────
@app.get("/api/models")
def get_models():
    return [
        {
            "key":   m,
            "label": get_model_label(m),
            "color": get_model_color(m),
        }
        for m in list_models()
    ]


# ── Summarize: plain text ─────────────────────────────────────────────────────
class SummarizeTextRequest(BaseModel):
    text:        str
    model:       str  = "bart-cnn"
    length_mode: str  = "standard"   # brief | standard | detailed
    reference:   Optional[str] = None


@app.post("/api/summarize")
def summarize_text(req: SummarizeTextRequest):
    try:
        cleaned = deep_clean(req.text)
        if len(cleaned.split()) < 10:
            raise HTTPException(400, "Text too short (need ≥ 10 words).")
        use_long = len(cleaned.split()) > 900
        summary  = summarize_long(cleaned, req.model) if use_long else summarize(cleaned, req.model)
        ref      = req.reference.strip() if req.reference else cleaned
        metrics  = _metrics_for(summary, ref, cleaned)
        return {"summary": summary, "metrics": metrics, "word_count": len(summary.split())}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Summarize: PDF upload ─────────────────────────────────────────────────────
@app.post("/api/summarize/pdf")
async def summarize_pdf(
    file:        UploadFile = File(...),
    model:       str        = Form("bart-cnn"),
    section:     str        = Form("full_text"),   # full_text | abstract | introduction | conclusion
    length_mode: str        = Form("standard"),
):
    try:
        data = await file.read()
        buf  = io.BytesIO(data)
        buf.name = file.filename or "upload.pdf"

        result = extract_pdf(buf)
        if not result:
            raise HTTPException(400, "Could not extract text from PDF.")

        sections = result.get("sections", {})
        is_paper = result.get("is_research_paper", False)

        # Pick the right section text
        if is_paper and section in sections and sections[section]:
            input_text = sections[section]
        else:
            input_text = sections.get("full_text", "")

        if not input_text:
            raise HTTPException(400, "No text extracted from PDF.")

        cleaned = deep_clean(input_text)
        use_long = len(cleaned.split()) > 900
        summary  = summarize_long(cleaned, model) if use_long else summarize(cleaned, model)
        metrics  = _metrics_for(summary, cleaned, cleaned)

        return {
            "summary":          summary,
            "metrics":          metrics,
            "is_research_paper": is_paper,
            "sections_available": list(sections.keys()),
            "word_count":       len(summary.split()),
            "input_word_count": len(cleaned.split()),
            "preview":          cleaned[:1000],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Summarize: URL scraper ────────────────────────────────────────────────────
class URLRequest(BaseModel):
    url:   str
    model: str = "bart-cnn"


@app.post("/api/summarize/url")
def summarize_url(req: URLRequest):
    try:
        result     = scrape_url(req.url)
        input_text = result.get("text", "")
        if not input_text:
            raise HTTPException(400, "Could not extract text from URL.")
        cleaned  = deep_clean(input_text)
        use_long = len(cleaned.split()) > 900
        summary  = summarize_long(cleaned, req.model) if use_long else summarize(cleaned, req.model)
        metrics  = _metrics_for(summary, cleaned, cleaned)
        return {
            "summary":   summary,
            "metrics":   metrics,
            "title":     result.get("title", ""),
            "preview":   cleaned[:800],
            "word_count": len(summary.split()),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Fetch URL text only (for preview before summarize) ───────────────────────
@app.post("/api/fetch/url")
def fetch_url_text(req: URLRequest):
    try:
        result     = scrape_url(req.url)
        input_text = result.get("text", "")
        if not input_text:
            raise HTTPException(400, "Could not extract text from URL.")
        return {
            "title":      result.get("title", ""),
            "preview":    input_text[:800],
            "word_count": len(input_text.split()),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Summarize: YouTube transcript ─────────────────────────────────────────────
class YouTubeRequest(BaseModel):
    url:   str
    model: str = "bart-cnn"


@app.post("/api/summarize/youtube")
def summarize_youtube(req: YouTubeRequest):
    try:
        if not is_valid_youtube_url(req.url):
            raise HTTPException(400, "Invalid YouTube URL.")
        yt_result  = extract_youtube(req.url)
        input_text = yt_result.get("transcript", "")
        if not input_text:
            raise HTTPException(400, "No transcript found for this video.")
        cleaned  = deep_clean(input_text)
        use_long = len(cleaned.split()) > 900
        summary  = summarize_long(cleaned, req.model) if use_long else summarize(cleaned, req.model)
        metrics  = _metrics_for(summary, cleaned, cleaned)
        return {
            "summary":          summary,
            "metrics":          metrics,
            "word_count":       len(summary.split()),
            "transcript_words": yt_result.get("word_count", 0),
            "duration_seconds": yt_result.get("duration_seconds"),
            "language":         yt_result.get("language", "en"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Summarize: arXiv paper ────────────────────────────────────────────────────
class ArXivRequest(BaseModel):
    query:  str                   # arXiv ID, URL, or DOI
    model:  str = "bart-cnn"
    source: str = "abstract"      # abstract | full_paper


@app.post("/api/summarize/arxiv")
def summarize_arxiv(req: ArXivRequest):
    try:
        paper = fetch_arxiv(req.query)
        if req.source == "full_paper" and paper.get("full_text"):
            input_text = paper["full_text"]
        else:
            input_text = paper["abstract"]
        cleaned  = deep_clean(input_text)
        use_long = len(cleaned.split()) > 900
        summary  = summarize_long(cleaned, req.model) if use_long else summarize(cleaned, req.model)
        metrics  = _metrics_for(summary, cleaned, cleaned)
        return {
            "summary":       summary,
            "metrics":       metrics,
            "title":         paper.get("title", ""),
            "authors":       paper.get("authors", []),
            "arxiv_id":      paper.get("arxiv_id", ""),
            "published":     paper.get("published", ""),
            "has_full_text": bool(paper.get("full_text")),
            "abstract_words": len(paper["abstract"].split()),
            "word_count":    len(summary.split()),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Compare models ────────────────────────────────────────────────────────────
class CompareRequest(BaseModel):
    text:      str
    models:    List[str] = []
    reference: Optional[str] = None


@app.post("/api/compare")
def compare_models(req: CompareRequest):
    try:
        cleaned   = deep_clean(req.text)
        if len(cleaned.split()) < 10:
            raise HTTPException(400, "Text too short.")
        use_long  = len(cleaned.split()) > 900
        models    = req.models if req.models else [m for m in list_models() if m != "mt5"]
        reference = req.reference.strip() if req.reference else cleaned

        results = {}
        for mk in models:
            summary = summarize_long(cleaned, mk) if use_long else summarize(cleaned, mk)
            metrics = evaluate_all([summary], [reference])
            metrics["abstractiveness"] = abstractiveness_score(summary, cleaned)
            metrics["compression"]     = compression_ratio(summary, cleaned)
            metrics["composite"]       = composite_score(metrics)
            results[mk] = {
                "label":   get_model_label(mk),
                "color":   get_model_color(mk),
                "summary": summary,
                "metrics": metrics,
                "words":   len(summary.split()),
            }

        best = pick_best_model({mk: v["metrics"] for mk, v in results.items()})
        recommendation = generate_recommendation_text({mk: v["metrics"] for mk, v in results.items()})

        return {
            "results":        results,
            "best_model":     best,
            "recommendation": recommendation,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Metrics dashboard ─────────────────────────────────────────────────────────
class MetricsRequest(BaseModel):
    text:      str
    models:    List[str] = []
    reference: Optional[str] = None


@app.post("/api/metrics")
def metrics_dashboard(req: MetricsRequest):
    try:
        cleaned   = deep_clean(req.text)
        if len(cleaned.split()) < 10:
            raise HTTPException(400, "Text too short.")
        models    = req.models if req.models else [m for m in list_models() if m != "mt5"]
        reference = req.reference.strip() if req.reference else cleaned

        all_scores     = {}
        all_summaries  = {}
        novel_metrics  = {}

        for mk in models:
            summary = summarize(cleaned, mk)
            all_summaries[mk] = summary
            scores = evaluate_all([summary], [reference])
            all_scores[mk]    = scores
            novel_metrics[mk] = {
                "abstractiveness": abstractiveness_score(summary, cleaned),
                "compression":     compression_ratio(summary, cleaned),
            }
            all_scores[mk]["composite"] = composite_score(scores)

        ranking        = rank_models(all_scores) if all_scores else []
        recommendation = generate_recommendation_text(all_scores)

        return {
            "scores":         all_scores,
            "summaries":      all_summaries,
            "novel_metrics":  novel_metrics,
            "ranking":        ranking,
            "recommendation": recommendation,
            "model_meta": {
                mk: {"label": get_model_label(mk), "color": get_model_color(mk)}
                for mk in models
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Batch evaluate ────────────────────────────────────────────────────────────
@app.post("/api/batch")
async def batch_evaluate(
    file:     UploadFile = File(...),
    model:    str        = Form("bart-cnn"),
    max_rows: int        = Form(50),
):
    try:
        import pandas as pd
        data  = await file.read()
        df    = pd.read_csv(io.BytesIO(data))
        if "text" not in df.columns:
            raise HTTPException(400, "CSV must have a 'text' column.")
        has_ref  = "reference" in df.columns
        max_rows = min(max_rows, len(df))
        results  = []

        for idx in range(max_rows):
            row     = df.iloc[idx]
            text    = deep_clean(str(row["text"]))
            ref     = str(row.get("reference", "")) if has_ref else text
            summary = summarize(text, model)
            scores  = evaluate_all([summary], [ref if ref.strip() else text])
            scores["composite"] = composite_score(scores)
            results.append({
                "row":     idx + 1,
                "preview": text[:120] + "…",
                "summary": summary,
                **scores,
            })

        result_df = pd.DataFrame(results)
        avg = {
            col: float(result_df[col].mean())
            for col in result_df.columns
            if result_df[col].dtype in ["float64", "float32"]
        }
        return {
            "model":       model,
            "model_label": get_model_label(model),
            "rows_processed": max_rows,
            "averages":    avg,
            "results":     results,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Multilingual ──────────────────────────────────────────────────────────────
class MultilingualRequest(BaseModel):
    text:      str
    lang_override: Optional[str] = None   # ISO 639-1 code, e.g. "fr"


@app.post("/api/multilingual")
def multilingual_summarize(req: MultilingualRequest):
    try:
        cleaned   = deep_clean(req.text)
        if len(cleaned.split()) < 5:
            raise HTTPException(400, "Text too short.")

        lang_result = detect_language(req.text)
        lang_code   = req.lang_override or lang_result["code"]

        # Normalise
        if lang_code not in SUPPORTED_LANGUAGES:
            lang_code = "en"

        lang_name, flag, _ = SUPPORTED_LANGUAGES.get(lang_code, ("English", "🇬🇧", ""))
        model_key = "mt5" if lang_code != "en" else "pegasus"

        summary = summarize(cleaned, model_key)

        # Translate non-English → English
        translated = None
        if lang_code != "en":
            try:
                import torch
                from transformers import MarianMTModel, MarianTokenizer
                model_name = TRANSLATION_MODELS.get(lang_code)
                if model_name:
                    tok = MarianTokenizer.from_pretrained(model_name)
                    mdl = MarianMTModel.from_pretrained(model_name)
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    mdl = mdl.to(device).eval()
                    inputs = tok(summary, return_tensors="pt",
                                 max_length=512, truncation=True).to(device)
                    with torch.no_grad():
                        out = mdl.generate(inputs["input_ids"],
                                           max_length=512, num_beams=4)
                    translated = tok.decode(out[0], skip_special_tokens=True)
            except Exception:
                translated = "(Translation unavailable)"

        # BERTScore
        try:
            from evaluation.bert_scorer import compute_bertscore
            bert_lang = lang_code.split("-")[0]
            bert = compute_bertscore([summary], [cleaned], lang=bert_lang)
        except Exception:
            bert = {}

        return {
            "summary":            summary,
            "translated_summary": translated,
            "lang_code":          lang_code,
            "lang_name":          lang_name,
            "flag":               flag,
            "model_used":         model_key,
            "model_label":        get_model_label(model_key),
            "bert_scores":        bert,
            "needs_translation":  lang_code != "en",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Language detection only ───────────────────────────────────────────────────
class DetectLangRequest(BaseModel):
    text: str


@app.post("/api/detect-language")
def detect_lang(req: DetectLangRequest):
    try:
        result = detect_language(req.text)
        supported = list(SUPPORTED_LANGUAGES.keys())
        return {**result, "supported_languages": supported}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Blueprint summarizer ──────────────────────────────────────────────────────
@app.post("/api/blueprint")
async def blueprint_analyze(
    file:   UploadFile = File(...),
    method: str        = Form("auto"),
    unit:   str        = Form("sq ft"),
):
    try:
        from vision_summarizer.blueprint_reader    import load_blueprint
        from vision_summarizer.image_preprocessor  import preprocess_blueprint
        from vision_summarizer.vlm_extractor       import extract_with_best_available
        from vision_summarizer.room_parser         import validate_and_clean
        from vision_summarizer.area_calculator     import calculate_all_areas, get_room_breakdown
        from vision_summarizer.blueprint_summarizer import json_to_plain_english, generate_short_summary
        from vision_summarizer.blueprint_metrics   import (
            extraction_confidence, room_coverage_score,
            overall_quality_score, dimension_completeness,
        )

        data_bytes = await file.read()
        buf        = io.BytesIO(data_bytes)
        buf.name   = file.filename or "blueprint.png"

        pages      = load_blueprint(buf)
        processed  = preprocess_blueprint(pages[0])

        raw_data   = extract_with_best_available(processed, force_method=method)
        raw_data   = validate_and_clean(raw_data)
        data       = calculate_all_areas(raw_data)

        confidence = extraction_confidence(data)
        coverage   = room_coverage_score(data)
        quality    = overall_quality_score(data)

        plain      = json_to_plain_english(data)
        short      = generate_short_summary(plain, summarize)
        breakdown  = get_room_breakdown(data)
        dim_status = dimension_completeness(data)

        area_val   = (data.get("grand_total_sqm") if unit == "sq m"
                      else data.get("grand_total_sqft"))

        return {
            "short_summary":     short,
            "full_description":  plain,
            "confidence":        confidence,
            "coverage":          coverage,
            "quality":           quality,
            "engine_used":       data.get("_extraction_method", "unknown"),
            "total_area":        area_val,
            "unit":              unit,
            "num_bedrooms":      data.get("num_bedrooms"),
            "num_bathrooms":     data.get("num_bathrooms"),
            "num_floors":        data.get("num_floors"),
            "rooms_detected":    sum(len(f["rooms"]) for f in data.get("floors", [])),
            "room_breakdown":    breakdown,
            "special_features":  {
                "garage":   data.get("has_garage"),
                "basement": data.get("has_basement"),
                "garden":   data.get("has_garden"),
                "extras":   data.get("special_features", []),
            },
            "dimension_completeness": dim_status,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, traceback.format_exc())


# ── History ───────────────────────────────────────────────────────────────────
@app.get("/api/history")
def get_history(limit: int = 50):
    history = _load_history()
    return {"history": history[-limit:], "total": len(history)}


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8502, reload=True)
