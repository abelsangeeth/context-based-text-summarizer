# COMPASS — Project Modules Reference

> **COMPASS** — Comparative Multi-model Pipeline with Adaptive Semantic Scoring  
> A research-grade, multi-domain text summarizer built with Streamlit and HuggingFace Transformers.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Entry Points](#2-entry-points)
3. [`summarizer/` — Core Summarization Engine](#3-summarizer--core-summarization-engine)
4. [`input_handlers/` — Input Ingestion Layer](#4-input_handlers--input-ingestion-layer)
5. [`evaluation/` — Metrics & Scoring](#5-evaluation--metrics--scoring)
6. [`vision_summarizer/` — Blueprint Analysis](#6-vision_summarizer--blueprint-analysis)
7. [`components/` — UI Components](#7-components--ui-components)
8. [`pages/` — Streamlit Pages](#8-pages--streamlit-pages)
9. [`modules/` — Experimental ML Modules](#9-modules--experimental-ml-modules)
10. [`config/` — Configuration](#10-config--configuration)
11. [Utility & Dev Scripts](#11-utility--dev-scripts)
12. [Data Flow Diagram](#12-data-flow-diagram)

---

## 1. Project Overview

COMPASS is a multi-model abstractive text summarizer that accepts diverse inputs (plain text, PDFs, URLs, YouTube videos, audio files, arXiv papers, and architectural blueprints), routes them through specialized handlers, and produces summaries using five transformer models: **T5-Base**, **T5-Large**, **BART-CNN**, **PEGASUS**, and **mT5 Multilingual**.

Every summary can be evaluated using six metrics — ROUGE, BLEU, BERTScore, METEOR, Abstractiveness, and Compression Ratio — and models are ranked by a configurable weighted composite score.

---

## 2. Entry Points

### `app.py`
**Role:** Streamlit root page / landing dashboard.

- Sets global page config (title, icon, layout).
- Applies the Stitch Material Design 3 theme via `components/stitch_theme.py`.
- Renders the hero section, feature cards (6 capabilities), and a comparison table showing how COMPASS differs from ChatGPT/Gemini.
- Points users to the sidebar pages.

### `main.py`
**Role:** Minimal CLI or standalone script launcher. Thin wrapper for non-Streamlit invocations.

### `api_server.py`
**Role:** FastAPI-based REST server (~25 KB) exposing the summarization pipeline over HTTP. Useful for integrating COMPASS into external systems without the Streamlit UI.

---

## 3. `summarizer/` — Core Summarization Engine

This is the heart of COMPASS. All model loading, inference, text preparation, and long-document handling live here.

---

### `summarizer/models.py`
**Role:** Unified model registry with lazy loading and GPU auto-detection.

| Function | Purpose |
|---|---|
| `load_model(model_key)` | Lazily loads and caches a `(tokenizer, model)` pair using `@st.cache_resource`. Supports T5, BART, PEGASUS, and mT5. |
| `get_model_config(model_key)` | Reads per-model settings (HF name, max tokens, colors) from `config/model_config.yaml`. |
| `get_device()` | Auto-detects CUDA GPU; falls back to CPU. |
| `list_models()` | Returns all registered model keys. |
| `get_model_label/color()` | UI helpers for human-readable labels and chart colors. |
| `get_weights_config()` | Returns metric weights from config for composite scoring. |

**Key design:** A `_MODEL_CLASS_MAP` dict maps key prefixes (`t5`, `bart`, `pegasus`, `mt5`) to the correct HuggingFace model class, so adding a new model only requires a YAML entry and no code changes.

---

### `summarizer/pipeline.py`
**Role:** Single-function public interface for all summarization calls.

| Function | Purpose |
|---|---|
| `summarize(text, model_key, **kwargs)` | Core inference function. Prepares input, tokenizes, runs beam-search generation, and postprocesses. Scales `min_length` dynamically based on input word count to avoid hallucination on short inputs. |
| `summarize_long(text, model_key)` | Iterative chunk-and-merge strategy for documents exceeding the model's token limit. Splits text → summarizes each chunk → merges → repeats up to 5 iterations until the combined summary fits in a single pass. |
| `_token_safe_chunk_size(model_key)` | Calculates a safe per-chunk word count (~1.5 tokens/word) to stay within the model's `max_input` limit. |

**Generation config:** 4-beam search, `no_repeat_ngram_size=3`, `length_penalty=2.0`, early stopping — tuned for quality over speed.

---

### `summarizer/preprocessing.py`
**Role:** Text cleaning and chunking before model input.

| Function | Purpose |
|---|---|
| `clean_text(text)` | Strips extra whitespace, citation brackets `[1]`, bullet characters, PDF page-number artifacts, and excess punctuation. |
| `chunk_text(text, max_words, overlap)` | Sliding-window chunker with configurable overlap (default 100 words) to preserve cross-chunk context. |
| `prepare_input(text, model_key)` | Calls `clean_text` then prepends `"summarize: "` prefix required by T5/mT5 models. |

---

### `summarizer/postprocessing.py`
**Role:** Cleans and normalizes raw model output.

| Function | Purpose |
|---|---|
| `remove_duplicate_sentences(text)` | Deduplicates sentences — common artifact in beam-search outputs. |
| `clean_summary(text)` | Normalizes whitespace, fixes punctuation spacing, and capitalizes the first letter. |
| `postprocess_summary(text)` | Full 7-step pipeline: dedup → remove repeated consecutive words (BART artifact) → capitalize → ensure sentence-ending punctuation → strip trailing ellipsis → collapse spaces → fix punctuation spacing. |

---

### `summarizer/section_detector.py`
**Role:** Splits documents into semantically coherent sections for long-form summarization.

| Function | Purpose |
|---|---|
| `auto_detect_source_type(text, metadata)` | Heuristically classifies text as `research_paper`, `news_article`, or `long_text` using keyword signals and average paragraph length. |
| `detect_sections(text, source_type)` | Dispatcher: routes to the appropriate splitter based on source type. |
| `_split_by_headings(text)` | For research papers: splits on Markdown/numbered/ALL-CAPS headings, labels sections by type (`introduction`, `methodology`, `results`, `conclusion`, `body`). |
| `_split_by_paragraphs(text, min_words)` | For news: merges short paragraphs until they reach `min_words`. |
| `_split_by_topic_shift(text)` | For long text: uses TF-IDF + cosine similarity in a sliding window to detect topic shifts (threshold 0.45). Splits when similarity drops OR section exceeds 500 words. |
| `_force_split_by_size(text, target_words)` | Last-resort sentence-boundary splitter for texts that produce only one section. |
| `_split_by_rooms(text)` | For blueprints: groups sentences by room keyword (bedroom, kitchen, etc.). |
| `merge_short_sections(sections, min_words)` | Post-split pass: merges sections under `min_words` (default 80) into their neighbour to prevent near-empty chunks. |

---

### `summarizer/subheading_generator.py`
**Role:** Generates a concise, relevant heading for each detected section.

**Three-tier candidate generation:**
1. **YAKE keyword extraction** → top-5 keywords filtered by noise list → joined as a noun phrase.
2. **TF-IDF centrality** (`tf_idf_centrality_heading`) → selects the most central sentence, strips stopwords, truncates to 8 words.
3. **AI abstractive candidate** → calls `summarize()` with `max_length=15` to generate a short abstractive heading. If successful, this is returned immediately (highest quality).

**Scoring fallback:** When AI generation fails, `get_best_heading()` uses `sentence-transformers/all-MiniLM-L6-v2` to score candidates against the section text and pick the most semantically similar one.

`format_subheading()` then sanitizes the result: strips markdown, citation noise, trailing function words, and caps at 7 words.

---

### `summarizer/longform_builder.py`
**Role:** Orchestrates per-section abstractive summarization to produce a full long-form structured summary.

| Function | Purpose |
|---|---|
| `extractive_prefilter(text, keep_ratio)` | TF-IDF pre-filter that keeps the top-scoring sentences before sending to BART/PEGASUS. Reduces noise and improves output quality. |
| `_summarize_section(text, model, ...)` | Summarizes a single chunk with mode-aware token bounds (`detailed/standard/brief`). |
| `_split_oversized_section(section, model)` | Splits sections exceeding the model's safe chunk size at sentence boundaries. |
| `_summarize_chunked_section(chunks, ...)` | Hierarchical: summarizes each sub-chunk briefly, then does a final pass over the combined summaries. |
| `build_longform_summary(sections, subheadings, model, length_mode)` | Main entry: iterates sections + headings, calls appropriate summarizer, postprocesses each paragraph, returns `{sections, total_word_count, model_used, length_mode}`. |
| `estimate_reading_time(word_count)` | Estimates reading time at 200 WPM. |

**Length modes:** `detailed` (max 450 tokens), `standard` (max 300), `brief` (max 150).

---

### `summarizer/advanced.py`
**Role:** Optional NLP enhancements — entity checking, highlight scoring, extractive summarization.

| Function | Purpose |
|---|---|
| `get_keywords(text, top_n)` | YAKE-based keyword extraction (2-gram, deduplicated). |
| `check_entities(source, summary)` | Uses SpaCy NER to verify that key named entities (PERSON, ORG, GPE, DATE) from the source appear in the summary. Returns `(preserved, missing)` lists. |
| `highlight_sentences(source, summary)` | Scores each source sentence by token overlap with the summary (Jaccard-style recall), normalized to 0–1. Used to highlight source sentences that contributed to the summary. |
| `extractive_summary(text, top_n)` | TextRank extractive summarizer using TF-IDF + NetworkX PageRank. Selects the top-N sentences in original order. |

---

## 4. `input_handlers/` — Input Ingestion Layer

All input types are normalized into a unified dict before being passed to the summarizer.

---

### `input_handlers/router.py`
**Role:** Single entry point for all Streamlit pages. Detects input type and dispatches to the correct handler.

```python
result = route_input(input_data)
text   = get_summary_target(result)
```

**Supported types:** `text`, `pdf`, `url`, `youtube`, `audio`, `arxiv`, `blueprint`, `batch`

**Auto-detection logic (`_detect_type`):**
- String → checks for YouTube URL → HTTP URL → arXiv ID pattern → plain text
- UploadedFile → routes by extension: `.mp3/.wav/...` → audio, `.csv` → batch, `.pdf` → pdf or blueprint, `.jpg/.png/...` → blueprint

**Output dict keys:** `type`, `source`, `summary_target`, `chunks`, `word_count`, `language`, `recommended_model`, `metadata`

After routing, the router also runs **language detection** and sets `recommended_model` (overrides to BART-CNN for news URLs).

---

### `input_handlers/pdf_handler.py`
**Role:** Extracts text from PDF files using PyMuPDF/pdfplumber. Detects if the PDF is a research paper and extracts key sections (Abstract, Introduction, Conclusion).

---

### `input_handlers/url_handler.py`
**Role:** Scrapes web pages using `requests` + `BeautifulSoup`. Detects news articles via domain/content heuristics and sets `recommended_model` to BART-CNN accordingly.

---

### `input_handlers/youtube_handler.py`
**Role:** Extracts transcripts from YouTube videos using the `youtube-transcript-api`. Returns the full transcript, video metadata, and duration.

---

### `input_handlers/audio_handler.py`
**Role:** Transcribes audio files (MP3, WAV, M4A, OGG, FLAC) using OpenAI Whisper locally. Returns the transcript and detected language.

---

### `input_handlers/arxiv_handler.py`
**Role:** Fetches arXiv papers by ID or URL using the `arxiv` Python library. Returns abstract, title, authors, publication date, and structured summary target.

---

### `input_handlers/blueprint_handler.py`
**Role:** Orchestrates blueprint image analysis. Delegates to `vision_summarizer/` for structured extraction, then formats results into a human-readable summary target for the text summarizer.

---

### `input_handlers/batch_handler.py`
**Role:** Parses CSV files for bulk evaluation. Auto-detects text and reference summary columns, returns a list of rows for batch processing.

---

### `input_handlers/text_handler.py`
**Role:** Handles raw plain text input. Classifies length (`short/long`) and returns chunks for long texts.

---

### `input_handlers/cleaner.py`
**Role:** Deep text cleaning used primarily after PDF/URL extraction. Handles encoding artifacts, boilerplate removal, and structure normalization (~8.5 KB of cleaning rules).

---

### `input_handlers/lang_detector.py`
**Role:** Detects text language using `langdetect` (seed=42 for reproducibility) and returns the recommended HuggingFace model.

**Supported languages:** English (PEGASUS), Tamil, Hindi, French, German, Spanish, Chinese, Arabic (all routed to mT5 multilingual).

Also provides a `TRANSLATION_MODELS` dict for Helsinki-NLP opus-mt models for each supported non-English language.

---

### `input_handlers/layout_classifier.py`
**Role:** Classifies the layout of a PDF page (single-column, multi-column, table-heavy) to guide extraction strategy.

---

### `input_handlers/region_handlers.py`
**Role:** Handles specific regions of a PDF page (e.g., extracting tables vs. body text separately).

---

## 5. `evaluation/` — Metrics & Scoring

---

### `evaluation/aggregator.py`
**Role:** Combines all metric scores, computes weighted composite rankings, and generates human-readable recommendations.

| Function | Purpose |
|---|---|
| `evaluate_all(predictions, references, lang)` | Runs all four scorers and returns a combined dict. |
| `composite_score(metrics)` | Weighted sum: ROUGE-1 (20%), ROUGE-2 (20%), ROUGE-L (15%), BLEU (10%), BERTScore F1 (25%), METEOR (10%). |
| `rank_models(all_model_scores)` | Returns models sorted by composite score descending. |
| `pick_best_model(all_model_scores)` | Returns the model key with the highest composite score. |
| `generate_recommendation_text(all_model_scores)` | Human-readable text identifying the best model and which model leads on semantic vs. n-gram metrics. |
| `abstractiveness_score(summary, source)` | Fraction of summary unigrams not found in source (novel n-gram novelty). Higher = more abstractive. |
| `compression_ratio(summary, source)` | `len(summary) / len(source)`. Lower = more compressed. |

---

### `evaluation/rouge_scorer.py`
Wraps `rouge_score` library to compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

### `evaluation/bleu_scorer.py`
Wraps `nltk.translate.bleu_score` for corpus-level BLEU.

### `evaluation/bert_scorer.py`
Wraps `bert_score` library (baseline-rescaled F1) for semantic similarity evaluation.

### `evaluation/meteor_scorer.py`
Wraps `nltk` METEOR score for recall-oriented evaluation with stemming and synonym matching.

---

## 6. `vision_summarizer/` — Blueprint Analysis

This sub-system enables COMPASS to analyze architectural blueprint images and extract structured room data.

---

### `vision_summarizer/vlm_extractor.py`
**Role:** Master VLM extraction engine with a three-tier fallback strategy.

| Tier | Method | Notes |
|---|---|---|
| **Tier 1** | `extract_with_claude(img)` | Claude Vision API (`claude-opus-4-5`). Best accuracy. Requires `ANTHROPIC_API_KEY`. Includes exponential-backoff retry. |
| **Tier 2** | `extract_with_llava(img)` | LLaVA 13B via local Ollama server. Free, offline. Retries with progressively stronger prompts if the initial extraction is empty. |
| **Tier 3** | `extract_with_ocr(img)` | pytesseract OCR fallback. Text-only; cannot infer spatial layout. Confidence: 0.40. |
| **Auto** | `extract_with_best_available(img)` | Tries Claude → LLaVA → OCR in order. |

All tiers use a shared `EXTRACTION_PROMPT` requesting a structured JSON output with floors, rooms, dimensions, and features.

`_safe_parse_json(text)` provides a robust 7-strategy JSON parser handling: markdown fences, trailing commas, truncated JSON, and extra prose — with auto-bracket-closing as a last resort.

---

### `vision_summarizer/blueprint_reader.py`
**Role:** Loads blueprint images from file upload or file path, preprocesses them (resize, enhance contrast), and passes them to `vlm_extractor`.

### `vision_summarizer/image_preprocessor.py`
**Role:** PIL-based image enhancement pipeline (grayscale conversion, contrast/sharpness enhancement, resizing) to improve OCR and VLM extraction accuracy.

### `vision_summarizer/room_parser.py`
**Role:** Post-processes the raw JSON from VLM extraction into clean, structured room objects. Handles unit conversions (mm → ft), area calculations, and data validation.

### `vision_summarizer/area_calculator.py`
**Role:** Computes room areas (width × length), floor totals, and building total area. Handles null dimensions gracefully.

### `vision_summarizer/blueprint_metrics.py`
**Role:** Derives high-level metrics from extracted data: bedroom/bathroom counts, has_garage/basement/garden flags, floor count, total area, and a confidence score.

### `vision_summarizer/blueprint_summarizer.py`
**Role:** Converts the structured blueprint data into a plain-English paragraph suitable for the text summarizer pipeline (e.g., "This 2-floor residential property has 3 bedrooms, 2 bathrooms...").

---

## 7. `components/` — UI Components

### `components/stitch_theme.py`
**Role:** Material Design 3-inspired global CSS theme (~970 lines). Injected into every page via `apply_stitch_theme()`.

Key exports:
- `apply_stitch_theme()` — injects all fonts (Manrope, Inter), Material Symbols icons, and CSS overrides.
- `sidebar_brand()` — renders the COMPASS logo and subtitle in the sidebar.
- `sidebar_nav(active)` — renders the navigation menu with active-page highlighting.
- `sidebar_footer()` — renders model info and version badge at the bottom of the sidebar.
- `page_header(title, subtitle)` — renders a consistent page title + subtitle.

CSS covers: sidebar, buttons, text inputs, selectboxes, metric containers, file uploader, tabs, data tables, progress bars, expanders, alerts, and a full set of reusable utility classes (`cfx-card`, `cfx-badge`, `cfx-metric-pill`, `cfx-model-chip`, `cfx-score-bar`, etc.).

### `components/model_card.py`
**Role:** Renders a styled model result card (summary text + metric scores + score bar) for the comparison page.

### `components/radar_chart.py`
**Role:** Generates a Plotly radar chart comparing all models across all metrics simultaneously.

### `components/longform_renderer.py`
**Role:** Renders long-form structured summaries as a series of section cards with subheadings and word counts.

### `components/export_utils.py`
**Role:** Utilities for exporting summaries and metric results as Markdown, JSON, or CSV downloads via Streamlit's `st.download_button`.

---

## 8. `pages/` — Streamlit Pages

| File | Page Name | Description |
|---|---|---|
| `1_Summarize.py` | **Summarize** | Main summarization page. Accepts all input types via the router. Supports short-form single-model and long-form multi-section output. Shows language detection, model recommendation, and inline metrics. |
| `2_Compare_Models.py` | **Compare Models** | Runs the same input through all 5 models simultaneously and shows a side-by-side card grid + radar chart + composite ranking table. |
| `3_Metrics_Dashboard.py` | **Metrics Dashboard** | Interactive dashboard for exploring metric scores for a custom input/reference pair. Shows bar charts and a recommendation text. |
| `4_Batch_Evaluate.py` | **Batch Evaluate** | Uploads a CSV of (text, reference) pairs, runs all models, exports a full metric results CSV. |
| `5_Multilingual.py` | **Multilingual** | Language-aware summarization page. Auto-detects language, offers translation preview, and routes to mT5 for non-English text. |
| `6_Blueprint_Summarizer.py` | **Blueprint Summarizer** | Blueprint image upload → VLM extraction → structured data display → plain-English summary generation. |
| `7_History.py` | **History** | Reads `summary_history.json` and displays past summarization sessions with timestamps, model used, and scores. |

---

## 9. `modules/` — Experimental ML Modules

A collection of standalone module stubs for a training/fine-tuning pipeline (currently experimental/scaffold):

| File | Purpose |
|---|---|
| `data_module.py` | Dataset loading and preprocessing abstraction. |
| `model_module.py` | Model instantiation and configuration for fine-tuning. |
| `training_module.py` | Training loop scaffold (Trainer API integration). |
| `inference_module.py` | Inference wrapper for fine-tuned models. |
| `preprocessing_module.py` | Training-time preprocessing (tokenization, padding). |
| `evaluation_module.py` | Evaluation loop using the `evaluate` library. |
| `model_comparison.py` | Side-by-side comparison helper for baseline vs. fine-tuned models. |

---

## 10. `config/` — Configuration

### `config/model_config.yaml`
Central configuration file for all registered models and metric weights.

```yaml
models:
  t5-base:
    hf_name: "t5-base"
    max_input: 512
    max_output: 512
    min_output: 80
    color: "#3B82F6"
    label: "T5 Base"

  t5-large:    { hf_name: "t5-large",                    max_input: 512,  color: "#10B981" }
  bart-cnn:    { hf_name: "facebook/bart-large-cnn",     max_input: 1024, color: "#F97316" }
  pegasus:     { hf_name: "google/pegasus-cnn_dailymail", max_input: 1024, color: "#8B5CF6" }
  mt5:         { hf_name: "csebuetnlp/mT5_multilingual_XLSum", max_input: 512, color: "#EF4444" }

weights:
  ROUGE-1: 0.20 | ROUGE-2: 0.20 | ROUGE-L: 0.15
  BLEU: 0.10    | BERTScore_F1: 0.25 | METEOR: 0.10
```

To add a new model: add a YAML block here. No Python code changes required.

---

## 11. Utility & Dev Scripts

| File | Purpose |
|---|---|
| `generate_bert_graphs.py` | Generates BERT score comparison bar charts and tables for all 5 models across test cases. Saves to `bert_graphs/`. |
| `generate_bug_reports.py` | Generates terminal-themed PNG bug report cards (pass/fail) for 13 test cases. Saves to `bug_reports/`. |
| `generate_heatmap.py` | Generates a metric heatmap across all models for visualization. Saves to `charts/`. |
| `compare_models.py` | CLI script to run all models on a given text and print a comparison table. |
| `debug_summaries.py` | Dev helper to quickly print summarizer output for debugging. |
| `app_history.py` | Reads/writes `summary_history.json` for session persistence. |
| `parse_builder.py` | Helper to build structured parse trees from extracted document sections. |
| `test_*.py` | 10+ unit and integration test scripts for scorers, section detection, subheadings, and preprocessing. |

---

## 12. Data Flow Diagram

```
User Input (text / PDF / URL / YouTube / Audio / arXiv / Blueprint / CSV)
        │
        ▼
input_handlers/router.py  ──►  Auto-detect type
        │
        ▼
Appropriate Handler
  ├─ text_handler    ─► clean & chunk
  ├─ pdf_handler     ─► PyMuPDF extract + section detect
  ├─ url_handler     ─► scrape + clean
  ├─ youtube_handler ─► transcript API
  ├─ audio_handler   ─► Whisper transcription
  ├─ arxiv_handler   ─► arXiv API fetch
  ├─ blueprint_handler ─► vision_summarizer pipeline
  └─ batch_handler   ─► CSV parse
        │
        ▼
lang_detector.py  ──►  Language → recommended_model
        │
        ▼
   Unified Result Dict
  { type, source, summary_target, chunks, word_count, language, recommended_model, metadata }
        │
        ▼
summarizer/pipeline.py
  ├─ preprocessing.py   (clean → chunk → prefix)
  ├─ models.py          (load tokenizer + model)
  ├─ HuggingFace generate()
  └─ postprocessing.py  (dedup → clean → punctuate)
        │
   [ Long-form path ]
        ├─ section_detector.py    (split into sections)
        ├─ subheading_generator.py (YAKE + TF-IDF + AI heading)
        └─ longform_builder.py    (per-section summarize → assemble)
        │
        ▼
    Summary Output
        │
        ▼
evaluation/aggregator.py
  ├─ rouge_scorer.py
  ├─ bleu_scorer.py
  ├─ bert_scorer.py
  └─ meteor_scorer.py
        │
        ▼
  Composite Score + Rankings + Recommendation
        │
        ▼
components/ + pages/  ──►  Streamlit UI rendering
```

---

*Generated: 2026-04-28 | COMPASS v1.0*
