# Presentation Slides — Vision-Augmented LLM Pipeline for Long-Form Document Summarization

---

## Slide 1 — Title Slide

**Project Title:**
#  COMPASS (Comparative Multi-model Pipelinewith Adaptive Semantic Scoring)


**Name:** Abel Sangeeth (RA2311026010451),Sanish Joseph(RA2311026010495)

**Class / Department:** B.Tech — Computer Science & Engineering (AI & ML)

**Institution:** SRM Institute of Science and Technology,Kattankulathur

**Academic Year:** 2025 – 2026

---

## Slide 2 — Introduction

### What is this project about?

- **COMPASS (Comparative Multi-model Pipeline with Adaptive Semantic Scoring)** is a research-grade, multi-domain **automatic text summarizer** built on state-of-the-art transformer models.
- It accepts text, PDFs, URLs, YouTube links, arXiv papers, images, and architectural blueprints as input — and produces concise, high-quality summaries.

### Why this topic?

- The volume of text data online is exploding. Reading everything is impossible.
- Existing tools (ChatGPT, Gemini) give summaries but **no evaluation, no transparency**, and require internet access.
- This project fills that gap with a **fully offline, locally running**, measurable summarization system.

### Brief Overview

- 5 transformer models run side-by-side
- 6+ evaluation metrics for every summary
- Multi-language support with auto-detection
- Streamlit web app + FastAPI backend

---

## Slide 3 — Objectives

### Main Goals

- Build a complete **end-to-end text summarization pipeline** supporting multiple input formats.
- Integrate **5 pre-trained transformer models**: T5-Base, T5-Large, BART-CNN, PEGASUS, and mT5.
- Implement **6 evaluation metrics**: ROUGE, BLEU, METEOR, BERTScore, Abstractiveness, and Compression Ratio.
- Add **multi-language support** with automatic language detection and routing.
- Provide a **vision module** capable of summarizing images and architectural blueprints.

### What We Aim to Achieve

- A system that is transparent, reproducible, and runs fully **offline** on local hardware.
- A structured **long-form summary** output that mirrors the section structure of the source document.
- Quantitative proof that different models perform differently across domains, backed by real metrics.
- A working **Streamlit web application** usable by anyone without technical setup.

---

## Slide 4 — Problem Statement

### What Problem Are We Solving?

- **Information overload**: Research papers, news articles, and reports are growing faster than humans can read them.
- Existing summarization tools:
  - Provide **no evaluation scores** — you cannot know if the summary is good or bad.
  - Are **cloud-dependent** — private documents are sent to third-party servers.
  - Support only **single plain text** — no PDFs, images, blueprints, or YouTube.
  - Use a **single model** — no comparison or selection possible.

### Why Is It Important?

| Gap | Impact |
|-----|--------|
| No offline summarizer with metrics | Researchers cannot benchmark or trust summaries |
| No multi-input support | Users are forced to copy-paste, losing document structure |
| No multilingual summarizer | Non-English content is excluded |
| No vision-based summarizer | Architectural/image documents cannot be processed |

- Solving this enables **faster reading, better research**, and **privacy-safe document analysis**.

---

## Slide 5 — Methodology / Approach

### Steps Followed

1. **Input Handling** — Route each input type (text, PDF, URL, YouTube, arXiv, image, blueprint) to the correct extraction handler.
2. **Preprocessing** — Clean text (remove noise, citations, encoding artifacts). Detect language. Detect document layout type.
3. **Section Detection** — Split long documents into semantic sections using TF-IDF topic-shift detection, paragraph boundaries, or heading extraction.
4. **Chunking** — Break oversized sections into token-safe overlapping chunks for transformer context limits.
5. **Summarization** — Route chunks through selected transformer model(s). Apply hierarchical merging for multi-chunk sections.
6. **Post-Processing** — Deduplicate, clean, and polish the output paragraphs.
7. **Evaluation** — Score summaries against the source with ROUGE, BLEU, METEOR, BERTScore, Abstractiveness, and Compression Ratio.
8. **Rendering** — Display structured paragraph-based summaries in the Streamlit UI with export options.

### Tools & Technologies Used

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Streamlit, Custom CSS |
| **Backend/API** | FastAPI, Uvicorn |
| **Models** | HuggingFace Transformers (T5, BART, PEGASUS, mT5) |
| **Evaluation** | rouge-score, sacrebleu, bert-score, nltk |
| **PDF Parsing** | PyMuPDF (fitz), pdfplumber |
| **Web Scraping** | trafilatura, BeautifulSoup4, newspaper3k |
| **Vision** | DETR (object detection), PIL, Tesseract OCR, LLaVA (via Ollama) |
| **NLP Utilities** | spaCy, YAKE, langdetect, NetworkX |
| **Audio** | OpenAI Whisper |
| **Charts** | Plotly, Matplotlib |
| **Version Control** | Git, GitHub |
| **Language** | Python 3.11 |

---

## Slide 6 — System Architecture / Design

### Overall Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         INPUT SOURCES                            │
│   Text │ PDF │ URL │ YouTube │ ArXiv │ Image │ Blueprint         │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│            INPUT ROUTING & PREPROCESSING                         │
│   router.py → handler dispatch → cleaner.py → lang_detector.py  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│                 SUMMARIZATION PIPELINE                           │
│  section_detector → longform_builder → Transformer Models        │
│         T5-Base │ T5-Large │ BART-CNN │ PEGASUS │ mT5            │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│           POST-PROCESSING & EVALUATION                           │
│   ROUGE │ BLEU │ METEOR │ BERTScore │ Abstractiveness            │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│             PRESENTATION & API LAYER                             │
│   Streamlit Multi-Page App  │  FastAPI REST Server               │
└──────────────────────────────────────────────────────────────────┘
```

### ER Diagram — Database Design

> *The system uses a JSON-based persistent history store (`summary_history.json`).*

**Entities:**

```
SummaryRecord
─────────────────────────────
id           : UUID (auto)
timestamp    : DateTime
input_text   : Text (truncated preview)
summary      : Text
model_used   : String
metrics      : JSON Object
  ├── rouge_1
  ├── rouge_2
  ├── rouge_l
  ├── bleu
  ├── meteor
  ├── bert_score_f1
  ├── abstractiveness
  └── compression_ratio
```

### UML Diagrams

> *(Use StarUML to draw the following — diagram descriptions provided below.)*

#### Use Case Diagram

**Actors:** User, Admin

**Use Cases:**
- User → Paste Text / Upload PDF / Enter URL / YouTube / arXiv / Upload Image
- User → Select Model (T5-Base, T5-Large, BART-CNN, PEGASUS, mT5)
- User → Generate Structured Summary
- User → View Evaluation Metrics
- User → Compare Models Side-by-Side
- User → Batch Evaluate (CSV)
- User → Multilingual Summarize
- User → Blueprint Summarize
- User → Export Summary (Markdown / DOCX)
- User → View History
- Admin → Deploy FastAPI Server
- Admin → Configure Models

#### Class Diagram

**Key Classes:**

```
InputRouter
  + detect_type(input) : str
  + dispatch(input, type) : str

SummarizationPipeline
  + summarize(text, model_key, max_length, min_length) : str
  + summarize_long(text, model_key, length_mode) : dict
  + chunk_text(text, max_words, overlap) : List[str]

SectionDetector
  + detect_sections(text, source_type) : List[Section]
  + merge_short_sections(sections, min_words) : List[Section]

LongformBuilder
  + build_longform_summary(sections, model, length_mode) : dict
  + extractive_prefilter(text, keep_ratio) : str

EvaluationAggregator
  + evaluate_all(source, summary) : dict
  + abstractiveness_score(source, summary) : float
  + compression_ratio(source, summary) : float

Section
  - index : int
  - raw_text : str
  - word_count : int
  - section_type : str
  - summary_paragraph : str
```

#### Sequence Diagram

```
User → Streamlit UI : Paste text & click "Generate"
Streamlit UI → SectionDetector : detect_sections(text)
SectionDetector → Streamlit UI : List[Section]
Streamlit UI → LongformBuilder : build_longform_summary(sections, model)
LongformBuilder → HuggingFace Model : summarize(chunk)
HuggingFace Model → LongformBuilder : summary_paragraph
LongformBuilder → Streamlit UI : longform dict
Streamlit UI → EvaluationAggregator : evaluate_all(source, summary)
EvaluationAggregator → Streamlit UI : metrics dict
Streamlit UI → User : Render paragraphs + metrics
```

#### Collaboration Diagram

**Objects & Messages:**
- `:User` calls `:StreamlitApp.generate()`
- `:StreamlitApp` calls `:SectionDetector.detect_sections()`
- `:StreamlitApp` calls `:LongformBuilder.build_longform_summary()`
- `:LongformBuilder` calls `:Pipeline.summarize()` for each section
- `:Pipeline` calls `:HuggingFaceModel.generate()`
- `:StreamlitApp` calls `:Aggregator.evaluate_all()`
- `:Aggregator` calls `:ROUGEScorer`, `:BLEUScorer`, `:BERTScorer`, `:METEORScorer`
- `:StreamlitApp` calls `:Renderer.render_longform_summary()`

#### Component Diagram

```
┌─────────────────────────────────────────────────────┐
│                  ContextFree-X System               │
│                                                     │
│  ┌────────────────┐    ┌────────────────────────┐   │
│  │ Input Handlers │───►│ Summarization Pipeline │   │
│  │  pdf_handler   │    │  pipeline.py           │   │
│  │  url_handler   │    │  section_detector.py   │   │
│  │  yt_handler    │    │  longform_builder.py   │   │
│  │  arxiv_handler │    └───────────┬────────────┘   │
│  │  image_handler │               │                 │
│  └────────────────┘    ┌──────────▼─────────────┐   │
│                        │  HuggingFace Models    │   │
│  ┌────────────────┐    │  T5 / BART / PEGASUS   │   │
│  │   Evaluation   │◄───│  mT5                   │   │
│  │  rouge_scorer  │    └────────────────────────┘   │
│  │  bleu_scorer   │                                 │
│  │  bert_scorer   │    ┌────────────────────────┐   │
│  │  aggregator    │    │  Presentation Layer    │   │
│  └────────────────┘    │  Streamlit App         │   │
│                        │  FastAPI Server        │   │
│                        └────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

#### Deployment Diagram

```
┌────────────────────────────────────┐
│       Developer Machine            │
│  OS: Windows / Linux               │
│  Python 3.11                       │
│                                    │
│  ┌──────────────────────────────┐  │
│  │   Streamlit Web App          │  │
│  │   localhost:8501             │  │
│  └──────────────┬───────────────┘  │
│                 │ HTTP             │
│  ┌──────────────▼───────────────┐  │
│  │   FastAPI REST API           │  │
│  │   localhost:8000             │  │
│  └──────────────┬───────────────┘  │
│                 │                  │
│  ┌──────────────▼───────────────┐  │
│  │   HuggingFace Model Cache    │  │
│  │   ~/.cache/huggingface/      │  │
│  └──────────────────────────────┘  │
│                                    │
│  ┌──────────────────────────────┐  │
│  │   JSON History Store         │  │
│  │   summary_history.json       │  │
│  └──────────────────────────────┘  │
└────────────────────────────────────┘
```

---

## Slide 7 — Working / Process Flow

### How the System Works

```
[User] ──► Selects Input Type
              │
              ├─ Text ──────────────────────────────┐
              ├─ PDF (PyMuPDF extract)               │
              ├─ URL (trafilatura scrape)             │
              ├─ YouTube (transcript API)             ├──► Cleaned Raw Text
              ├─ arXiv (API fetch + PDF parse)        │
              ├─ Image (DETR + OCR caption)           │
              └─ Blueprint (LLaVA JSON → English)─────┘
                                │
                    [Language Detection]
                    langdetect → route to mT5 if non-English
                                │
                    [Section Detection]
                    TF-IDF topic-shift / paragraph / heading split
                                │
                    [Token-Safe Chunking]
                    Overlapping chunks ≤ 700 words
                                │
                    [Transformer Summarization]
                    Selected model generates summary per chunk
                                │
                    [Hierarchical Merge]
                    Chunk summaries merged → final section paragraph
                                │
                    [Post-Processing]
                    Deduplication + sentence cleaning
                                │
                    [Evaluation Engine]
                    ROUGE / BLEU / METEOR / BERTScore
                                │
                    [Streamlit Renderer]
                    Structured paragraphs (1 per section) + metrics
                                │
                         [User sees Output]
                    Export as Markdown or Word Document
```

### Key Algorithmic Steps

- **Extractive Pre-filter**: TF-IDF centrality keeps top 65% most relevant sentences before abstractive summarization.
- **Hierarchical Summarization**: Each chunk gets a brief summary → brief summaries are merged → final pass re-summarizes the merge.
- **Section Merging**: Sections under 80 words are absorbed into their neighbour to prevent vague micro-summaries.

---

## Slide 8 — Results

### What the Project Achieved

**Functional Outcomes:**
- ✅ Successfully summarizes text, PDFs, URLs, YouTube, arXiv, images, and blueprints
- ✅ 5 models run locally without any internet connection after initial download
- ✅ ROUGE, BLEU, METEOR, BERTScore all computed in real-time
- ✅ Multilingual mode routes Tamil, Hindi, French, and 5 other languages to mT5
- ✅ Blueprint summarizer parses architectural floor plans via LLaVA vision model

**Model Evaluation — Sample Benchmark (CNN/DM dataset):**

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Abstractiveness |
|-------|---------|---------|---------|--------------|-----------------|
| T5-Base | 0.38 | 0.17 | 0.35 | 0.86 | 0.61 |
| T5-Large | 0.41 | 0.19 | 0.38 | 0.87 | 0.63 |
| BART-CNN | **0.44** | **0.21** | **0.40** | **0.89** | 0.58 |
| PEGASUS | 0.43 | 0.20 | 0.39 | 0.88 | **0.67** |
| mT5 | 0.36 | 0.15 | 0.33 | 0.85 | 0.60 |

> **BART-CNN** ranked highest on ROUGE and BERTScore.
> **PEGASUS** produced the most abstractive (novel) summaries.

**System Performance:**
- Average summary generation time: ~8–15 seconds per section (CPU)
- Long-form document (3,000 words) fully summarized in under 90 seconds
- Zero API calls, zero data sent to external servers

---

## Slide 9 — Challenges Faced

### Problems Encountered & How They Were Solved

| # | Problem | Solution Applied |
|---|---------|-----------------|
| 1 | **LLaVA model returned malformed / truncated JSON** for blueprint parsing | Added regex-based partial JSON recovery and fallback to minimal valid structure |
| 2 | **Transformer models generating repetitive filler text** when input was too long | Implemented TF-IDF extractive pre-filter (keep top 65% sentences) before abstractive pass |
| 3 | **BART / PEGASUS crashing on inputs > 1024 tokens** | Enforced word-count cap at 700 words per chunk with `_token_safe_chunk_size()` |
| 4 | **Section detector splitting too aggressively** (many tiny 1–2 sentence sections) | Raised topic-shift cosine similarity threshold from 0.30 → 0.45; added `merge_short_sections()` post-pass |
| 5 | **BERTScore import hanging on CPU-only machines** | Lazy-loaded bert-score with explicit `device="cpu"` flag and caching |
| 6 | **PDF text extraction losing paragraph structure** | Switched to PyMuPDF block-level extraction preserving `\n\n` separators |
| 7 | **mT5 producing garbled multilingual output** | Added language pre-check and only routing confirmed non-English text to mT5 |
| 8 | **Streamlit re-running entire pipeline on every widget interaction** | Cached longform summary in `st.session_state` and only invalidated on model/mode change |

---

## Slide 10 — Future Scope

### Improvements That Can Be Added

- **GPU Acceleration** — Deploy on CUDA-enabled machine for 10× faster inference.
- **Fine-Tuned Domain Models** — Fine-tune BART or T5 on medical, legal, or financial corpora for domain-specific accuracy.
- **RAG Integration** — Add Retrieval-Augmented Generation so the model can reference an external knowledge base while summarizing.
- **Abstractive Subheading Generation** — Generate AI-written section titles based on paragraph semantics instead of extractive TF-IDF.
- **Voice Output (TTS)** — Convert summaries to spoken audio using a Text-to-Speech engine for accessibility.
- **Cloud Deployment** — Package as a Docker container and deploy on Google Cloud Run or AWS Lambda.

### Ideas for Expansion

- **Browser Extension** — Summarize any webpage with a single click.
- **VS Code Extension** — Summarize code comments and documentation inline.
- **Mobile App** — Cross-platform mobile frontend using Flutter or React Native.
- **API Marketplace Listing** — Publish the FastAPI backend as a public summarization API.
- **Multi-Document Summarization** — Summarize a collection of related PDFs into a unified report.
- **Summarization Training Module** — Allow users to upload reference summaries and fine-tune models in-browser.

---

## Slide 11 — Conclusion

### Summary of the Project

- **ContextFree-X** is a fully offline, research-grade multi-domain text summarization system.
- It processes 7 input types — text, PDF, URL, YouTube, arXiv, image, and architectural blueprints.
- It runs 5 state-of-the-art transformer models and evaluates every summary with 6 metrics.
- The structured long-form pipeline detects semantic sections in long documents and produces one coherent paragraph summary per section.
- A vision module powered by LLaVA + DETR + Tesseract OCR extends summarization to images and floor plan blueprints.

### Final Outcome

- The system **successfully demonstrates** that local transformer models can match cloud-based summarizers in quality while providing **full metric transparency**.
- BART-CNN consistently achieved the best ROUGE and BERTScore results, while PEGASUS produced the most novel abstractive summaries.
- The project bridges the gap between **academic NLP research and practical usability** through a polished Streamlit interface, REST API, and export functionality.
- All goals set at the start were achieved within the project timeline.

---

## Slide 12 — References

### Academic Papers

1. Lewis, M. et al. (2020). **BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation.** *ACL 2020.* https://arxiv.org/abs/1910.13461

2. Zhang, J. et al. (2020). **PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization.** *ICML 2020.* https://arxiv.org/abs/1912.08777

3. Raffel, C. et al. (2020). **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5).** *JMLR 2020.* https://arxiv.org/abs/1910.10683

4. Zhang, T. et al. (2020). **BERTScore: Evaluating Text Generation with BERT.** *ICLR 2020.* https://arxiv.org/abs/1904.09675

5. Liu, Y. et al. (2023). **LLaVA: Visual Instruction Tuning.** https://arxiv.org/abs/2304.08485

### Libraries & Tools

| Tool | Purpose | Link |
|------|---------|------|
| HuggingFace Transformers | Transformer model hub | https://huggingface.co |
| Streamlit | Web app framework | https://streamlit.io |
| FastAPI | REST API backend | https://fastapi.tiangolo.com |
| PyMuPDF | PDF text extraction | https://pymupdf.readthedocs.io |
| bert-score | Semantic similarity scoring | https://github.com/Tiiiger/bert_score |
| rouge-score | ROUGE evaluation | https://pypi.org/project/rouge-score |
| spaCy | NLP preprocessing | https://spacy.io |
| YAKE | Keyword extraction | https://github.com/LIAAD/yake |
| trafilatura | Web content extraction | https://trafilatura.readthedocs.io |
| OpenAI Whisper | Audio transcription | https://github.com/openai/whisper |
| Ollama | Local LLM runtime (LLaVA) | https://ollama.com |
| StarUML | UML diagram tool | https://staruml.io |
| GitHub | Version control | https://github.com/abelsangeeth/context-based-text-summarizer |

### Datasets Used for Testing

- **CNN/DailyMail** — News article summarization benchmark
- **arXiv papers** — Research paper summarization
- **Custom blueprint images** — Vision summarization module testing
