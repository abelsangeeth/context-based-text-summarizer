# Long-Form Summary: How It Works

> A complete, code-level walkthrough of the long-form summarization pipeline —  
> from raw text → sections → subheadings → final structured summary document.

---

## Table of Contents

1. [Big Picture: The Overall Flow](#1-big-picture-the-overall-flow)
2. [Step 1 — Section Detection (`section_detector.py`)](#2-step-1--section-detection-section_detectorpy)
3. [Step 2 — Subheading Generator (`subheading_generator.py`)](#3-step-2--subheading-generator-subheading_generatorpy)
4. [Step 3 — Long-Form Builder (`longform_builder.py`)](#4-step-3--long-form-builder-longform_builderpy)
5. [The Core Summarization Engine (`pipeline.py`)](#5-the-core-summarization-engine-pipelinepy)
6. [Full Data Flow Diagram](#6-full-data-flow-diagram)
7. [Key Design Decisions](#7-key-design-decisions)

---

## 1. Big Picture: The Overall Flow

The long-form summary is **not** a single monolithic summarization. It is a structured pipeline of three stages:

```
Raw Text
   │
   ▼
[Section Detector] ──► splits text into logical sections (each is a dict with raw_text, word_count, section_type)
   │
   ▼
[Subheading Generator] ──► generates a title/heading for each section
   │
   ▼
[Long-Form Builder] ──► summarizes each section individually, then assembles the final structured output
   │
   ▼
Final Output: { sections: [{subheading, summary_paragraph, word_count}], total_word_count, model_used, length_mode }
```

Each stage is in its own file under `summarizer/`.

---

## 2. Step 1 — Section Detection (`section_detector.py`)

**Goal:** Split the raw input text into meaningful logical chunks (sections), each carrying metadata about its type.

### 2.1 — Auto-detecting the Source Type

```python
def auto_detect_source_type(text: str, metadata: dict = None) -> str:
```

Before splitting, the detector figures out what kind of text it is:

| Condition | Detected Type |
|---|---|
| Contains `"Abstract"` AND `"References"` | `research_paper` |
| Contains `"Reuters"` or `"AP"` | `news_article` |
| Average paragraph length < 80 words | `news_article` |
| Average paragraph length > 200 words | `long_text` |
| Fallback | `long_text` |

### 2.2 — Dispatching to the Right Splitter

```python
def detect_sections(text: str, source_type: str = "auto") -> list[dict]:
```

Based on the detected type, one of four splitter functions is called:

| Source Type | Splitter Used |
|---|---|
| `research_paper` | `_split_by_headings()` |
| `news_article` | `_split_by_paragraphs()` |
| `blueprint` | `_split_by_rooms()` |
| `long_text` (default) | `_split_by_topic_shift()` |

---

### 2.3 — `_split_by_headings()` (Research Papers)

Walks through lines and uses regex to detect headings:

```python
re.match(r"^#{1,3}\s.+", line)          # Markdown headings (## Introduction)
re.match(r"^\d+\.\s[A-Z].+", line)      # Numbered headings (1. Introduction)
re.match(r"^[A-Z][A-Z\s]{4,}$", line)  # ALL CAPS headings (ABSTRACT)
```

Each detected heading starts a new section. Section types are assigned by keyword matching:

| Heading keyword | `section_type` |
|---|---|
| `abstract`, `introduction` | `introduction` |
| `method`, `approach` | `methodology` |
| `result`, `finding` | `results` |
| `conclusion`, `future` | `conclusion` |
| anything else | `body` |

> **Fallback:** If fewer than 2 sections are found, it falls back to `_split_by_paragraphs()`.

---

### 2.4 — `_split_by_paragraphs()` (News Articles)

Splits on double-newline `\n\n` boundaries and merges short paragraphs until each chunk is at least **30 words**. All sections are tagged as `section_type = "body"`.

> **Single-newline handling:** If `\n\n` is not found but `\n` is, all single newlines are converted to double newlines first.

---

### 2.5 — `_split_by_topic_shift()` (Long Texts — the smart splitter)

This is the most sophisticated splitter. It uses **TF-IDF + Cosine Similarity** to detect when the topic of the text changes:

```
1. Split text into sliding windows of 50 words each
2. Vectorize all windows using TF-IDF
3. Compute cosine similarity between consecutive windows
4. If similarity < 0.30  →  topic has shifted → start new section
5. If a section grows > 800 words → force-split regardless of similarity
```

```python
if sim < 0.30 or current_word_count + next_words > MAX_SECTION_WORDS:
    # Start a new section
```

> **Last resort:** If TF-IDF still produces only 1 section for a large text (>500 words), it calls `_force_split_by_size()` which splits at sentence boundaries every ~500 words.

---

### 2.6 — `_split_by_rooms()` (Blueprint Text)

Specialized for architectural blueprints. Tokenizes sentences and groups them by room keyword (`bedroom`, `bathroom`, `kitchen`, etc.). Unmatched sentences go to a "general" introduction section.

---

### Section Output Format

Every splitter returns a list of dicts in this format:

```python
{
    "index": 0,             # Position in the document
    "raw_text": "...",      # The actual text content of this section
    "word_count": 312,      # Word count of raw_text
    "section_type": "body"  # One of: introduction, methodology, results, conclusion, body, or room name
}
```

---

## 3. Step 2 — Subheading Generator (`subheading_generator.py`)

**Goal:** Generate a short, clean, title-cased heading for each section.

### 3.1 — `generate_all_subheadings()`

The main entry point. Loops over all sections:

```python
def generate_all_subheadings(sections: list[dict], model_name: str = "t5-base") -> list[str]:
```

For **known section types**, it uses a fixed preset heading:

| `section_type` | Heading assigned |
|---|---|
| `introduction` | `"Overview"` |
| `methodology` | `"Approach & Methodology"` |
| `results` | `"Key Results & Findings"` |
| `conclusion` | `"Conclusion & Takeaways"` |
| blueprint room names | formatted room name (e.g., `"Bedroom"`) |
| `body` | calls `generate_subheading()` dynamically |

---

### 3.2 — `generate_subheading()` (Dynamic Generation for Body Sections)

```python
def generate_subheading(section_text: str, section_type: str, model_name: str = "t5-base") -> str:
```

It tries two strategies in order:

#### Strategy 1 — YAKE Keyword Extraction (Primary)

Uses the **YAKE** (Yet Another Keyword Extractor) library to pull the most statistically significant phrases:

```python
_kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=5, features=None)
kws = _kw_extractor.extract_keywords(section_text[:1000])
```

- `n=2` → extracts bigrams (2-word phrases)
- `dedupLim=0.9` → deduplicates similar keywords
- Uses only the **first 1000 characters** of the section

Extracted keywords are then **filtered for noise**:

```python
_NOISE = {"open", "text", "paper", "study", "work", "data", "result", "method",
          "approach", "model", "system", "image", "figure", "table", "page",
          "section", "chapter", "report", "journal", "article", "abstract",
          "introduction", "conclusion", "review"}
```

Also rejects:
- Keywords shorter than 4 characters
- Single ALL-CAPS tokens (e.g., `ACM`, `PDF`, `OPEN`)

Up to 3 filtered keywords are joined with `, ` and passed to `format_subheading()`.

#### Strategy 2 — AI Model Summarization (Fallback)

Only triggered if:
- YAKE returned no usable keywords **AND**
- The section has ≥ 100 words

```python
heading = summarize(prompt, model_key=model_name, max_length=20, min_length=4)
```

For T5 models, the text is prefixed with `"summarize: "`. The generated heading is then trimmed with `_trim_dangling()`.

#### Final Fallback

If neither YAKE nor the AI model produces a result:
```python
return "Topic Overview"
```

---

### 3.3 — `format_subheading()`

Cleans and formats any heading string:

```python
def format_subheading(heading: str) -> str:
```

Steps applied in order:
1. Strip leading/trailing whitespace
2. Remove leading numbering (`1. `, `2. `)
3. Remove citation noise (`[12]`, `[3]`)
4. Apply **Title Case** (`str.title()`)
5. Strip trailing punctuation (`. , ; : ! ?`)
6. Remove **dangling connectors** via `_trim_dangling()`
7. **Cap at 7 words** — trim and re-run dangling removal
8. Return `"Topic Overview"` if result is empty

---

### 3.4 — `_trim_dangling()`

Removes trailing "weak" words that make a heading end awkwardly:

```python
_DANGLING_TAIL = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
                  "of", "by", "from", "with", "is", "are", "was", "were",
                  "that", "which", "who", "while", "when", "if", "than", ...}
```

It pops words from the end until the last word is not in this set.

**Example:**
- `"Effects of Climate Change And"` → `"Effects Of Climate Change"`

---

## 4. Step 3 — Long-Form Builder (`longform_builder.py`)

**Goal:** Take the sections and their subheadings and produce a full structured summary with one summarized paragraph per section.

### 4.1 — `build_longform_summary()`

```python
def build_longform_summary(
    sections: list[dict],
    subheadings: list[str],
    model_name: str = "bart-cnn",
    length_mode: str = "detailed",
) -> dict:
```

#### Length Mode Config

Controls `max_length` and `min_length` passed to the summarization model (these are **token counts**, not word counts):

| `length_mode` | `max_length` (tokens) | `min_length` (tokens) |
|---|---|---|
| `"detailed"` | 450 | 200 |
| `"standard"` | 300 | 120 |
| `"brief"` | 150 | 50 |

---

### 4.2 — Oversized Section Splitting

Before summarizing, each section is checked for size via `_split_oversized_section()`:

```python
def _split_oversized_section(section: dict, model_name: str) -> list[dict]:
```

It calls `_token_safe_chunk_size(model_name)` from `pipeline.py` to get the safe word limit for that model:

```python
# Conservative: ~1.3 tokens per word, leave 10% margin
max_safe_words = max(100, int(cfg["max_input"] / 1.5))
```

For BART/PEGASUS with `max_input=1024` tokens, this is roughly **~680 words per chunk**.

If a section exceeds this, it is split at **sentence boundaries** into sub-chunks. This prevents the model from being fed text it cannot handle.

---

### 4.3 — Summarizing Each Section

For each (section, subheading) pair:

**Case 1: Single chunk**
- If `word_count < 50`: use the raw text directly (too short to summarize meaningfully)
- Otherwise: call `_summarize_section()` which applies further safety bounds

**Case 2: Multiple chunks** (section was oversized)
- Each chunk is summarized independently
- All chunk summaries are **concatenated** with a space
- The combined text is used as the section's `summary_paragraph`

---

### 4.4 — `_summarize_section()` — The Safety Layer

```python
def _summarize_section(text: str, model_name: str, max_length: int, min_length: int) -> str:
```

This function enforces hard limits to prevent the model from hallucinating or crashing:

```python
# Hard cap input at 700 words
if len(words) > 700:
    text = " ".join(words[:700])

# Scale max output relative to input size
input_tokens = int(len(words) * 1.3)
safe_max = min(max_length, max(50, int(input_tokens * 0.40)))   # at most 40% of input

# Scale min output very conservatively to prevent hallucination
safe_min = min(min_length, max(10, int(input_tokens * 0.10)))   # at most 10% of input
safe_min = min(safe_min, 60)                                     # absolute cap at 60

# safe_min must be strictly less than safe_max
if safe_min >= safe_max:
    safe_min = max(10, safe_max - 5)
```

It then calls the core `summarize()` pipeline with `length_penalty=1.0`.

---

### 4.5 — Final Output Structure

```python
return {
    "sections": [
        {
            "subheading": "Climate Change Effects",
            "summary_paragraph": "Rising global temperatures have led to...",
            "word_count": 87,
            "section_type": "body"
        },
        # ... one entry per section
    ],
    "total_word_count": 412,
    "model_used": "bart-cnn",
    "length_mode": "detailed"
}
```

The reading time is also computable from this output:
```python
def estimate_reading_time(word_count: int) -> str:
    minutes = math.ceil(word_count / 200)  # 200 words per minute average
    return f"{minutes} min read"
```

---

## 5. The Core Summarization Engine (`pipeline.py`)

Both the subheading generator and the long-form builder ultimately call `summarize()`:

```python
def summarize(text: str, model_key: str, **kwargs) -> str:
```

### Inference Flow:

```
1. Load tokenizer + model (cached after first load)
2. prepare_input(text, model_key)  ← adds T5 "summarize:" prefix if needed
3. Tokenize with truncation at cfg["max_input"] tokens
4. Dynamically scale min_length based on actual input word count:
      > 300 words → min_len = min(60, ...)
      > 100 words → min_len = min(50, ...)
        ≤ 100 words → min_len = min(20, ...)
5. Run model.generate() with:
      - num_beams = 4 (beam search)
      - no_repeat_ngram_size = 3 (no trigram repetition)
      - early_stopping = True
6. Decode output tokens → raw string
7. postprocess_summary(raw) → cleaned final text
```

---

## 6. Full Data Flow Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                         RAW INPUT TEXT                             │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
              ┌─────────────────────────────────┐
              │     auto_detect_source_type()   │
              │  research_paper / news_article  │
              │  blueprint / long_text          │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │        detect_sections()        │
              │                                 │
              │  ┌──────────────────────────┐   │
              │  │ _split_by_headings()     │   │  ← research_paper
              │  │ _split_by_paragraphs()   │   │  ← news_article
              │  │ _split_by_topic_shift()  │   │  ← long_text (TF-IDF)
              │  │ _split_by_rooms()        │   │  ← blueprint
              │  └──────────────────────────┘   │
              └────────────────┬────────────────┘
                               │
                   List of section dicts
              {index, raw_text, word_count, section_type}
                               │
              ┌────────────────▼────────────────┐
              │    generate_all_subheadings()   │
              │                                 │
              │  Known type? → fixed preset     │
              │  Body type?  → generate_subheading()
              │                │                │
              │         ┌──────▼──────┐         │
              │         │ YAKE        │  ← primary
              │         │ (keywords)  │         │
              │         └──────┬──────┘         │
              │                │ no keywords?   │
              │         ┌──────▼──────┐         │
              │         │ AI Model    │  ← fallback (≥100 words)
              │         │ (t5-base)   │         │
              │         └──────┬──────┘         │
              │                │                │
              │         format_subheading()     │
              └────────────────┬────────────────┘
                               │
                   List of heading strings
                               │
              ┌────────────────▼────────────────┐
              │       build_longform_summary()  │
              │                                 │
              │  For each section + heading:    │
              │  1. _split_oversized_section()  │
              │  2. _summarize_section()        │
              │     └── summarize() pipeline    │
              │         └── model.generate()   │
              │  3. Combine chunk summaries     │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │         FINAL OUTPUT            │
              │  {                              │
              │    sections: [                  │
              │      { subheading,              │
              │        summary_paragraph,       │
              │        word_count,              │
              │        section_type }           │
              │    ],                           │
              │    total_word_count,            │
              │    model_used,                  │
              │    length_mode                  │
              │  }                              │
              └─────────────────────────────────┘
```

---

## 7. Key Design Decisions

| Decision | Reason |
|---|---|
| **Sentence-boundary splitting** | Avoids cutting sentences mid-way, preserving grammar and coherence |
| **YAKE first, AI model second** for subheadings | YAKE is fast, deterministic, and doesn't require GPU. AI fallback is only used when YAKE fails |
| **Hard 700-word input cap** per summarization call | BART/PEGASUS have a 1024-token limit (~700 words). Exceeding it causes silent truncation or errors |
| **`safe_min = min(safe_min, 60)`** | Prevents forcing the model to produce over-long outputs for short sections, which causes hallucination |
| **`length_penalty = 1.0`** in `_summarize_section` | Neutral penalty — lets the model choose a natural length rather than forcing longer outputs |
| **`no_repeat_ngram_size = 3`** | Prevents verbatim phrase repetition in generated summaries |
| **Cosine similarity threshold < 0.30** | Empirically chosen cutoff for topic-shift detection using TF-IDF vectors |
| **`estimate_reading_time` at 200 WPM** | Standard accepted average adult reading speed |

---

*Files covered: `summarizer/section_detector.py`, `summarizer/subheading_generator.py`, `summarizer/longform_builder.py`, `summarizer/pipeline.py`*
