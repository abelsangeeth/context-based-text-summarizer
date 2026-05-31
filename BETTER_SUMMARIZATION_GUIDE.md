# Better Summarization — A Full Improvement Guide

> A diagnosis of what's going wrong in your current pipeline, and concrete fixes for each stage —  
> better headings, better summaries, better structure.

---

## Table of Contents

1. [What's Actually Wrong](#1-whats-actually-wrong)
2. [Fix 1 — Better Headings (Subheading Generator)](#2-fix-1--better-headings-subheading-generator)
3. [Fix 2 — Better Summaries (Long-Form Builder)](#3-fix-2--better-summaries-long-form-builder)
4. [Fix 3 — Better Section Detection](#4-fix-3--better-section-detection)
5. [Fix 4 — Post-Processing Pass](#5-fix-4--post-processing-pass)
6. [Fix 5 — Switch or Upgrade the Model](#6-fix-5--switch-or-upgrade-the-model)
7. [Quick-Win Checklist](#7-quick-win-checklist)

---

## 1. What's Actually Wrong

Before fixing anything, it helps to know *why* these problems happen:

### Problem A — Bad Headings

Your current flow for `body` sections is:
```
YAKE bigrams → filter noise → join with ", " → title-case
```

The problem: YAKE extracts the most *statistically frequent* phrases, not the most *meaningful* ones. For a section about "the economic consequences of climate migration," YAKE might give you `"migration economic"` → heading becomes `"Migration Economic"` — which is clunky and unnatural.

Also, **joining keywords with commas** produces headings like `"Carbon Emissions, Policy Reform, Global Targets"` — these look like tags, not headings.

---

### Problem B — Bad Summaries

Your summarizer chains BART/PEGASUS directly on raw section text. The three root causes of bad summaries are:

| Cause | What it looks like |
|---|---|
| Input too long → truncated silently | Summary starts well, then cuts off mid-idea |
| `min_length` too low | Summaries are 1–2 vague sentences that say nothing |
| No extraction step before generation | Model has to guess what's important |

---

### Problem C — Sections are Wrong

If `_split_by_topic_shift()` produces sections that don't actually represent topic changes, every downstream step suffers. A bad section boundary → bad heading → bad summary.

---

## 2. Fix 1 — Better Headings (Subheading Generator)

This is the highest-impact fix and requires no new models.

### 2.1 — Replace comma-join with a phrase, not a list

**Current behavior:**
```python
heading = ", ".join(top_keywords[:3])
# → "Carbon Emissions, Climate Policy, Migration"
```

**Better approach — turn keywords into a noun phrase:**
```python
def keywords_to_heading(keywords: list[str]) -> str:
    if len(keywords) == 0:
        return "General Discussion"
    if len(keywords) == 1:
        return keywords[0].title()
    # Use only the top 2, connected naturally
    main = keywords[0].title()
    supporting = keywords[1].title()
    return f"{main} and {supporting}"
```

This produces: `"Carbon Emissions and Climate Policy"` — readable as an actual heading.

---

### 2.2 — Use the first sentence as a heading fallback (better than AI model)

When YAKE fails, your fallback is `t5-base` summarization with `max_length=20`. T5-base is not good at this. A much simpler and more reliable fallback:

```python
def first_sentence_heading(section_text: str) -> str:
    """
    Extract the first sentence and truncate it to heading length.
    Works better than T5-base for short sections.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', section_text.strip())
    first = sentences[0] if sentences else section_text

    # Strip leading stopwords/articles
    first = re.sub(r'^(The|A|An|This|These|In|At|On)\s+', '', first, flags=re.IGNORECASE)

    # Truncate at 8 words
    words = first.split()
    truncated = " ".join(words[:8])
    if len(words) > 8:
        truncated += "..."

    return truncated.title()
```

**Fallback priority (updated order):**
```
1. Fixed preset   → for known section_types (introduction, methodology, etc.)
2. YAKE bigrams   → with noun-phrase joining (not comma join)
3. First sentence → truncated to 8 words (replaces t5-base)
4. Section type   → generic fallback ("Section 1", "Section 2")
```

---

### 2.3 — Expand the NOISE filter

Your current noise set is missing many common filler phrases that YAKE extracts from academic and news text:

```python
_NOISE = {
    # existing
    "open", "text", "paper", "study", "work", "data", "result", "method",
    "approach", "model", "system", "image", "figure", "table", "page",
    "section", "chapter", "report", "journal", "article", "abstract",
    "introduction", "conclusion", "review",

    # ADD THESE
    "however", "therefore", "furthermore", "moreover", "although",
    "according", "based", "include", "including", "provide", "provides",
    "suggest", "suggests", "show", "shows", "found", "find", "use",
    "used", "using", "also", "well", "many", "several", "various",
    "different", "important", "significant", "current", "new", "high",
    "large", "small", "first", "second", "third", "one", "two", "three",
    "number", "type", "level", "part", "case", "time", "year", "years"
}
```

Also add a **minimum character length of 6** (not 4) — this kills short junk like `"use"`, `"type"`, `"high"`.

---

### 2.4 — Use `n=3` (trigrams) instead of `n=2` in YAKE

Bigrams (`n=2`) give you `"climate change"`. Trigrams (`n=3`) give you `"climate change effects"` — which makes a better heading.

```python
_kw_extractor = yake.KeywordExtractor(
    lan="en",
    n=3,          # changed from 2 → 3
    dedupLim=0.7, # tighter dedup
    top=5,
    features=None
)
```

---

## 3. Fix 2 — Better Summaries (Long-Form Builder)

### 3.1 — Add an extraction step before generation

The single biggest improvement you can make to summary quality: **before sending text to BART/PEGASUS, extract the most important sentences first** using a fast extractive method. This acts as a pre-filter.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def extractive_prefilter(text: str, keep_ratio: float = 0.5) -> str:
    """
    Keep only the top sentences by TF-IDF centrality.
    Reduces input noise before abstractive summarization.
    """
    import nltk
    sentences = nltk.sent_tokenize(text)

    if len(sentences) <= 3:
        return text  # Too short to filter

    # Score each sentence by its similarity to the whole document
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences + [text])
    except ValueError:
        return text  # Fallback on empty vocabulary

    doc_vector = tfidf_matrix[-1]
    sentence_vectors = tfidf_matrix[:-1]
    scores = cosine_similarity(sentence_vectors, doc_vector).flatten()

    # Keep top sentences, preserving original order
    n_keep = max(3, int(len(sentences) * keep_ratio))
    top_indices = sorted(np.argsort(scores)[-n_keep:])
    return " ".join(sentences[i] for i in top_indices)
```

**Integration point** — call this inside `_summarize_section()` before tokenizing:

```python
def _summarize_section(text: str, model_name: str, max_length: int, min_length: int) -> str:
    words = text.split()

    # NEW: pre-filter for long sections
    if len(words) > 200:
        text = extractive_prefilter(text, keep_ratio=0.5)
        words = text.split()

    # ... rest of existing logic
```

---

### 3.2 — Raise `min_length` per mode

Your current `min_length` values are too low, which is why summaries feel thin:

| Mode | Current `min_length` | Recommended |
|---|---|---|
| `"detailed"` | 200 | 250 |
| `"standard"` | 120 | 150 |
| `"brief"` | 50 | 60 |

Also, the absolute cap `safe_min = min(safe_min, 60)` is too aggressive for `detailed` mode. Change it per mode:

```python
# In _summarize_section, pass length_mode through
mode_min_caps = {"detailed": 120, "standard": 80, "brief": 40}
safe_min = min(safe_min, mode_min_caps.get(length_mode, 80))
```

---

### 3.3 — Fix chunk-concatenation (the worst summary killer)

**Current behavior for oversized sections:**
```python
# Each chunk summarized independently, then just concatenated:
summary = " ".join(chunk_summaries)
```

This produces summaries that repeat context, lose thread, or contradict each other across chunks. Fix: **summarize chunk summaries together in a final pass**.

```python
def _summarize_chunked_section(
    chunks: list[dict],
    model_name: str,
    max_length: int,
    min_length: int
) -> str:
    # Step 1: summarize each chunk at brief level
    chunk_summaries = []
    brief_max = max(50, max_length // len(chunks))
    brief_min = max(10, min_length // len(chunks))

    for chunk in chunks:
        s = _summarize_section(chunk["raw_text"], model_name, brief_max, brief_min)
        chunk_summaries.append(s)

    # Step 2: concatenate chunk summaries (now much shorter total)
    combined = " ".join(chunk_summaries)

    # Step 3: one final summarization pass on the combined chunk summaries
    if len(combined.split()) > 100:
        return _summarize_section(combined, model_name, max_length, min_length)
    return combined
```

This is called a **hierarchical summarization** approach — summarize parts, then summarize the parts. It is the standard fix for long-document summarization.

---

### 3.4 — Increase `length_penalty` for fluency

Your current `length_penalty=1.0` is neutral. For more complete, coherent summaries, use `length_penalty=1.5`:

```python
output = model.generate(
    input_ids,
    num_beams=4,
    no_repeat_ngram_size=3,
    early_stopping=True,
    length_penalty=1.5,    # changed from 1.0 → encourages longer outputs
    max_length=safe_max,
    min_length=safe_min,
)
```

Use `length_penalty=2.0` for `detailed` mode and `0.8` for `brief` mode to give each mode a natural character.

---

## 4. Fix 3 — Better Section Detection

### 4.1 — Raise the TF-IDF threshold for topic shift

The current threshold is `0.30`. This is too low — consecutive windows with 30% similarity can still be about the same topic. Raise it:

```python
TOPIC_SHIFT_THRESHOLD = 0.45  # was 0.30
```

This produces fewer, more coherent sections, which gives the summarizer cleaner input.

---

### 4.2 — Reduce MAX_SECTION_WORDS from 800 to 500

Summarizers perform best on 150–400 word sections. Forcing 800-word sections into BART causes the most quality degradation:

```python
MAX_SECTION_WORDS = 500  # was 800
```

This will create more sections, but each will summarize cleaner.

---

### 4.3 — Merge tiny sections before summarizing

After detection, add a merge pass that combines sections under 80 words with their neighbor:

```python
def merge_short_sections(sections: list[dict], min_words: int = 80) -> list[dict]:
    merged = []
    buffer = None

    for section in sections:
        if buffer is None:
            buffer = dict(section)
        elif buffer["word_count"] < min_words:
            # Merge into buffer
            buffer["raw_text"] += " " + section["raw_text"]
            buffer["word_count"] += section["word_count"]
        else:
            merged.append(buffer)
            buffer = dict(section)

    if buffer:
        merged.append(buffer)

    # Re-index
    for i, s in enumerate(merged):
        s["index"] = i

    return merged
```

Call this after `detect_sections()` and before `generate_all_subheadings()`.

---

## 5. Fix 4 — Post-Processing Pass

After building the final output, run a cleanup pass on each `summary_paragraph`:

```python
import re

def postprocess_summary(text: str) -> str:
    # 1. Remove repeated words/phrases (common BART artifact)
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)

    # 2. Capitalize first letter
    text = text[0].upper() + text[1:] if text else text

    # 3. Ensure ends with a period
    if text and text[-1] not in '.!?':
        text += '.'

    # 4. Remove trailing "..." artifacts from truncation
    text = re.sub(r'\.{3,}$', '.', text)

    # 5. Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text).strip()

    return text
```

Apply this to every `summary_paragraph` before building the final output dict.

---

## 6. Fix 5 — Switch or Upgrade the Model

If you've applied all the above and summaries are still poor, the bottleneck is the model. Here are your options ranked by quality improvement:

| Model | Quality | Speed | VRAM | Notes |
|---|---|---|---|---|
| `facebook/bart-large-cnn` | ★★★★☆ | Slow | ~3 GB | Your best in-house upgrade. Significant quality jump over `bart-base` |
| `google/pegasus-xsum` | ★★★★☆ | Medium | ~2.5 GB | Very concise summaries, good for `brief` mode |
| `allenai/led-base-16384` | ★★★★☆ | Medium | ~2 GB | Handles long documents natively, no chunking needed |
| `claude-sonnet-4-20250514` (API) | ★★★★★ | Fast | None | Best quality by far. Use via API if cost is acceptable |

**Recommended migration path:**

If you're on `bart-base` → switch to `bart-large-cnn`. This single change will visibly improve summary coherence and coverage without changing any other code.

If you're on `bart-large-cnn` and still getting poor results → the issue is the pipeline (sections, chunking), not the model. Go back and apply Fixes 1–4.

---

## 7. Quick-Win Checklist

Apply these in order — each is independently useful:

- [ ] **Fix heading join** — replace `", ".join(keywords)` with `"X and Y"` noun phrase format
- [ ] **Expand noise filter** — add the filler words listed in Fix 1.3
- [ ] **Switch YAKE to trigrams** — change `n=2` to `n=3`
- [ ] **Replace T5-base fallback** with first-sentence extractor (Fix 1.2)
- [ ] **Add extractive pre-filter** before BART/PEGASUS for sections > 200 words (Fix 2.1)
- [ ] **Raise `min_length`** per mode (Fix 2.2)
- [ ] **Replace chunk concatenation** with hierarchical summarization (Fix 2.3)
- [ ] **Raise TF-IDF threshold** from `0.30` to `0.45` (Fix 3.1)
- [ ] **Lower `MAX_SECTION_WORDS`** from `800` to `500` (Fix 3.2)
- [ ] **Add merge pass** for short sections under 80 words (Fix 3.3)
- [ ] **Run post-processing** on all summary paragraphs (Fix 4)
- [ ] *(Optional)* Upgrade to `bart-large-cnn` if still unsatisfied (Fix 5)

---

*This guide targets: `summarizer/section_detector.py`, `summarizer/subheading_generator.py`, `summarizer/longform_builder.py`, `summarizer/pipeline.py`*
