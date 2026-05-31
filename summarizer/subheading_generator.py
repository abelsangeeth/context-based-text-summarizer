import re
import yake

_kw_extractor = None
_sim_model = None

def get_similarity_model():
    global _sim_model
    if _sim_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Lightweight and fast CPU model
            _sim_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            _sim_model = "fallback"
    return _sim_model

def get_best_heading(candidates: list, text: str) -> str:
    """Use sentence-transformers to score candidates against the text."""
    # Filter out empty or generic candidates
    candidates = [c for c in candidates if c and c.lower() not in {"topic overview", "general discussion"}]
    if not candidates:
        return "Topic Overview"
    if len(candidates) == 1:
        return candidates[0]
        
    model = get_similarity_model()
    if model == "fallback":
        return candidates[0]
        
    # To avoid scoring against noisy headers at the start of a section,
    # we extract the most central sentences first.
    try:
        from summarizer.longform_builder import extractive_prefilter
        clean_text = extractive_prefilter(text, keep_ratio=0.4) if len(text.split()) > 15 else text
    except Exception:
        clean_text = text

    # Compare against the first ~150 words of the cleaned, dense text
    text_chunk = " ".join(clean_text.split()[:150])
    
    cand_emb = model.encode(candidates)
    text_emb = model.encode([text_chunk])[0]
    
    from sentence_transformers.util import cos_sim
    scores = cos_sim(text_emb, cand_emb)[0]
    
    best_idx = scores.argmax().item()
    return candidates[best_idx]

# Words that should never end a heading (articles, prepositions, conjunctions)
_DANGLING_TAIL = {
    "a", "an", "the", "and", "or", "but", "nor", "for", "yet", "so",
    "in", "on", "at", "to", "of", "by", "from", "with", "as", "into",
    "is", "are", "was", "were", "be", "been", "being",
    "that", "which", "who", "whom", "whose", "while", "when",
    "if", "than", "its", "their", "this", "these", "those",
    "has", "have", "had", "does", "do", "did", "can", "could",
    "will", "would", "shall", "should", "may", "might",
    "not", "also", "about", "over", "under", "between",
    "include", "including", "based", "use", "using", "used", "show", "shows"
}

# Fix 1.3 — Expanded noise filter (added filler/transition words)
_NOISE = {
    # Original set
    "open", "text", "paper", "study", "work", "data", "result",
    "method", "approach", "model", "system", "image", "figure",
    "table", "page", "section", "chapter", "report", "journal",
    "article", "abstract", "introduction", "conclusion", "review",
    # Added: Affiliation / Metadata noise
    "department", "institute", "university", "school", "faculty", 
    "student", "students", "author", "authors", "college", "laboratory",
    "science", "engineering", "technology", "research",
    # Added: filler words common in academic/news text
    "however", "therefore", "furthermore", "moreover", "although",
    "according", "based", "include", "including", "provide", "provides",
    "suggest", "suggests", "show", "shows", "found", "find", "use",
    "used", "using", "also", "well", "many", "several", "various",
    "different", "important", "significant", "current", "new", "high",
    "large", "small", "first", "second", "third", "one", "two", "three",
    "number", "type", "level", "part", "case", "time", "year", "years",
}


def _trim_dangling(heading: str) -> str:
    """Remove trailing incomplete words like 'while', 'from', 'and'."""
    words = heading.split()
    while words and words[-1].lower().rstrip(".,;:!?") in _DANGLING_TAIL:
        words.pop()
    return " ".join(words) if words else heading


# Extractive Summarization: Rank sentences based on term frequency (TF-IDF)
# or centrality and select the highest-scoring sentence as the heading.
def tf_idf_centrality_heading(section_text: str) -> str:
    """
    Extract the most central sentence using TF-IDF and truncate it to heading length.
    """
    try:
        import nltk
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        sentences = nltk.sent_tokenize(section_text)
        if not sentences:
            return section_text[:50]
            
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(sentences + [section_text])
        
        doc_vector = tfidf_matrix[-1]
        sentence_vectors = tfidf_matrix[:-1]
        scores = cosine_similarity(sentence_vectors, doc_vector).flatten()
        
        # Select the highest-scoring sentence
        best_idx = scores.argmax()
        best_sentence = sentences[best_idx]
    except Exception:
        # Fallback to first sentence if TF-IDF fails (e.g. empty vocabulary)
        sentences = re.split(r'(?<=[.!?])\s+', section_text.strip())
        best_sentence = sentences[0] if sentences else section_text

    # Strip leading stopwords/articles
    best_sentence = re.sub(r'^(The|A|An|This|These|In|At|On)\s+', '', best_sentence, flags=re.IGNORECASE)

    # Truncate at 8 words
    words = best_sentence.split()
    truncated = " ".join(words[:8])
    if len(words) > 8:
        truncated += "..."

    return truncated.title()


# Fix 1.1 — Replace comma-join with noun phrase joining
def keywords_to_heading(keywords: list) -> str:
    """Turn a list of keywords into a readable noun-phrase heading."""
    if not keywords:
        return "General Discussion"
    
    main = keywords[0].title()
    if len(keywords) == 1:
        return main
        
    main_words = set(main.lower().split())
    
    # Find a supporting keyword that has absolutely NO overlapping words with main
    supporting = None
    for kw in keywords[1:]:
        kw_lower = kw.lower()
        if kw_lower in main.lower() or main.lower() in kw_lower:
            continue
            
        kw_words = set(kw_lower.split())
        overlap = main_words.intersection(kw_words)
        
        if len(overlap) == 0:
            supporting = kw.title()
            break
            
    if not supporting:
        return main
        
    return f"{main} and {supporting}"


def generate_subheading(section_text: str, section_type: str, model_name: str = "t5-base") -> str:
    """Generate a concise, well-formed heading for a section using similarity scoring."""
    words = section_text.split()
    candidates = []

    # ── 1. Keyword candidates ──────────
    global _kw_extractor
    if not _kw_extractor:
        _kw_extractor = yake.KeywordExtractor(
            lan="en", n=3, dedupLim=0.7, top=5, features=None
        )
    kws = _kw_extractor.extract_keywords(section_text[:1000])
    if kws:
        filtered = [
            k[0] for k in kws
            if len(k[0]) >= 6
            and not any(w.lower() in _NOISE for w in k[0].split())
            and not (k[0].isupper() and len(k[0].split()) == 1)
        ]
        if filtered:
            # Option A: Joined noun phrase
            candidates.append(keywords_to_heading(filtered[:5]))
            # Option B: Individual strong keywords
            for kw in filtered[:3]:
                candidates.append(kw.title())

    # ── 2. Extractive TF-IDF fallback ─────────
    if len(words) >= 20:
        tf_heading = tf_idf_centrality_heading(section_text)
        if tf_heading and tf_heading.lower() not in {"topic overview", "general discussion"}:
            candidates.append(tf_heading)

    # ── 3. AI model candidate (Always try to generate an abstractive candidate) ────
    if len(words) >= 30:
        if model_name.startswith("t5"):
            prompt = f"summarize: {section_text[:600]}"
        else:
            prompt = section_text[:600]

        try:
            from summarizer.pipeline import summarize  # lazy import avoids circular dep
            heading = summarize(
                prompt, model_key=model_name, max_length=15, min_length=3
            )
            if heading and len(heading.split()) >= 2:
                # The AI model is significantly smarter than keyword extractors.
                # If it successfully generates a heading, return it immediately!
                return format_subheading(_trim_dangling(heading))
        except Exception as e:
            print(f"AI Heading Gen Error: {e}")

    if not candidates:
        return "Topic Overview"
        
    # Deduplicate and format candidates (Fallback for when AI fails)
    seen = set()
    unique_cands = []
    for c in candidates:
        c_fmt = format_subheading(c)
        if c_fmt.lower() not in seen:
            seen.add(c_fmt.lower())
            unique_cands.append(c_fmt)
            
    # Score fallback candidates
    best = get_best_heading(unique_cands, section_text)
    return best


def generate_all_subheadings(
    sections: list, model_name: str = "t5-base"
) -> list:
    subheadings = []
    for sec in sections:
        stype = sec.get("section_type", "body")
        if stype == "introduction":
            subheadings.append("Overview")
        elif stype == "methodology":
            subheadings.append("Approach & Methodology")
        elif stype == "results":
            subheadings.append("Key Results & Findings")
        elif stype == "conclusion":
            subheadings.append("Conclusion & Takeaways")
        elif stype != "body":
            # For blueprints (room names)
            subheadings.append(format_subheading(stype))
        else:
            subheadings.append(
                generate_subheading(sec["raw_text"], stype, model_name)
            )

    return subheadings


def format_subheading(heading: str) -> str:
    """Capitalise, trim, and sanitise a heading string."""
    # Strip markdown, leading digits, citation noise
    clean = heading.strip()
    clean = re.sub(r"^\d+\.\s*", "", clean)
    clean = re.sub(r"\[\d+\]", "", clean)

    clean = clean.title()

    # Remove trailing punctuation
    clean = clean.rstrip(".,;:!? ")

    # Remove dangling connectors
    clean = _trim_dangling(clean)

    # Cap at 7 words
    words = clean.split()
    if len(words) > 7:
        clean = " ".join(words[:7])
        clean = _trim_dangling(clean)

    return clean.strip() or "Topic Overview"
