import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def auto_detect_source_type(text: str, metadata: dict = None) -> str:
    if metadata is None: metadata = {}
    if "type" in metadata:
        return metadata["type"]
    
    if "Abstract" in text and "References" in text:
        return "research_paper"
    if "Reuters" in text or "AP" in text:
        return "news_article"
    
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    avg_len = sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0
    
    if avg_len < 80:
        return "news_article"
    if avg_len > 200:
        return "long_text"
        
    return "long_text"

def detect_sections(text: str, source_type: str = "auto") -> list[dict]:
    if source_type == "auto":
        source_type = auto_detect_source_type(text)
        
    if source_type == "research_paper":
        return _split_by_headings(text)
    elif source_type == "news_article":
        return _split_by_paragraphs(text)
    elif source_type == "blueprint":
        return _split_by_rooms(text)
    else:
        return _split_by_topic_shift(text)

def _split_by_headings(text: str) -> list[dict]:
    lines = text.split("\n")
    sections = []
    current_text = []
    current_heading = "Introduction"
    
    for line in lines:
        if re.match(r"^#{1,3}\s.+", line) or re.match(r"^\d+\.\s[A-Z].+", line) or re.match(r"^[A-Z][A-Z\s]{4,}$", line.strip()):
            if current_text:
                sections.append((current_heading, "\n".join(current_text).strip()))
            current_heading = line.replace("#", "").strip()
            current_heading = re.sub(r"^\d+\.\s", "", current_heading)
            current_text = []
        else:
            current_text.append(line)
            
    if current_text:
        sections.append((current_heading, "\n".join(current_text).strip()))
        
    if not sections or len(sections) == 1:
        return _split_by_paragraphs(text)
        
    results = []
    for i, (heading, content) in enumerate(sections):
        h_lower = heading.lower()
        if "abstract" in h_lower or "introduction" in h_lower:
            sec_type = "introduction"
        elif "method" in h_lower or "approach" in h_lower:
            sec_type = "methodology"
        elif "result" in h_lower or "finding" in h_lower:
            sec_type = "results"
        elif "conclusion" in h_lower or "future" in h_lower:
            sec_type = "conclusion"
        else:
            sec_type = "body"
            
        results.append({
            "index": i,
            "raw_text": heading + "\n\n" + content,
            "word_count": len(content.split()),
            "section_type": sec_type
        })
    return results

def _split_by_paragraphs(text: str, min_words: int = 30) -> list[dict]:
    # Support for plain text copy-pastes where paragraphs are just separated by \n
    if "\n\n" not in text and "\n" in text:
        text = text.replace("\n", "\n\n")
        
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    merged = []
    current = []
    current_words = 0
    
    for p in raw_paragraphs:
        words = len(p.split())
        current.append(p)
        current_words += words
        if current_words >= min_words:
            merged.append("\n\n".join(current))
            current = []
            current_words = 0
            
    if current and merged:
        merged[-1] += "\n\n" + "\n\n".join(current)
    elif current:
        merged.append("\n\n".join(current))
        
    results = []
    for i, content in enumerate(merged):
        sec_type = "body"
            
        results.append({
            "index": i,
            "raw_text": content,
            "word_count": len(content.split()),
            "section_type": sec_type
        })
    return results

def _split_by_topic_shift(text: str, chunk_size: int = 50) -> list[dict]:
    words = text.split()
    if len(words) < chunk_size * 1.5:
        return _split_by_paragraphs(text, min_words=25)
        
    windows = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    vectorizer = TfidfVectorizer(stop_words='english')
    
    try:
        tfidf_matrix = vectorizer.fit_transform(windows)
        cosine_sim = cosine_similarity(tfidf_matrix)
    except Exception:
        return _split_by_paragraphs(text, min_words=25)
        
    merged_sections = []
    current_section = [windows[0]]
    current_word_count = len(windows[0].split())
    
    MAX_SECTION_WORDS = 500  # Fix 3.2: was 800 — shorter sections summarize cleaner
    
    for i in range(len(windows) - 1):
        sim = cosine_sim[i][i+1]
        next_words = len(windows[i+1].split())
        
        # Split if topic shifts OR section is getting too large
        # Fix 3.1: raised threshold 0.30 → 0.45 for fewer, more coherent sections
        if sim < 0.45 or current_word_count + next_words > MAX_SECTION_WORDS:
            merged_sections.append(" ".join(current_section))
            current_section = [windows[i+1]]
            current_word_count = next_words
        else:
            current_section.append(windows[i+1])
            current_word_count += next_words
            
    if current_section:
        merged_sections.append(" ".join(current_section))

    # If we still only got 1 section from a large document, force-split by size
    if len(merged_sections) <= 1 and len(words) > 500:
        return _force_split_by_size(text, target_words=500)
        
    results = []
    for i, content in enumerate(merged_sections):
        sec_type = "body"
        
        results.append({
            "index": i,
            "raw_text": content,
            "word_count": len(content.split()),
            "section_type": sec_type
        })
    return results


def _force_split_by_size(text: str, target_words: int = 500) -> list[dict]:
    """Last-resort splitter that breaks text into chunks at sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sections = []
    current_sents = []
    current_wc = 0
    
    for sent in sentences:
        sw = len(sent.split())
        if current_wc + sw > target_words and current_sents:
            content = " ".join(current_sents)
            sections.append({
                "index": len(sections),
                "raw_text": content,
                "word_count": current_wc,
                "section_type": "body"
            })
            current_sents = []
            current_wc = 0
        current_sents.append(sent)
        current_wc += sw
    
    if current_sents:
        content = " ".join(current_sents)
        sections.append({
            "index": len(sections),
            "raw_text": content,
            "word_count": current_wc,
            "section_type": "body"
        })
    
    return sections

def _split_by_rooms(blueprint_plain_english: str) -> list[dict]:
    rooms = ["bedroom", "bathroom", "kitchen", "living", "dining", "garage", "basement", "floor"]
    try:
        sentences = nltk.sent_tokenize(blueprint_plain_english)
    except Exception:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(blueprint_plain_english)
        sentences = [s.text for s in doc.sents]
        
    room_groups = {r: [] for r in rooms}
    room_groups["general"] = []
    
    for s in sentences:
        s_lower = s.lower()
        assigned = False
        for r in rooms:
            if r in s_lower:
                room_groups[r].append(s)
                assigned = True
                break
        if not assigned:
            room_groups["general"].append(s)
            
    results = []
    idx = 0
    if room_groups["general"]:
        content = " ".join(room_groups["general"])
        results.append({
            "index": idx,
            "raw_text": content,
            "word_count": len(content.split()),
            "section_type": "introduction"
        })
        idx += 1
        
    for r in rooms:
        if room_groups[r]:
            content = " ".join(room_groups[r])
            results.append({
                "index": idx,
                "raw_text": content,
                "word_count": len(content.split()),
                "section_type": r
            })
            idx += 1
            
    return results


# Fix 3.3 — Merge pass: combine sections under min_words with their neighbour
def merge_short_sections(sections: list, min_words: int = 80, max_words: int = 500) -> list:
    """
    After detect_sections(), merge sections that are too short (<min_words)
    into their preceding section. Prevents the summarizer from receiving
    near-empty chunks that produce vague or repetitive output.

    A short section is only absorbed while the result stays within
    max_words (the topic-shift splitter's own section cap). Without that cap,
    a document split into many ~50-word windows snowballed into one section
    holding the whole text, so the "structured" summary had one paragraph.
    Call this before generate_all_subheadings().
    """
    merged = []
    buffer = None

    for section in sections:
        if buffer is None:
            buffer = dict(section)
        elif buffer["word_count"] < min_words or (
            section["word_count"] < min_words
            and buffer["word_count"] + section["word_count"] <= max_words
        ):
            # The buffer is too short to stand alone, or the incoming section is
            # too short and still fits. Absorb this section into the buffer.
            buffer["raw_text"] += "\n\n" + section["raw_text"]
            buffer["word_count"] += section["word_count"]
        else:
            merged.append(buffer)
            buffer = dict(section)

    if buffer:
        if buffer["word_count"] < min_words and merged:
            # Don't leave a near-empty section at the end
            merged[-1]["raw_text"] += "\n\n" + buffer["raw_text"]
            merged[-1]["word_count"] += buffer["word_count"]
        else:
            merged.append(buffer)

    # Re-index
    for i, s in enumerate(merged):
        s["index"] = i

    return merged
