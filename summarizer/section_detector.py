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
    
    for i in range(len(windows) - 1):
        sim = cosine_sim[i][i+1]
        if sim < 0.45:  # hyper aggressive topic shift threshold
            merged_sections.append(" ".join(current_section))
            current_section = [windows[i+1]]
        else:
            current_section.append(windows[i+1])
            
    if current_section:
        merged_sections.append(" ".join(current_section))
        
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
