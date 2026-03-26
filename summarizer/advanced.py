"""Advanced summarization: Hybrid, NER, Keywords, Highlights."""

import re
import nltk
from collections import defaultdict
import numpy as np

# Load YAKE lazily
_kw_extractor = None

def get_keywords(text: str, top_n: int = 5):
    """Extract keywords using YAKE."""
    global _kw_extractor
    if not _kw_extractor:
        import yake
        _kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=10, features=None)
    
    kws = _kw_extractor.extract_keywords(text)
    # Return just the terms, sorted by score (lower is better in YAKE)
    return [kw[0] for kw in kws[:top_n]]

# Load SpaCy lazily
_nlp = None

def _get_spacy():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            import sys
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            _nlp = spacy.load("en_core_web_sm")
    return _nlp

def check_entities(source: str, summary: str):
    """Verify person, location, org, date from source appear in summary."""
    nlp = _get_spacy()
    
    doc_src = nlp(source)
    doc_sum = nlp(summary)

    valid_labels = {'PERSON', 'ORG', 'GPE', 'LOC', 'DATE'}
    
    src_ents = {}
    for ent in doc_src.ents:
        if ent.label_ in valid_labels:
            term = ent.text.strip()
            # Basic normalization (remove punctuation at ends, make lower)
            if term:
                src_ents[term.lower()] = (ent.text, ent.label_)

    sum_ents = set()
    for ent in doc_sum.ents:
        if ent.label_ in valid_labels:
            sum_ents.add(ent.text.strip().lower())

    preserved = []
    missing = []
    
    for norm, (orig, label) in src_ents.items():
        if norm in sum_ents:
            preserved.append((orig, label))
        else:
            # We can do a string search fallback in case NER missed it in summary
            if orig.lower() in summary.lower():
                preserved.append((orig, label))
            else:
                missing.append((orig, label))
                
    # Deduplicate
    return list(set(preserved)), list(set(missing))


def highlight_sentences(source: str, summary: str):
    """Return an array of sentences from source with a 'score' showing how much it contributed to the summary.
    We compute token overlap between source sentence and the entire summary."""
    try:
        sentences = nltk.sent_tokenize(source)
    except LookupError:
        nltk.download('punkt')
        sentences = nltk.sent_tokenize(source)
        
    summary_tokens = set(re.findall(r'\w+', summary.lower()))
    
    scored_sentences = []
    max_score = 0.001
    
    for s in sentences:
        s_tokens = set(re.findall(r'\w+', s.lower()))
        if not s_tokens:
            scored_sentences.append((s, 0.0))
            continue
        # Use simple Jaccard-like or recall score
        overlap = len(s_tokens & summary_tokens)
        score = overlap / (len(s_tokens) + 1.0)
        scored_sentences.append((s, score))
        if score > max_score:
            max_score = score
            
    # Normalize scores to 0-1
    return [(s, float(score / max_score)) for s, score in scored_sentences]


def extractive_summary(text: str, top_n: int = 5) -> str:
    """Fast TextRank based extractive summarizer."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import networkx as nx

    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        sentences = nltk.sent_tokenize(text)
        
    if len(sentences) <= top_n:
        return text

    vectorizer = TfidfVectorizer(stop_words='english')
    # Filter out empty or too short sentences
    valid_sentences = [s for s in sentences if len(s.split()) > 3]
    if len(valid_sentences) <= top_n:
        return text
        
    X = vectorizer.fit_transform(valid_sentences)
    sim_mat = cosine_similarity(X)
    
    # TextRank PageRank
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    
    ranked = sorted(((scores[i], s, i) for i, s in enumerate(valid_sentences)), reverse=True)
    top_sentences = sorted(ranked[:top_n], key=lambda x: x[2]) # restore order
    
    return " ".join([s[1] for s in top_sentences])
