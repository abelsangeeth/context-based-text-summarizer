import yake
from summarizer.pipeline import summarize

_kw_extractor = None

def generate_subheading(section_text: str, section_type: str, model_name: str = "t5-base") -> str:
    words = section_text.split()
    
    # Tier 1: Keyword extraction for short segments
    if len(words) < 100:
        global _kw_extractor
        if not _kw_extractor:
            _kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=3, features=None)
        kws = _kw_extractor.extract_keywords(section_text)
        if kws:
            combined = " ".join([k[0] for k in kws])
            return format_subheading(combined)
        return "Topic Overview"
        
    # Tier 2: AI Generation for dense text chunks
    if model_name.startswith("t5"):
        prompt = f"summarize: {section_text[:500]}"
    else:
        prompt = section_text[:500]
        
    heading = summarize(prompt, model_key=model_name, max_length=25, min_length=4)
    heading_words = heading.split()[:7] # Ensure it's not a huge paragraph
    heading = " ".join(heading_words).rstrip(",.-:; ")
        
    return format_subheading(heading)

def generate_all_subheadings(sections: list[dict], model_name: str = "t5-base") -> list[str]:
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
            subheadings.append(generate_subheading(sec["raw_text"], stype, model_name))
            
    return subheadings

def format_subheading(heading: str) -> str:
    clean = heading.strip().title()
    if clean.endswith(":") or clean.endswith("."):
        clean = clean[:-1]
        
    words = clean.split()
    if len(words) > 8:
        clean = " ".join(words[:8])
    return clean.strip()
