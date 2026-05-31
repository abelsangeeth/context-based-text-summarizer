import json
from summarizer.longform_builder import build_longform_summary

sections = [{'index': 0, 'raw_text': 'Artificial intelligence (AI) is rapidly transforming industries across the globe. In healthcare, AI-powered diagnostic tools are enabling earlier detection of diseases such as cancer and diabetes, improving patient outcomes significantly. In the financial sector, machine learning algorithms are being used for fraud detection, risk assessment, and automated trading strategies. ' * 50, 'section_type': 'body'}]
subheadings = ['Introduction']

res = build_longform_summary(sections, subheadings, 'bart-cnn', 'standard')
with open("test_b_out.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("Done")
