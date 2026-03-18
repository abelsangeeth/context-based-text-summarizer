from modules.model_module import ModelModule
from modules.inference_module import InferenceModule

text = """Artificial intelligence (AI) is rapidly transforming industries across the globe. In healthcare, AI-powered diagnostic tools are enabling earlier detection of diseases such as cancer and diabetes, improving patient outcomes significantly. In the financial sector, machine learning algorithms are being used for fraud detection, risk assessment, and automated trading strategies. The manufacturing industry is leveraging AI for predictive maintenance, quality control, and supply chain optimization. Furthermore, AI is revolutionizing transportation through the development of autonomous vehicles and smart traffic management systems. Despite these advances, concerns about job displacement, data privacy, and algorithmic bias remain significant challenges that need to be addressed through thoughtful regulation and ethical AI development practices."""

for model_name in ["facebook/bart-large-cnn", "t5-base", "google/pegasus-cnn_dailymail"]:
    print(f"\n=== {model_name} ===")
    mm = ModelModule(model_name)
    tok, mod = mm.load_model()
    inf = InferenceModule(mod, tok)
    summary = inf.generate_summary(text, model_name=model_name)
    print(f"Summary: [{summary}]")
    print(f"Length: {len(summary)} chars, {len(summary.split())} words")
