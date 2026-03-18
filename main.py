from modules.model_module import ModelModule
from modules.inference_module import InferenceModule

text = "Artificial intelligence is transforming industries."

model_module = ModelModule("facebook/bart-large-cnn")
tokenizer, model = model_module.load_model()

inference = InferenceModule(model, tokenizer)
summary = inference.generate_summary(text)

print("Generated Summary:\n", summary)