import torch


class InferenceModule:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_summary(self, text):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=40,
            num_beams=4,
            early_stopping=True
        )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)