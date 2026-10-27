import torch


class InferenceModule:

    def __init__(self, model, tokenizer, model_name: str = ""):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name

    def generate_summary(self, text):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        # The original T5 checkpoints are multi-task: without the task prefix
        # they don't know they are being asked to summarize.
        if self.model_name.split("/")[-1].startswith("t5"):
            text = "summarize: " + text

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            min_length=40,
            num_beams=4,
            early_stopping=True
        )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
