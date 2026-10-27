from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class ModelModule:

    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.model_name = model_name

    def load_model(self):

        if not any(family in self.model_name for family in ("bart", "t5", "pegasus")):
            raise ValueError("Unsupported model")

        # Auto* picks the right tokenizer/model class from the checkpoint config
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        return tokenizer, model
