from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from summarizer.models import restore_sinusoidal_positions


class ModelModule:

    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.model_name = model_name

    def load_model(self):

        if not any(family in self.model_name for family in ("bart", "t5", "pegasus")):
            raise ValueError("Unsupported model")

        # Auto* picks the right tokenizer/model class from the checkpoint config
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        # PEGASUS scored ROUGE 0.0 in an earlier run: its position tables were left random
        restore_sinusoidal_positions(model)

        return tokenizer, model
