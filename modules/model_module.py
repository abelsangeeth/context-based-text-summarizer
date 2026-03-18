from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    T5Tokenizer,
    T5ForConditionalGeneration,
    PegasusTokenizer,
    PegasusForConditionalGeneration
)


class ModelModule:

    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.model_name = model_name

    def load_model(self):

        if "bart" in self.model_name:
            tokenizer = BartTokenizer.from_pretrained(self.model_name)
            model = BartForConditionalGeneration.from_pretrained(self.model_name, device_map=None)

        elif "t5" in self.model_name:
            tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            model = T5ForConditionalGeneration.from_pretrained(self.model_name, device_map=None)

        elif "pegasus" in self.model_name:
            tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
            model = PegasusForConditionalGeneration.from_pretrained(self.model_name, device_map=None)

        else:
            raise ValueError("Unsupported model")

        return tokenizer, model