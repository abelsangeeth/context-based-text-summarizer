class PreprocessingModule:

    def __init__(self, tokenizer, max_input_length=512, max_target_length=128):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def preprocess(self, dataset):

        def tokenize_function(example):
            model_inputs = self.tokenizer(
                example["article"],
                max_length=self.max_input_length,
                truncation=True,
                padding="max_length"
            )

            labels = self.tokenizer(
                example["highlights"],
                max_length=self.max_target_length,
                truncation=True,
                padding="max_length"
            )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        return dataset.map(tokenize_function, batched=True)