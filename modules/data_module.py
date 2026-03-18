from datasets import load_dataset


class DataModule:

    def __init__(self, dataset_name="cnn_dailymail", version="3.0.0"):
        self.dataset_name = dataset_name
        self.version = version

    def load_data(self, split="train", sample_size=None):
        dataset = load_dataset(self.dataset_name, self.version, split=split)

        if sample_size:
            dataset = dataset.select(range(sample_size))

        return dataset