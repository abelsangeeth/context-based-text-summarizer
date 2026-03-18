import torch
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq


class TrainingModule:

    def __init__(self, model, tokenizer, train_dataset, val_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train(self, output_dir="./results", epochs=3, batch_size=8):

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_total_limit=2,
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=50,
            fp16=torch.cuda.is_available(),
            load_best_model_at_end=True
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator
        )

        trainer.train()
        trainer.save_model("./final_model")
        self.tokenizer.save_pretrained("./final_model")

        return trainer