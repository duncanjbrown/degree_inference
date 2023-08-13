import torch
from torch.nn import CrossEntropyLoss
from transformers import Trainer, TrainingArguments
from .cah_data import CAHData

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = CrossEntropyLoss(weight=torch.tensor(CAHData().class_weights().astype('float32'), device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def train(model,data,learning_rate=1e-5,epochs=1):
    training_args = TrainingArguments(
        output_dir='./results',
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        report_to="tensorboard",
        evaluation_strategy="steps",
        logging_steps=50,
        eval_steps=100,
    )

    datasets = data.datasets()

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['test'],
    )

    trainer.train()
    return trainer

