import torch
import time
from torch.nn import CrossEntropyLoss
from transformers import Trainer, TrainingArguments
from .cah_data import CAHData

def train(model,data,learning_rate=1e-5,epochs=1,batch_size=8,comment=False):
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")

            outputs = model(**inputs)
            logits = outputs.get("logits")

            loss_fct = CrossEntropyLoss(weight=torch.tensor(data.class_weights().astype('float32'), device=model.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    model_name = f"E{epochs}-LR{learning_rate}-BS{batch_size}-{int(time.time())}"

    if comment:
        model_name = f"{model_name}-{comment}"

    logging_dir = f"results/runs/{model_name}"

    training_args = TrainingArguments(
        output_dir=f"./results/{model_name}",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        report_to="tensorboard",
        evaluation_strategy="steps",
        logging_steps=50,
        eval_steps=200,
        logging_dir=logging_dir,
    )

    datasets = data.datasets()

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['test'],
    )

    trainer.train()

    best_checkpoint = trainer.state.best_model_checkpoint
    print(best_checkpoint)

    return trainer

