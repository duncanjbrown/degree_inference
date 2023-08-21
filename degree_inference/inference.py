from .predict import predict
import csv
import io
from transformers import BertForSequenceClassification
from .cah_data import CAHData

def run(model, inputs):
    model = BertForSequenceClassification.from_pretrained(model, num_labels=166)
    outputs = predict(CAHData(),model,inputs)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["input", "CAH3", "CAH3 description"])
    writer.writerows(outputs)

    return output.getvalue()
