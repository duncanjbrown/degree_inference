import csv
import sys
from transformers import BertForSequenceClassification

from degree_inference.cah_data import CAHData
from degree_inference.predict import predict

degrees= [
        'mechanical engineering',
        'fashion design',
        'early childhood studies',
        'horticulture',
        'textile design',
        'geography',
        'developmental psychology',
        'history',
        'dance',
        'psychology',
        'pharmacy',
        'biology',
        'criminology',
        'chemical engineering',
        'philosophy',
        'health studies',
        'sociology',
]

data = CAHData()
model = BertForSequenceClassification.from_pretrained('./30-epoch-gpt2-ilr-augmented-1e-5/', num_labels=len(data.df['label'].unique()))
model.to("cpu")

rows = predict(data,model,degrees)

writer = csv.writer(sys.stdout)
writer.writerow(["Degree name", "CAH3 code", "CAH3 category"])
for row in rows:
    writer.writerow(row)

