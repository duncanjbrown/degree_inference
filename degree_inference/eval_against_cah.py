import csv
import sys
import os
import pandas as pd
from transformers import BertForSequenceClassification

from .cah_data import CAHData
from .predict import predict

data = CAHData()
hecos_cah = pd.read_csv('data/HECoS_CAH_Mappings.csv')
xs = hecos_cah.CAH3_Label.tolist()
cah3_codes_x = hecos_cah.CAH3_Code.tolist()
cah2_codes_x = hecos_cah.CAH2_Code.tolist()
cah1_codes_x = hecos_cah.CAH1_Code.tolist()

def evaluate_against_cah(model_name, model):
    ys = predict(data,model,xs)

    cah3_codes_y = []
    for idx, (cah3_label, cah3_code, _cah3_category) in enumerate(ys):
        cah3_codes_y.append(cah3_code)

    hits = 0
    for idx, cah3_code_y in enumerate(cah3_codes_y):
        if cah3_code_y == cah3_codes_x[idx]:
            hits += 1

    cah3_correctly_mapped = round(float(hits) / float(len(xs)), 2)

    cah2_codes_y = [code[0:5] for code in cah3_codes_y]

    hits = 0
    for idx, cah2_code_y in enumerate(cah2_codes_y):
        if cah2_code_y == cah2_codes_x[idx]:
            hits += 1

    cah2_correctly_mapped = round(float(hits) / float(len(xs)), 2)

    cah1_codes_y = [int(code[0:2]) for code in cah3_codes_y]

    hits = 0
    for idx, cah1_code_y in enumerate(cah1_codes_y):
        if cah1_code_y == cah1_codes_x[idx]:
            hits += 1

    cah1_correctly_mapped = round(float(hits) / float(len(xs)), 2)

    return (cah3_correctly_mapped, cah2_correctly_mapped, cah1_correctly_mapped)

print("model_name,cah3_correctly_mapped,cah2_correctly_mapped,cah1_correctly_mapped")
for model_name in os.listdir('models'):
    model = BertForSequenceClassification.from_pretrained(f"./models/{model_name}", num_labels=166)
    model.to("cpu")
    cah3_correctly_mapped, cah2_correctly_mapped, cah1_correctly_mapped = evaluate_against_cah(model_name, model)
    print(f"{model_name},{cah3_correctly_mapped},{cah2_correctly_mapped},{cah1_correctly_mapped}")

# writer = csv.writer(sys.stdout)
# writer.writerow(["Degree name", "CAH3 code", "CAH3 category"])
# for input, cah3_code in rows:
#     writer.writerow(row)
# #
