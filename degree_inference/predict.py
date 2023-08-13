import torch

def predict(data,model,text_inputs):
    inputs = data.tokenizer(text_inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)

    inputs = inputs.to("cpu")
    outputs = model(**inputs)
    _predictions = torch.argmax(outputs.logits, dim=-1)

    predictions = data.encoder.inverse_transform(_predictions.cpu())
    predicted_categories = [data.cah3_mapping().get(key) for key in predictions]
    labelled_data = dict(zip(text_inputs, predicted_categories))

    return zip(text_inputs, list(predictions), predicted_categories)

