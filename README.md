# CAH3 code inference model

This repository contains code to train and use a [BERT
model](https://en.wikipedia.org/wiki/BERT_(language_model)) to
assign [CAH3](https://www.hesa.ac.uk/support/documentation/hecos/cah)
codes to degrees.

## Running locally

Start by setting up a virtual env and installing dependencies.

    python -m venv .venv
    . ./.venv/bin/activate
    pip install -r requirements.txt

To train the model run the cells in the `train.ipynb` file via
`jupyter notebook`. Training took about an hour on my MacBook pro. You
may need to adjust the `model.to()` call if your system doesn't support
[Metal Performance
Shaders](https://huggingface.co/docs/accelerate/usage_guides/mps).

Run `tensorboard` while training to see graphs.

Once the model is trained you can save it and load it --- see the last
cell in the notebook, and `app.py`, for examples.

Then to run inference on a set of example degrees hardcoded in the
`app.py` file, run `python app.py`.

The output will be a CSV with columns for degree name (the input), CAH3
code (the output) and the human-readable CAH3 category.

## Training data

The `/data` folder contains the training data:

- manually mapped codes from ILR, which require some cleaning (see
  `CAHData`)
- the original HECoS \> CAH mapping which exhaustively lists
  "official" degrees
- 2000 rows of mappings from "unofficial" degrees, produced by GPT-4

### Todo

- Accept STDIN/file input, not just hardcoded strings in the file
- and downcase input so it works better with the model
- A nice way to organise training run logs and file away models
- accept additional input fields so we can map to IDs
- env var for model location
