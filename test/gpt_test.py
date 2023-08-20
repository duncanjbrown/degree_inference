import pytest
from pathlib import Path
from dataclasses import dataclass
import pandas as pd

from degree_inference.gpt import GPT


class DummyBQClient:
    def __init__(self, degrees):
        self.degrees = degrees

    def get_degrees(self):
        return self.degrees


class DummyOpenAIClient:
    def complete(self, degrees):
        mapped_degrees = [f"{d},{d}-gpt" for d in degrees]
        return '\n'.join(mapped_degrees)

    def stats(self):
        return "Some stats"


def test_integration(tmpdir):
    gpt = GPT(DummyOpenAIClient(), DummyBQClient(degrees=["science"]))
    outfile = gpt.infer(outdir=tmpdir)
    txt = Path(outfile).read_text()
    assert txt == "text,label\nscience,science-gpt"

