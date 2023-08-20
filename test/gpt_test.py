import pytest
from dataclasses import dataclass
import pandas as pd

from degree_inference.gpt import GPT


class DummyBQClient:
    def __init__(self, degrees=["mathematics", "english"]):
        self.data = {"degree_subject": degrees}

    @dataclass
    class DummyBQQueryResponse:
        data: dict

        def to_dataframe(self):
            return pd.DataFrame(data=self.data)

    def query(self, query):
        return self.DummyBQQueryResponse(self.data)


class DummyOpenAIClient:
    def complete(self, degrees):
        mapped_degrees = [d + "-gpt" for d in degrees]
        return ','.join(mapped_degrees)


def test_integration():
    gpt = GPT(DummyOpenAIClient(), DummyBQClient(degrees=["science"]))
    assert gpt.infer() == "science-gpt"
