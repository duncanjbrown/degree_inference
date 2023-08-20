import pytest
import json
from degree_inference.openai import OpenAIClient
import responses

@pytest.fixture
def completion_response():
    with open('test/fixtures/completion.json') as file:
        return json.load(file)

@pytest.fixture
def openai_with_network_stub(completion_response):
    url = "https://api.openai.com/v1/chat/completions"
    with responses.RequestsMock() as rsps:
        rsps.add(responses.POST, url, json=completion_response, status=200)
        openai = OpenAIClient(openai_key="fake_key")
        yield openai, rsps

def test_completion(openai_with_network_stub):
    client, rsps = openai_with_network_stub
    result = client.complete([])
    assert result == "This is a test"

def test_stats(openai_with_network_stub):
    client, rsps = openai_with_network_stub
    client.complete([])
    assert '"prompt_token_usage": 13, "completion_token_usage": 7, "total_token_usage": 20, "cost": 0.001}' in json.dumps(client.stats())

def test_logging_token_usage(openai_with_network_stub):
    client, rsps = openai_with_network_stub
    client.complete([])
    assert client.total_token_usage == 20
    assert client.prompt_token_usage == 13
    assert client.completion_token_usage == 7

def test_system_prompt(openai_with_network_stub):
    client, rsps = openai_with_network_stub
    client.complete([])
    assert "25-01-03\\tdesign studies" in rsps.calls[0].request.body.decode("utf-8")
    # assert "25-01-03\tdesign studies" in openai.system_prompt()

def test_user_prompt():
    openai = OpenAIClient()
    assert "mathematics\nenglish" in openai.user_prompt(['mathematics', 'english'])
