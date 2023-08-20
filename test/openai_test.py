from degree_inference.openai import OpenAIClient

def test_system_prompt():
    openai = OpenAIClient()
    assert "25-01-03\tdesign studies" in openai.system_prompt()

def test_user_prompt():
    openai = OpenAIClient()
    assert "mathematics\nenglish" in openai.user_prompt(['mathematics', 'english'])
