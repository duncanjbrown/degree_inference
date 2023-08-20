from dotenv import load_dotenv
import pandas as pd
import openai
import uuid
import os

load_dotenv()

class OpenAIClient:
    def __init__(self, openai_key=os.getenv('OPENAI_API_KEY')):
        openai.api_key = openai_key
        self.prompt_token_usage = 0
        self.completion_token_usage = 0
        self.total_token_usage = 0

        random_uuid = uuid.uuid4()
        self.run_id = str(random_uuid).replace('-', '')[:10]

    def complete(self, degrees):
        answer = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                    {"role": "system", "content": self.system_prompt()},
                    {"role": "user", "content": self.user_prompt(degrees)},
                ]
            )

        self.prompt_token_usage += answer["usage"]["prompt_tokens"]
        self.completion_token_usage += answer["usage"]["completion_tokens"]
        self.total_token_usage += answer["usage"]["total_tokens"]

        return answer["choices"][0]["message"]["content"]

    def user_prompt(self, degrees):
        degrees = "\n".join(degrees)
        prompt = "Classify the following degrees according to the CAH. For each degree, return the degree and the CAH category in CSV format.\n" \
        f"{degrees}"

        return prompt

    def system_prompt(self):
        cah_mappings = pd.read_csv('./data/HECoS_CAH_Mappings.csv')
        cah3_mappings = cah_mappings[['CAH3_Code', 'CAH3_Label']].drop_duplicates()
        c = cah3_mappings.to_csv(sep="\t", index=False)

        string = "Your purpose is to classify degree subjects into the " \
        "Common Aggregation Heirarchy (CAH)." \
        "The following is a list of all the CAH codes. "\
        f"\n{c}" \
        "\nI will be asking you to classify lists of degree subjects according to this taxonomy."

        return string

    def stats(self):
        output_cost = 0.06 * (float(self.completion_token_usage) / 1000)
        input_cost = 0.03 * (float(self.prompt_token_usage) / 1000)

        return {
                "run_id": self.run_id,
                "prompt_token_usage": self.prompt_token_usage,
                "completion_token_usage": self.completion_token_usage,
                "total_token_usage": self.total_token_usage,
                "cost": round(output_cost + input_cost, 3)
                }


