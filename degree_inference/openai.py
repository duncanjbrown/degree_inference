from dotenv import load_dotenv
import pandas as pd
import openai

load_dotenv()

class OpenAIClient:
    def complete(self, degrees):
        pass

    def user_prompt(self, degrees):
        degrees = "\n".join(degrees)
        prompt = "Classify the following degrees according to the CAH. For each degree, return the degree and the CAH category in CSV format.\n" \
        f"{degrees}"

        return prompt

    def system_prompt(self):
        cah_mappings = pd.read_csv('./data/HECoS_CAH_Mappings.csv')
        cah3_mappings = cah_mappings[['CAH3_Code', 'CAH3_Label']].drop_duplicates()
        c = cah3_mappings.to_csv(sep="\t", index=False)

        string = "Your purpose is to classify degree subjects into the" \
        "Common Aggregation Heirarchy (CAH)." \
        "The following is a list of all the CAH codes."\
        f"\n{c}" \
        "\nI will be asking you to classify lists of degree subjects according to this taxonomy."

        return string

