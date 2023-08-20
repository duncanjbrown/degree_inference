from dotenv import load_dotenv
from os import getenv
import pandas as pd
import openai
from google.cloud import bigquery

from degree_inference import openai

load_dotenv()

class GPT:
    BASE_QUERY = """
        WITH (
            SELECT
                cq.subject AS degree_subject
            FROM
                `rugged-abacus-218110.dataform_ABS_2_dev.application_choice_details`
            LEFT JOIN
                UNNEST(candidate_qualifications) AS cq
            LEFT JOIN `rugged-abacus-218110.dfe_reference_data.cah_categories_l3_v2` AS cah_codes ON cah_codes.id = degree_subject_cah_l3
            WHERE degree_level IS NOT NULL AND degree_level !='unknown'
            AND sd_unsubmitted IS FALSE
            AND degree_subject_cah_l3 IS NULL
            GROUP BY
                degree_subject,
        ) AS free_text_degrees
        """

    def __init__(self, openai_client, bq_client):
        self.openai_client = openai_client
        self.bq_client = bq_client

    def infer(self):
        degrees = list(self.bq_client.query(self.degree_query()).to_dataframe().degree_subject)
        return self.openai_client.complete(degrees)

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def degree_query(self):
        return self.BASE_QUERY + " SELECT degree_subject FROM free_text_degrees"


# def available_rows(self):


def run():
    # openai.api_key = getenv('OPENAI_API_KEY')
    print(getenv('OPENAI_API_KEY'))
    exit

    cah_mappings = pd.read_csv('./data/HECoS_CAH_Mappings.csv')
    cah3_mappings = cah_mappings[['CAH3_Code', 'CAH3_Label']].drop_duplicates()
    c = cah3_mappings.to_csv(sep="\t")

    system_message = f"""
        Your purpose classify degree subjects into the Common Aggregation Heirarchy (CAH).
        The following is a list of all the CAH codes.

        {c}

        I will be asking you to classify lists of degree subjects according to this taxonomy.
        """



    def get_unmapped_degrees():
        client = bigquery.Client()
        query = """
        SELECT
            cq.subject AS degree_subject,
            cq.degree_subject_cah_l3,
            cah_codes.name AS cah_category_name
        FROM
            `rugged-abacus-218110.dataform_ABS_2_dev.application_choice_details`
        LEFT JOIN
            UNNEST(candidate_qualifications) AS cq
        LEFT JOIN `rugged-abacus-218110.dfe_reference_data.cah_categories_l3_v2` AS cah_codes ON cah_codes.id = degree_subject_cah_l3
        WHERE degree_level IS NOT NULL AND degree_level !='unknown'
        AND degree_subject_cah_l3 IS NULL
        GROUP BY
            degree_subject,
            degree_subject_cah_l3,
            cah_category_name
        LIMIT 200
        OFFSET 1000
        """

        return list(client.query(query).to_dataframe().degree_subject)

    def map_degrees(degrees):
        for chunk in chunks(degrees, 200):
            classification_query = (
                    "Classify the following degrees according to the CAH. For each degree, return the degree and the CAH category in CSV format.\n"
                    f"{chunk}"
            )

            answer = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": classification_query},
                    ]
                )

            yield answer["choices"][0]["message"]["content"]

    if __name__ == "__main__":
        for csv in map_degrees(get_unmapped_degrees()):
            print(csv)

