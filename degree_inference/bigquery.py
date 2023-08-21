import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()

class BigQueryClient:
    BASE_QUERY = """
        WITH free_text_degrees AS (
            SELECT
                cq.subject AS degree_subject
            FROM
                `rugged-abacus-218110.dataform_ABS_2_dev.application_choice_details`
            LEFT JOIN
                UNNEST(candidate_qualifications) AS cq
            LEFT JOIN `rugged-abacus-218110.dfe_reference_data.cah_categories_l3_v2` AS cah_codes ON cah_codes.id = degree_subject_cah_l3
            WHERE degree_level IS NOT NULL AND degree_level !='unknown'
            AND sd_unsubmitted IS FALSE
            AND nationality_group = "British"
            AND degree_subject_cah_l3 IS NULL
            GROUP BY
                degree_subject)
        """

    def __init__(self):
        self.client = bigquery.Client()

    def get_degrees(self):
        query = self.BASE_QUERY + " SELECT degree_subject FROM free_text_degrees"
        return list(self.client.query(query).to_dataframe().dropna().degree_subject)

    def count_degrees(self):
        query = self.BASE_QUERY + " SELECT COUNT(*) as count FROM free_text_degrees"
        return self.client.query(query).to_dataframe()['count'][0]
