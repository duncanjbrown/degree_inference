from dotenv import load_dotenv
import time
import json
import csv
import sys
import logging
from os import getenv

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

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

    def infer(self, outdir="gpt_output", n=200):
        outfile = f"{outdir}/{int(time.time())}.csv"

        with open(outfile, "w") as file:
            file.write("text,label\n")

            for chunk in self.degrees(n=n):
                logging.info(f"Writing to {outfile}")
                logging.info(f"Completing {len(chunk)} degrees...")
                file.write(self.openai_client.complete(chunk))
                logging.info(json.dumps(self.openai_client.stats()))

        return outfile

    def degrees(self, n=200, chunk_size=200):
        """Yield n degrees in chunks of chunk_size"""
        n_degrees = self.bq_client.get_degrees()[:n]
        # deal in chunks of 200 for 8k context window
        for i in range(0, len(n_degrees), chunk_size):
            yield n_degrees[i:i + chunk_size]
