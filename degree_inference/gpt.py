from dotenv import load_dotenv
import time
import json
import csv
import sys
import logging
from os import getenv

logger = logging.getLogger()
logger.setLevel(logging.INFO)

log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('log/gpt.log')
file_handler.setFormatter(log_format)
logger.addHandler(file_handler)

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setFormatter(log_format)
logger.addHandler(stderr_handler)

class GPT:
    def __init__(self, openai_client, bq_client):
        self.openai_client = openai_client
        self.bq_client = bq_client

    def infer(self, outdir="gpt_output", n=200, chunk_size=200, offset=0):
        outfile = f"{outdir}/{int(time.time())}.csv"
        logging.info(f"Writing to {outfile}")

        with open(outfile, "w") as file:
            file.write("text,label\n")

            for chunk in self.degrees(n=n, chunk_size=chunk_size, offset=offset):
                logging.info(f"Completing {len(chunk)} degrees...")
                file.write(self.openai_client.complete(chunk))
                file.write("\n")
                logging.info(json.dumps(self.openai_client.stats()))

        return outfile

    def degrees(self, n=200, chunk_size=200, offset=0):
        """Yield n degrees in chunks of chunk_size"""
        n_degrees = self.bq_client.get_degrees()[offset:n]
        # deal in chunks of chunk_size for 8k context window
        for i in range(0, len(n_degrees), chunk_size):
            yield n_degrees[i:i + chunk_size]
